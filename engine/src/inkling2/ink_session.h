// engine/src/inkling2/ink_session.h - Binary session SAVE/RESTORE + RAM LRU & SSD spill conversation cache for strata-inkling
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <random>
#include <string>
#include <vector>

#include "ink_backend.h"

#ifdef _WIN32
#include <process.h>
#define INK_GETPID() _getpid()
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#define INK_GETPID() getpid()
#endif

namespace ink2 {

// FNV-1a 64-bit helper for deterministic model/config session fingerprints and payload checksums
struct InkFnv64 {
    uint64_t h = 14695981039346656037ULL;
    void feed(const void* p, size_t n) noexcept {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    }
    void feed_str(const std::string& s) noexcept {
        const uint64_t len = s.size();
        feed(&len, sizeof(len));
        if (!s.empty()) feed(s.data(), s.size());
    }
    void feed_u64(uint64_t v) noexcept { feed(&v, sizeof(v)); }
};

inline uint64_t ink_model_fingerprint(const std::string& model_path, const InkCfg& c) {
    InkFnv64 f;
    f.feed_str("inkling2-model-v1");
    std::filesystem::path p(model_path);
    f.feed_str(p.filename().string());
    std::error_code ec;
    const auto sz = std::filesystem::file_size(p, ec);
    f.feed_u64(ec ? 0ULL : static_cast<uint64_t>(sz));
    f.feed_u64(static_cast<uint64_t>(c.n_phys));
    f.feed_u64(static_cast<uint64_t>(c.n_visits));
    f.feed_u64(static_cast<uint64_t>(c.hidden));
    f.feed_u64(static_cast<uint64_t>(c.vocab_unpadded));
    return f.h;
}

inline uint64_t ink_config_fingerprint(const std::string& version, KvDtype kv_dtype, const InkCfg& c) {
    InkFnv64 f;
    f.feed_str("inkling2-cfg-v1");
    f.feed_str(version);
    f.feed_u64(kv_dtype == KvDtype::FP8_E4M3 ? 1ULL : 0ULL);
    f.feed_u64(static_cast<uint64_t>(c.swa_win));
    return f.h;
}

#pragma pack(push, 1)
struct InkSessionFileHeader {
    char     magic[8];        // "INKSESS1"
    uint32_t version;         // 1
    uint32_t kv_is_fp8;       // 0 = FP16, 1 = FP8_E4M3
    uint64_t model_fp;
    uint64_t config_fp;
    uint32_t n_tokens;
    uint32_t swa_len;
    uint32_t swa_start;
    uint32_t reserved0;
    uint64_t payload_bytes;
    uint64_t payload_fnv64;
};
#pragma pack(pop)
static_assert(sizeof(InkSessionFileHeader) == 64, "InkSessionFileHeader must be 64 bytes");

inline bool ink_session_write(const std::string& path, const InkSlotSnapshot& snap,
                              uint64_t& out_bytes, std::string& err, std::string& err_kind) {
    out_bytes = 0;
    if (snap.n_tokens <= 0 || static_cast<int>(snap.tokens.size()) != snap.n_tokens) {
        err_kind = "invalid";
        err = "empty conversation snapshot";
        return false;
    }
    InkSessionFileHeader hdr{};
    std::memcpy(hdr.magic, "INKSESS1", 8);
    hdr.version       = 1;
    hdr.kv_is_fp8     = static_cast<uint32_t>(snap.kv_is_fp8 ? 1 : 0);
    hdr.model_fp      = snap.model_fp;
    hdr.config_fp     = snap.config_fp;
    hdr.n_tokens      = static_cast<uint32_t>(snap.n_tokens);
    hdr.swa_len       = static_cast<uint32_t>(snap.swa_len);
    hdr.swa_start     = static_cast<uint32_t>(snap.swa_start);
    hdr.payload_bytes = static_cast<uint64_t>(snap.payload.size());
    InkFnv64 chk;
    chk.feed(snap.tokens.data(), snap.tokens.size() * sizeof(int32_t));
    if (!snap.payload.empty()) chk.feed(snap.payload.data(), snap.payload.size());
    hdr.payload_fnv64 = chk.h;

    std::error_code ec;
    std::filesystem::path dst(path);
    if (dst.has_parent_path()) {
        std::filesystem::create_directories(dst.parent_path(), ec);
    }
    std::filesystem::path tmp = dst;
    tmp += ".tmp." + std::to_string(static_cast<long long>(INK_GETPID()));

    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) {
            err_kind = "io";
            err = "cannot open session file for writing: " + tmp.string();
            return false;
        }
        os.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        os.write(reinterpret_cast<const char*>(snap.tokens.data()),
                 static_cast<std::streamsize>(snap.tokens.size() * sizeof(int32_t)));
        if (!snap.payload.empty()) {
            os.write(reinterpret_cast<const char*>(snap.payload.data()),
                     static_cast<std::streamsize>(snap.payload.size()));
        }
        os.flush();
        if (!os) {
            os.close();
            std::filesystem::remove(tmp, ec);
            err_kind = "storage";
            err = "write failed (disk full or I/O error): " + tmp.string();
            return false;
        }
    }

    std::filesystem::rename(tmp, dst, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        err_kind = "io";
        err = "cannot publish session file: " + ec.message();
        return false;
    }
    out_bytes = sizeof(hdr) + snap.tokens.size() * sizeof(int32_t) + snap.payload.size();
    return true;
}

inline bool ink_session_read(const std::string& path, uint64_t expect_model_fp, uint64_t expect_config_fp,
                             int expect_kv_fp8, int max_ctx, InkSlotSnapshot& out,
                             uint64_t& out_bytes, std::string& err, std::string& err_kind) {
    out = {};
    out_bytes = 0;
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        err_kind = "io";
        err = "cannot open session file: " + path;
        return false;
    }
    InkSessionFileHeader hdr{};
    is.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!is || std::memcmp(hdr.magic, "INKSESS1", 8) != 0 || hdr.version != 1) {
        err_kind = "invalid";
        err = "not a valid INKSESS1 session file";
        return false;
    }
    if (hdr.model_fp != expect_model_fp || hdr.config_fp != expect_config_fp ||
        static_cast<int>(hdr.kv_is_fp8) != (expect_kv_fp8 ? 1 : 0)) {
        err_kind = "invalid";
        err = "session file fingerprint does not match the loaded model/config";
        return false;
    }
    if (hdr.n_tokens == 0 || static_cast<int>(hdr.n_tokens) >= max_ctx) {
        err_kind = "invalid";
        err = "session token count out of range for current --max-context";
        return false;
    }
    out.model_fp  = hdr.model_fp;
    out.config_fp = hdr.config_fp;
    out.n_tokens  = static_cast<int32_t>(hdr.n_tokens);
    out.kv_is_fp8 = static_cast<int32_t>(hdr.kv_is_fp8);
    out.swa_len   = static_cast<int32_t>(hdr.swa_len);
    out.swa_start = static_cast<int32_t>(hdr.swa_start);
    out.tokens.resize(hdr.n_tokens);
    is.read(reinterpret_cast<char*>(out.tokens.data()),
            static_cast<std::streamsize>(hdr.n_tokens * sizeof(int32_t)));
    out.payload.resize(static_cast<size_t>(hdr.payload_bytes));
    if (hdr.payload_bytes > 0) {
        is.read(reinterpret_cast<char*>(out.payload.data()),
                static_cast<std::streamsize>(hdr.payload_bytes));
    }
    if (!is) {
        err_kind = "invalid";
        err = "truncated session file";
        return false;
    }
    InkFnv64 chk;
    chk.feed(out.tokens.data(), out.tokens.size() * sizeof(int32_t));
    if (!out.payload.empty()) chk.feed(out.payload.data(), out.payload.size());
    if (chk.h != hdr.payload_fnv64) {
        err_kind = "invalid";
        err = "session file checksum mismatch";
        return false;
    }
    out_bytes = sizeof(hdr) + out.tokens.size() * sizeof(int32_t) + out.payload.size();
    return true;
}

// Native Strata-style two-tier conversation cache (RAM LRU + ephemeral per-PID SSD spill folder)
class InkConversationCache {
public:
    struct RamEntry {
        InkSlotSnapshot snap;
    };
    struct SpillEntry {
        std::vector<int32_t> tokens;
        std::string path;
        uint64_t bytes = 0;
    };

    void configure(int64_t ram_mib, int64_t spill_mib, const std::string& spill_root,
                   uint64_t model_fp, uint64_t config_fp, int kv_is_fp8, int max_ctx) {
        ram_budget_   = (ram_mib > 0) ? (static_cast<uint64_t>(ram_mib) << 20) : 0ULL;
        spill_budget_ = (spill_mib > 0) ? (static_cast<uint64_t>(spill_mib) << 20) : 0ULL;
        model_fp_     = model_fp;
        config_fp_    = config_fp;
        kv_is_fp8_    = kv_is_fp8;
        max_ctx_      = max_ctx;

        if (spill_budget_ > 0 && !spill_root.empty()) {
            std::error_code ec;
            std::filesystem::path root(spill_root);
            std::filesystem::create_directories(root, ec);
            if (!ec) {
                // Remove stale ended-PID spill directories from previous runs
                size_t cleaned = 0;
                for (const auto& de : std::filesystem::directory_iterator(root, ec)) {
                    if (!de.is_directory()) continue;
                    const std::string name = de.path().filename().string();
                    const size_t dash = name.find('-');
                    if (dash == std::string::npos) continue;
                    const long long pid = std::atoll(name.substr(0, dash).c_str());
                    if (pid <= 0) continue;
#ifndef _WIN32
                    if (::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH) {
                        std::filesystem::remove_all(de.path(), ec);
                        if (!ec) ++cleaned;
                    }
#endif
                }
                std::random_device rd;
                uint64_t nonce = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
                char dname[64];
                std::snprintf(dname, sizeof(dname), "%lld-%016llx",
                              static_cast<long long>(INK_GETPID()),
                              static_cast<unsigned long long>(nonce));
                spill_dir_ = root / dname;
                std::filesystem::create_directories(spill_dir_, ec);
                if (ec) {
                    spill_dir_.clear();
                    spill_budget_ = 0;
                } else {
                    std::fprintf(stderr,
                                 "strata-inkling serve: conversation cache: RAM up to %llu MiB, SSD spill in %s up to %llu MiB%s\n",
                                 static_cast<unsigned long long>(ram_budget_ >> 20),
                                 spill_dir_.string().c_str(),
                                 static_cast<unsigned long long>(spill_budget_ >> 20),
                                 cleaned ? (", " + std::to_string(cleaned) + " ended folder(s) removed").c_str() : "");
                }
            } else {
                spill_budget_ = 0;
            }
        } else if (ram_budget_ > 0) {
            std::fprintf(stderr, "strata-inkling serve: conversation cache: RAM up to %llu MiB (SSD spill off)\n",
                         static_cast<unsigned long long>(ram_budget_ >> 20));
        }
    }

    ~InkConversationCache() {
        if (!spill_dir_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(spill_dir_, ec);
        }
    }

    bool enabled() const noexcept { return ram_budget_ > 0 || spill_budget_ > 0; }

    // Park a conversation snapshot into RAM (evicting LRU to SSD spill if needed, or spilling directly if > ram_budget)
    void park(InkSlotSnapshot&& snap) {
        if (!enabled() || snap.n_tokens < 64) return;
        const uint64_t bytes = snap.total_bytes();
        // Remove any existing duplicate with identical tokens
        for (auto it = ram_.begin(); it != ram_.end(); ) {
            if (it->snap.tokens == snap.tokens) {
                ram_used_ -= it->snap.total_bytes();
                it = ram_.erase(it);
            } else {
                ++it;
            }
        }
        if (ram_budget_ > 0 && bytes <= ram_budget_) {
            while (!ram_.empty() && ram_used_ + bytes > ram_budget_) {
                RamEntry victim = std::move(ram_.back());
                ram_.pop_back();
                ram_used_ -= victim.snap.total_bytes();
                spill_write(victim.snap);
            }
            ram_used_ += bytes;
            ram_.push_front(RamEntry{std::move(snap)});
        } else if (spill_budget_ > 0) {
            spill_write(snap);
        }
    }

    // Find an exact prefix match (entire cached snapshot is a prefix of prompt_ids[0..prompt_len-2])
    // Returns true and populates `out` if a match was found (checking RAM first, then SSD spill).
    bool lookup(const std::vector<int32_t>& prompt_ids, InkSlotSnapshot& out, std::string& tier_out) {
        if (!enabled() || prompt_ids.size() < 2) return false;
        const size_t max_match = prompt_ids.size() - 1;

        // 1. Check RAM tier (longest exact prefix match)
        auto best_ram = ram_.end();
        size_t best_ram_len = 0;
        for (auto it = ram_.begin(); it != ram_.end(); ++it) {
            const auto& t = it->snap.tokens;
            if (t.size() <= max_match && t.size() > best_ram_len &&
                std::memcmp(t.data(), prompt_ids.data(), t.size() * sizeof(int32_t)) == 0) {
                best_ram = it;
                best_ram_len = t.size();
            }
        }
        if (best_ram != ram_.end()) {
            out = best_ram->snap;
            // Move to front (MRU)
            if (best_ram != ram_.begin()) ram_.splice(ram_.begin(), ram_, best_ram);
            tier_out = "RAM";
            return true;
        }

        // 2. Check SSD spill tier (longest exact prefix match)
        auto best_sp = spill_.end();
        size_t best_sp_len = 0;
        for (auto it = spill_.begin(); it != spill_.end(); ++it) {
            const auto& t = it->tokens;
            if (t.size() <= max_match && t.size() > best_sp_len &&
                std::memcmp(t.data(), prompt_ids.data(), t.size() * sizeof(int32_t)) == 0) {
                best_sp = it;
                best_sp_len = t.size();
            }
        }
        if (best_sp != spill_.end()) {
            uint64_t rbytes = 0;
            std::string err, kind;
            if (ink_session_read(best_sp->path, model_fp_, config_fp_, kv_is_fp8_, max_ctx_, out, rbytes, err, kind)) {
                if (best_sp != spill_.begin()) spill_.splice(spill_.begin(), spill_, best_sp);
                tier_out = "SSD-spill";
                return true;
            }
            // Drop corrupt/missing spill file
            std::error_code ec;
            std::filesystem::remove(best_sp->path, ec);
            spill_used_ -= best_sp->bytes;
            spill_.erase(best_sp);
        }
        return false;
    }

private:
    void spill_write(const InkSlotSnapshot& snap) {
        if (spill_budget_ == 0 || spill_dir_.empty()) return;
        const uint64_t est = snap.total_bytes() + sizeof(InkSessionFileHeader);
        if (est > spill_budget_) return;
        // Evict oldest spill entries to make room
        while (!spill_.empty() && spill_used_ + est > spill_budget_) {
            std::error_code ec;
            std::filesystem::remove(spill_.back().path, ec);
            spill_used_ -= spill_.back().bytes;
            spill_.pop_back();
        }
        char fname[64];
        std::snprintf(fname, sizeof(fname), "spill-%llu.bin", static_cast<unsigned long long>(++spill_seq_));
        const std::string p = (spill_dir_ / fname).string();
        uint64_t wbytes = 0;
        std::string err, kind;
        if (ink_session_write(p, snap, wbytes, err, kind)) {
            spill_used_ += wbytes;
            spill_.push_front(SpillEntry{snap.tokens, p, wbytes});
        }
    }

    uint64_t ram_budget_   = 0;
    uint64_t ram_used_     = 0;
    uint64_t spill_budget_ = 0;
    uint64_t spill_used_   = 0;
    uint64_t spill_seq_    = 0;
    uint64_t model_fp_     = 0;
    uint64_t config_fp_    = 0;
    int      kv_is_fp8_    = 0;
    int      max_ctx_      = 32768;
    std::filesystem::path spill_dir_;
    std::list<RamEntry>   ram_;
    std::list<SpillEntry> spill_;
};

}  // namespace ink2
