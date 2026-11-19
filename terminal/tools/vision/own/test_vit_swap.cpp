// tools/vision/own/test_vit_swap.cpp - GPU unit test for VitGpu host-RAM <-> VRAM weight swapping (`--swap-vram`).
//
// Builds a synthetic 2-layer Qwen3-VL ViT GGUF (BF16 matrices + F32 norms/biases), encodes a test image on GPU,
// stashes weights into host RAM + frees GPU memory (`enable_swap`), wakes back onto GPU (`wake`), and checks
// 100% bit-for-bit identical output rows across multiple sleep/wake cycles and auto-wake on encode.
#include "image_prep.h"
#include "own_reader.h"
#include "vit_geometry.h"
#include "vit_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace strata_vit;

static int g_checks = 0;
static int g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

namespace {

struct GgufBuf {
    std::vector<uint8_t> b;
    template <class T> void put(T v) {
        const uint8_t* p = (const uint8_t*) &v;
        b.insert(b.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) {
        put<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void raw(const void* p, size_t n) {
        const uint8_t* c = (const uint8_t*) p;
        b.insert(b.end(), c, c + n);
    }
    void pad32() {
        while (b.size() % 32 != 0) b.push_back(0);
    }
};

static uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

struct TensorSpec {
    std::string name;
    std::vector<int64_t> ne;
    uint32_t type = 0;
    std::vector<uint8_t> data;
};

static TensorSpec make_f32(const std::string& name, std::vector<int64_t> ne, float scale, int seed) {
    size_t n = 1;
    for (int64_t d : ne) n *= (size_t) d;
    std::vector<float> vals(n);
    for (size_t i = 0; i < n; ++i) {
        vals[i] = scale * std::sin(0.037f * (float) (i + (size_t) seed * 17));
    }
    TensorSpec t;
    t.name = name;
    t.ne = std::move(ne);
    t.type = GT_F32;
    t.data.resize(n * 4);
    std::memcpy(t.data.data(), vals.data(), n * 4);
    return t;
}

static TensorSpec make_bf16(const std::string& name, std::vector<int64_t> ne, float scale, int seed) {
    size_t n = 1;
    for (int64_t d : ne) n *= (size_t) d;
    std::vector<uint16_t> vals(n);
    for (size_t i = 0; i < n; ++i) {
        const float f = scale * std::cos(0.029f * (float) (i + (size_t) seed * 31));
        vals[i] = f32_to_bf16(f);
    }
    TensorSpec t;
    t.name = name;
    t.ne = std::move(ne);
    t.type = GT_BF16;
    t.data.resize(n * 2);
    std::memcpy(t.data.data(), vals.data(), n * 2);
    return t;
}

static bool write_synthetic_vit_gguf(const std::string& path) {
    const int64_t layers = 2, W = 64, heads = 4, F = 128, patch = 4, merge = 2, pos_grid = 8, out_dim = 32;
    const int64_t merger_in = W * merge * merge;
    const int64_t merger_hidden = 96;

    std::vector<TensorSpec> ts;
    ts.push_back(make_f32("v.patch_embd.weight", {patch, patch, 3, W}, 0.05f, 1));
    ts.push_back(make_f32("v.patch_embd.weight.1", {patch, patch, 3, W}, 0.03f, 2));
    ts.push_back(make_f32("v.patch_embd.bias", {W}, 0.01f, 3));
    ts.push_back(make_f32("v.position_embd.weight", {W, pos_grid * pos_grid}, 0.02f, 4));

    for (int il = 0; il < (int) layers; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        ts.push_back(make_f32(p + "ln1.weight", {W}, 0.9f, 10 + il));
        ts.push_back(make_f32(p + "ln1.bias", {W}, 0.01f, 20 + il));
        ts.push_back(make_f32(p + "ln2.weight", {W}, 0.9f, 30 + il));
        ts.push_back(make_f32(p + "ln2.bias", {W}, 0.01f, 40 + il));
        ts.push_back(make_bf16(p + "attn_qkv.weight", {W, 3 * W}, 0.08f, 50 + il));
        ts.push_back(make_f32(p + "attn_qkv.bias", {3 * W}, 0.01f, 60 + il));
        ts.push_back(make_bf16(p + "attn_out.weight", {W, W}, 0.08f, 70 + il));
        ts.push_back(make_f32(p + "attn_out.bias", {W}, 0.01f, 80 + il));
        ts.push_back(make_bf16(p + "ffn_up.weight", {W, F}, 0.07f, 90 + il));
        ts.push_back(make_f32(p + "ffn_up.bias", {F}, 0.01f, 100 + il));
        ts.push_back(make_bf16(p + "ffn_down.weight", {F, W}, 0.07f, 110 + il));
        ts.push_back(make_f32(p + "ffn_down.bias", {W}, 0.01f, 120 + il));
    }

    ts.push_back(make_f32("v.post_ln.weight", {W}, 1.0f, 200));
    ts.push_back(make_f32("v.post_ln.bias", {W}, 0.01f, 201));
    ts.push_back(make_bf16("mm.0.weight", {merger_in, merger_hidden}, 0.06f, 202));
    ts.push_back(make_f32("mm.0.bias", {merger_hidden}, 0.01f, 203));
    ts.push_back(make_bf16("mm.2.weight", {merger_hidden, out_dim}, 0.06f, 204));
    ts.push_back(make_f32("mm.2.bias", {out_dim}, 0.01f, 205));

    GgufBuf g;
    g.raw("GGUF", 4);
    g.put<uint32_t>(3);
    g.put<uint64_t>(ts.size());
    g.put<uint64_t>(11);  // KV count

    auto kv_str = [&](const char* k, const char* v) { g.str(k); g.put<uint32_t>(8); g.str(v); };
    auto kv_bool = [&](const char* k, bool v) { g.str(k); g.put<uint32_t>(7); g.put<uint8_t>(v ? 1 : 0); };
    auto kv_u32 = [&](const char* k, uint32_t v) { g.str(k); g.put<uint32_t>(4); g.put<uint32_t>(v); };

    kv_str("general.architecture", "clip");
    kv_bool("clip.has_vision_encoder", true);
    kv_str("clip.projector_type", "qwen3vl_merger");
    kv_bool("clip.use_gelu", true);
    kv_u32("clip.vision.block_count", (uint32_t) layers);
    kv_u32("clip.vision.embedding_length", (uint32_t) W);
    kv_u32("clip.vision.attention.head_count", (uint32_t) heads);
    kv_u32("clip.vision.feed_forward_length", (uint32_t) F);
    kv_u32("clip.vision.patch_size", (uint32_t) patch);
    kv_u32("clip.vision.spatial_merge_size", (uint32_t) merge);
    kv_u32("clip.vision.projection_dim", (uint32_t) out_dim);

    uint64_t off = 0;
    for (const TensorSpec& t : ts) {
        g.str(t.name);
        g.put<uint32_t>((uint32_t) t.ne.size());
        for (int64_t d : t.ne) g.put<uint64_t>((uint64_t) d);
        g.put<uint32_t>(t.type);
        g.put<uint64_t>(off);
        off += (t.data.size() + 31u) / 32u * 32u;
    }
    g.pad32();
    for (const TensorSpec& t : ts) {
        g.raw(t.data.data(), t.data.size());
        g.pad32();
    }

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = std::fwrite(g.b.data(), 1, g.b.size(), fp) == g.b.size();
    std::fclose(fp);
    return ok;
}

}  // namespace

int main() {
    std::string why;
    if (!VitGpu::available(why)) {
        std::printf("strata-vision-own-swap-test: skipped (%s)\n", why.c_str());
        return 0;
    }

    const char* tmp_dir = std::getenv("TMPDIR");
#if !defined(_WIN32)
    const long pid = (long) getpid();
#else
    const long pid = (long) std::rand();
#endif
    const std::string path = std::string(tmp_dir ? tmp_dir : "/tmp") + "/strata_vit_swap_test_" + std::to_string(pid) + ".gguf";
    CHECK(write_synthetic_vit_gguf(path));

    {
        Gguf gg;
        VitGeometry geo;
        std::string err;
        CHECK(gg.open(path, err));
        CHECK(vit_geometry_from_gguf(gg, geo, err));
        CHECK(vit_geometry_supported(geo, err));

        VitGpu gpu;
        CHECK(gpu.load(gg, geo, 0, 64, err));
        CHECK(!gpu.is_sleeping());
        CHECK(gpu.device_bytes() > 0);

        PreparedImage img;
        img.w = 32;
        img.h = 32;
        img.chw.resize((size_t) 3 * img.w * img.h);
        for (size_t i = 0; i < img.chw.size(); ++i) {
            img.chw[i] = 0.25f * std::sin(0.013f * (float) i);
        }

        std::vector<float> baseline, after_wake, after_autowake;
        int nx = 0, ny = 0;
        CHECK(gpu.encode(img, baseline, nx, ny, err));
        CHECK(nx == 4 && ny == 4);
        CHECK(baseline.size() == (size_t) (nx * ny * geo.out_dim));

        CHECK(gpu.enable_swap(err));
        CHECK(gpu.is_sleeping());
        CHECK(gpu.device_bytes() == 0);
        CHECK(gpu.swap_slab_bytes() > 0);

        double wake_ms = -1.0;
        CHECK(gpu.wake(wake_ms, err));
        CHECK(wake_ms >= 0.0);
        CHECK(!gpu.is_sleeping());
        CHECK(gpu.device_bytes() == gpu.swap_slab_bytes());

        int nx2 = 0, ny2 = 0;
        CHECK(gpu.encode(img, after_wake, nx2, ny2, err));
        CHECK(nx2 == nx && ny2 == ny);
        CHECK(after_wake.size() == baseline.size());
        if (after_wake.size() == baseline.size() &&
            std::memcmp(after_wake.data(), baseline.data(), baseline.size() * sizeof(float)) != 0) {
            float max_diff = 0.0f;
            for (size_t i = 0; i < baseline.size(); ++i)
                max_diff = std::max(max_diff, std::fabs(after_wake[i] - baseline[i]));
            std::printf("diag after_wake: baseline[0..2]=(%g,%g,%g) after_wake[0..2]=(%g,%g,%g) max_diff=%g\n",
                        baseline[0], baseline[1], baseline[2], after_wake[0], after_wake[1], after_wake[2], max_diff);
        }
        CHECK(std::memcmp(after_wake.data(), baseline.data(), baseline.size() * sizeof(float)) == 0);

        // Idempotent sleep + auto-wake inside encode()
        CHECK(gpu.sleep());
        CHECK(gpu.sleep());
        CHECK(gpu.is_sleeping());
        CHECK(gpu.device_bytes() == 0);

        int nx3 = 0, ny3 = 0;
        CHECK(gpu.encode(img, after_autowake, nx3, ny3, err));
        CHECK(!gpu.is_sleeping());
        CHECK(after_autowake.size() == baseline.size());
        CHECK(std::memcmp(after_autowake.data(), baseline.data(), baseline.size() * sizeof(float)) == 0);
    }

    std::remove(path.c_str());
    if (g_fail) {
        std::printf("strata-vision-own-swap-test: %d FAILED of %d checks\n", g_fail, g_checks);
        return 1;
    }
    std::printf("strata-vision-own-swap-test: %d checks passed\n", g_checks);
    return 0;
}
