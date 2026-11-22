// engine/src/inkling2/ink_backend.h - Pluggable Backend ABI for strata-inkling (CPU / CUDA / HIP / Metal / SYCL)
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ink_batch.h"
#include "ink_manifest.h"

namespace ink2 {

struct ImageRec;  // defined in ink_vision.h

struct BackendOpts {
    int max_ctx = 32768;
    int parallel = 1;            // 1..64 concurrent slots
    int chunk = 512;
    int gpu_index = 0;
    int gpus = 1;                // 1 or 2 (--gpus 2 activates Dual-GPU Expert Parallelism, #6506)
    double ep_split = 0.5;       // GPU 1 expert share in (0, 1) (--ep-split / IK2_EP_SPLIT)
    int vram_reserve_mib = -1;   // >= 0 reserves N MiB free VRAM & activates hybrid offload (#6520)
    int cpu_threads = 0;         // > 0 activates CUDA-Graph mapped-memory doorbell NUMA CPU expert pool (#6520)
    int threads = 0;             // 0 = auto
    double vram_experts_gb = -1.0; // <0 = auto (3x loop-weighted vram_load_order!)
    double ram_experts_gb = -1.0;  // <0 = auto
    KvDtype kv_dtype = KvDtype::FP16;
    bool kv_dtype_auto = true;   // auto selects FP8_E4M3 when parallel >= 16 on <= 32 GiB GPU
    int kv_resident = -1;        // >= 0 caps global KV tokens in VRAM; overflow spills to mapped pinned host RAM (-1 = auto)
    bool enable_dspark = true;
    std::string dspark_path;     // optional external mtp.safetensors or dspark-*.gguf
    std::string vision_path;
};

struct BackendStats {
    size_t vram_experts_loaded = 0;
    size_t ram_experts_loaded = 0;
    size_t total_experts = 0;
    size_t vram_used_bytes = 0;
    size_t vram_total_bytes = 0;
    int active_slots = 0;
    int max_slots = 1;
    KvDtype kv_dtype = KvDtype::FP16;
};

struct InkSlotSnapshot {
    uint64_t model_fp = 0;
    uint64_t config_fp = 0;
    int32_t  n_tokens = 0;
    int32_t  kv_is_fp8 = 0;
    int32_t  swa_len = 0;
    int32_t  swa_start = 0;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> payload;

    size_t total_bytes() const noexcept {
        return tokens.size() * sizeof(int32_t) + payload.size();
    }
};

class InkBackend {
public:
    virtual ~InkBackend() = default;

    virtual const char* backend_name() const noexcept = 0;
    virtual std::string device_info() const = 0;
    virtual BackendStats stats() const = 0;

    // Load Section 1 trunk + DSpark + 3x loop-weighted Section 4 experts into device/host memory
    virtual bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) = 0;

    // Slot lifecycle (0 <= slot < opts.parallel)
    virtual bool alloc_slot(int slot, int max_ctx, KvDtype kv_dtype, std::string& err) = 0;
    virtual void free_slot(int slot) = 0;
    virtual void reset_slot(int slot) = 0;
    // Speculative rollback: restore KV + ShortConv state to `keep_len` (requires snapshot taken before verify!)
    virtual void snapshot_slot(int slot) = 0;
    virtual void rollback_slot(int slot, int keep_len) = 0;

    // Batched 48-visit forward across 1..64 active slots!
    // logits_out: [batch.count_logits(), vocab_unpadded] FP32 (already softcapped by 30*tanh(z/30) and sliced to 200058).
    virtual bool forward_batch(
        const StepBatch& batch,
        float* logits_out,
        TapCapture* taps_out,
        std::string& err) = 0;

    // On-device DSpark Block-8 Recurrent Markov drafting for one slot.
    // Returns number of drafted tokens (0..max_draft, typically up to 7); returns 0 if unavailable/disabled.
    virtual int dspark_draft(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_draft,
        int32_t* out_draft_ids,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_draft; (void)out_draft_ids; (void)err;
        return 0;
    }

    // Optional fused on-device greedy speculative step (W=4 Adaptive Beam Tree + zero-logits-D2H verify + commit).
    // Returns n_emit in 1..min(8, max_emit) on success, 0 to fall back to dspark_draft + forward_batch, or -1 on error.
    virtual int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_emit;
        (void)out_emitted; (void)out_drafted; (void)err;
        return 0;
    }

    // Optional GPU Vision ViT encoder: returns true if encoded on device, false to fall back to CPU ViT.
    virtual bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) {
        (void)img; (void)out_embeds; (void)err;
        return false;
    }

    // Verify Section 5 selfcheck / golden ops parity on this backend
    virtual bool selfcheck(std::string& err) = 0;

    // Export / import full conversation state for one slot (ShortConv4 + 8 Global KV + 40 SWA-1024 KV)
    virtual bool export_slot(int slot, int n_tokens, InkSlotSnapshot& out, std::string& err) {
        (void)slot; (void)n_tokens; (void)out;
        err = "slot export not supported on this backend";
        return false;
    }
    virtual bool import_slot(int slot, const InkSlotSnapshot& in, std::string& err) {
        (void)slot; (void)in;
        err = "slot import not supported on this backend";
        return false;
    }
    // Lightweight prompt-end ShortConv4 checkpoint for multi-turn prefix reuse past discarded thinking (#6627)
    virtual bool save_prompt_checkpoint(int slot, int n_tokens) { (void)slot; (void)n_tokens; return false; }
    virtual bool restore_prompt_checkpoint(int slot, int n_tokens) { (void)slot; (void)n_tokens; return false; }
    virtual int  swa_slack() const noexcept { return 0; }
};

// Factory functions (CPU always built; GPU backends plug in via CMake compile definitions!)
std::unique_ptr<InkBackend> make_cpu_backend();

#ifdef INKLING_ENABLE_CUDA
std::unique_ptr<InkBackend> make_cuda_backend();
#endif
#ifdef INKLING_ENABLE_EP_CUDA
std::unique_ptr<InkBackend> make_ep_cuda_backend();
#endif
#ifdef INKLING_ENABLE_HIP
std::unique_ptr<InkBackend> make_hip_backend();
#endif
#ifdef INKLING_ENABLE_METAL
std::unique_ptr<InkBackend> make_metal_backend();
#endif
#ifdef INKLING_ENABLE_SYCL
std::unique_ptr<InkBackend> make_sycl_backend();
#endif

inline std::unique_ptr<InkBackend> make_default_backend(bool force_cpu = false, const BackendOpts* opts = nullptr) {
    if (!force_cpu) {
#ifdef INKLING_ENABLE_EP_CUDA
        // Prefer modular Dual-GPU EP / Graph-Doorbell Hybrid CUDA backend when --gpus >= 2, --cpu-threads > 0,
        // --vram-reserve-mib >= 0, or when K8V8/K8V4 KV is selected (or when primary CUDA backend isn't built).
        const bool need_ep_or_hybrid = opts && (
            opts->gpus >= 2 || opts->cpu_threads > 0 || opts->vram_reserve_mib >= 0 ||
            opts->kv_dtype == KvDtype::K8V8 || opts->kv_dtype == KvDtype::K8V4);
        if (need_ep_or_hybrid) return make_ep_cuda_backend();
#endif
#ifdef INKLING_ENABLE_CUDA
        return make_cuda_backend();
#elif defined(INKLING_ENABLE_EP_CUDA)
        return make_ep_cuda_backend();
#elif defined(INKLING_ENABLE_HIP)
        return make_hip_backend();
#elif defined(INKLING_ENABLE_METAL)
        return make_metal_backend();
#elif defined(INKLING_ENABLE_SYCL)
        return make_sycl_backend();
#endif
    }
    return make_cpu_backend();
}

}  // namespace ink2
