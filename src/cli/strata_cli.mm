/*
 * STRATA-M5 v2.0 — Live Tri-Engine Command-Line Interface
 * File: src/cli/strata_cli.mm
 *
 * Subcommands:
 *   - synth    --out <dir_or_safetensors> [--preset mini-deepseek|mini-qwen|mini-glm] [--seed 42]
 *   - convert  --safetensors <path> --bundle <dir>
 *   - inspect  --bundle <dir>
 *   - generate --bundle <dir> --prompt "<text>" [--max-tokens 32] [--mode stationary|zero-stall]
 *              [--draft-k 5] [--greedy | --temp 0.7] [--seed 42]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_checkpoint.h"
#include "../../include/strata_paths.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <vector>

extern "C" {

typedef struct StrataInferEngine StrataInferEngine;

typedef void (*StrataTokenStreamFn)(
    uint32_t token_id,
    const char* piece_utf8,
    void* user_ctx
);

typedef struct StrataInferOptions {
    StrataExecMode mode;
    uint32_t       max_tokens;
    uint32_t       draft_k;
    bool           greedy;
    float          temperature;
    uint64_t       seed;
    bool           stop_on_eos;
} StrataInferOptions;

typedef struct StrataInferTelemetry {
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    uint32_t decode_steps;
    uint32_t drafted_tokens_total;
    uint32_t accepted_draft_tokens;
    double   prefill_wall_ms;
    double   decode_wall_ms;
    double   tokens_per_sec;
    double   avg_accepted_per_step;
    double   draft_accept_rate;

    bool     native_metal_gpu;
    char     gpu_device_name[128];
    uint64_t gpu_spine_calls;
    double   gpu_spine_us;
    uint64_t gpu_moe_calls;
    double   gpu_moe_us;
    uint64_t moe_hot_hits;
    uint64_t moe_ring_hits;
    uint64_t moe_svd_qtip_fallbacks;

    uint64_t ane_probe_calls;
    double   ane_probe_us;
    bool     ran_on_ane;
    bool     compute_plan_all_ane;
    double   ane_vs_sme2_sqnr_db;

    uint64_t sme2_phase1_calls;
    double   sme2_io_us;
    double   sme2_compute_us;
    uint64_t ngram_lfu_hits;
    uint64_t ngram_nvme_ios;
    bool     hardware_sme2_active;

    uint64_t ring_dma_waves;
    uint64_t ring_extents_streamed;
    uint64_t ring_bytes_streamed;
    double   ring_dma_us;
    uint64_t ring_wait_calls;
    double   ring_wait_us;

    uint64_t logits_nan_inf_count;
    long     init_major_faults;
    long     steady_major_faults;
    long     steady_minor_faults;
    double   peak_rss_gib;
} StrataInferTelemetry;

StrataInferOptions strata_infer_default_options(void);
StrataInferEngine* strata_infer_create(const char* bundle_dir);
void strata_infer_destroy(StrataInferEngine* eng);
void strata_infer_reset(StrataInferEngine* eng);
const StrataCheckpointBundle* strata_infer_get_bundle(const StrataInferEngine* eng);

int strata_infer_generate(
    StrataInferEngine* eng,
    const char* prompt_utf8,
    const StrataInferOptions* opts,
    uint32_t* out_token_ids,
    uint32_t max_out_tokens,
    uint32_t* num_generated_out,
    StrataTokenStreamFn stream_cb,
    void* stream_user_ctx,
    StrataInferTelemetry* telemetry_out
);

int strata_qwen38_cmd_convert_hf(int argc, char** argv);
int strata_qwen38_cmd_run_hf(int argc, char** argv);
int strata_qwen38_cmd_bench_hf(int argc, char** argv);

int strata_qwen_flash_cmd_convert(int argc, char** argv);
int strata_qwen_flash_cmd_repack(int argc, char** argv);
int strata_qwen_flash_cmd_inspect(int argc, char** argv);
int strata_qwen_flash_cmd_run(int argc, char** argv);
int strata_qwen_flash_cmd_bench(int argc, char** argv);

int strata_glm53_cmd_convert(int argc, char** argv);
int strata_glm53_cmd_convert_fast(int argc, char** argv);
int strata_glm53_cmd_inspect(int argc, char** argv);
int strata_glm53_cmd_run(int argc, char** argv);
int strata_glm53_cmd_bench(int argc, char** argv);

int strata_inkling2_cmd_convert(int argc, char** argv);
int strata_inkling2_cmd_convert_vision(int argc, char** argv);
int strata_inkling2_cmd_repack(int argc, char** argv);
int strata_inkling2_cmd_inspect(int argc, char** argv);
int strata_inkling2_cmd_run(int argc, char** argv);
int strata_inkling2_cmd_bench(int argc, char** argv);

int strata_cmd_chat(int argc, char** argv);
int strata_cmd_chat_flash(int argc, char** argv);
int strata_cmd_chat_hf(int argc, char** argv);
int strata_cmd_chat_glm53(int argc, char** argv);
int strata_cmd_chat_inkling2(int argc, char** argv);
int strata_cmd_serve(int argc, char** argv);

} /* extern "C" */

namespace {

static std::string peek_flag_value(int argc, char** argv, const char* f1, const char* f2 = nullptr) {
    for (int i = 2; i < argc; ++i) {
        if ((std::strcmp(argv[i], f1) == 0 || (f2 && std::strcmp(argv[i], f2) == 0)) && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return {};
}

static bool is_glm53_bundle_dir(const std::string& bundle_dir) {
    if (bundle_dir.empty()) return false;
    const std::string sp = bundle_dir + "/spine.strata";
    return access(sp.c_str(), R_OK) == 0;
}

static bool is_glm53_hf_source(const std::string& hf_src) {
    if (hf_src.empty()) return false;
    if (hf_src.find("GLM-5") != std::string::npos ||
        hf_src.find("glm-5") != std::string::npos ||
        hf_src.find("glm5") != std::string::npos ||
        hf_src.find("Glm5Next") != std::string::npos) {
        return true;
    }
    std::ifstream ifs(hf_src + "/config.json");
    if (ifs) {
        std::ostringstream oss;
        oss << ifs.rdbuf();
        const std::string cfg = oss.str();
        if (cfg.find("glm5_next") != std::string::npos ||
            cfg.find("Glm5NextForConditionalGeneration") != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_inkling2_bundle_dir(const std::string& bundle_dir) {
    if (bundle_dir.empty()) return false;
    const std::string p = bundle_dir + "/inkling2_uma.strata";
    return access(p.c_str(), R_OK) == 0;
}

static bool is_inkling2_hf_source(const std::string& hf_src) {
    if (hf_src.empty()) return false;
    if (hf_src.find("Inkling") != std::string::npos ||
        hf_src.find("inkling") != std::string::npos) {
        return true;
    }
    std::ifstream ifs(hf_src + "/config.json");
    if (ifs) {
        std::ostringstream oss;
        oss << ifs.rdbuf();
        const std::string cfg = oss.str();
        if (cfg.find("inkling2_loop") != std::string::npos ||
            cfg.find("Inkling2LoopForConditionalGeneration") != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_qwen_flash_bundle_dir(const std::string& bundle_dir) {
    if (bundle_dir.empty()) return false;
    const std::string p0 = bundle_dir + "/uma_part0.strata";
    return access(p0.c_str(), R_OK) == 0;
}

static bool is_qwen_flash_hf_source(const std::string& hf_src) {
    if (hf_src.empty() || is_glm53_hf_source(hf_src)) return false;
    if (hf_src.find("Flash") != std::string::npos ||
        hf_src.find("flash") != std::string::npos ||
        hf_src.find("qwen4_exp") != std::string::npos) {
        return true;
    }
    std::ifstream ifs(hf_src + "/config.json");
    if (ifs) {
        std::ostringstream oss;
        oss << ifs.rdbuf();
        const std::string cfg = oss.str();
        if (cfg.find("qwen4_exp") != std::string::npos ||
            cfg.find("Qwen4ExpForConditionalGeneration") != std::string::npos) {
            return true;
        }
    }
    return false;
}

static const char* regime_to_str(StrataRegime r) {
    switch (r) {
        case STRATA_REGIME_I_100PCT_UMA: return "REGIME_I_100PCT_UMA (Full UMA + PLE)";
        case STRATA_REGIME_I_BITNEST:    return "REGIME_I_BITNEST (QTIP 2b + 4-lvl Refine)";
        case STRATA_REGIME_II_BARY_SVD:  return "REGIME_II_BARY_SVD (4b Hot + Barycenter + W6A16 MoE-SVD)";
        default:                         return "UNKNOWN";
    }
}

static void print_usage(const char* prog) {
    std::printf(
        "Strata Apex v%s — Native Apple Silicon Metal 4 LLM & Swarm Inference Engine\n"
        "\n"
        "Quickstart:\n"
        "  %s doctor                                  Check Apple Silicon GPU, UMA budget & installed bundles\n"
        "  %s pull <inkling-2-mini|qwen3.8-flash-next|glm-5.3-flash> [--bundle-repo <hf-repo>] [--keep-hf]\n"
        "  %s serve [inkling-2-mini|flash|glm53] [--port 8080] [--max-batch 1..64] [--max-seq-len 65536..1048576]\n"
        "           [--kv-cache f16|q8] [--int8-prefill] [--no-refusal]\n"
        "  %s setup-opencode [--port 8080] [--context 432768] [--output 32768] [--dry-run]\n"
        "  %s --version\n"
        "\n"
        "All Subcommands:\n"
        "  %s synth         --out <dir_or_safetensors> [--preset mini-deepseek|mini-qwen|mini-glm] [--seed 42]\n"
        "  %s convert       --safetensors <path> --bundle <dir>\n"
        "  %s inspect       --bundle <dir>\n"
        "  %s generate      --bundle <dir> --prompt \"<text>\" [--max-tokens 32] [--mode stationary|zero-stall]\n"
        "                   [--draft-k 5] [--greedy | --temp 0.7] [--seed 42]\n"
        "  %s convert-hf    --hf-dir <dir_or_repo> --bundle <dir>\n"
        "  %s run-hf        --bundle <dir> [--prompt \"<text>\" | --prompt-file <path>] [--max-tokens 256]\n"
        "                   [--draft-k 0..4] [--greedy | --temp 0.6] [--no-think]\n"
        "  %s bench-hf      --bundle <dir> [--contexts 512,4096,16384,32768] [--gen-tokens 128]\n"
        "  %s convert-flash [--hf-dir Qwen/Qwen3.8-Flash-Next] [--bundle ~/.cache/strata-apex/qwen3.8-flash-next]\n"
        "  %s inspect-flash [--bundle ~/.cache/strata-apex/qwen3.8-flash-next]\n"
        "  %s run-flash     [--bundle <dir>] [--prompt \"<text>\" | --prompt-file <path>] [--no-think | --think]\n"
        "                   [--draft-k 0..4] [--max-tokens 256] [--max-seq-len 263168] [--greedy | --temp 0.6] [--verify-k0]\n"
        "  %s bench-flash   [--bundle <dir>] [--contexts 512,4096,16384,32768,65536,131072,262144] [--gen-tokens 128]\n"
        "  %s convert-glm53      [--hf-dir TensorFold/GLM-5.3-Flash-MLX-4bit-MTP] [--bundle ~/.cache/strata-apex/glm-5.3-flash]\n"
        "  %s convert-glm53-fast [--hf-repo zai-org/GLM-5.3-Flash] [--out ~/.cache/strata-apex/glm-5.3-flash-fast] [--streams 4] [--trials 3]\n"
        "  %s inspect-glm53      [--bundle <dir>]\n"
        "  %s run-glm53          [--bundle <dir>] [--mode lossless|adaptive] [--hot-cache-gib 76.0] [--prompt \"<text>\"]\n"
        "                        [--no-think | --think] [--effort low|medium|high|max] [--draft-k 0..4] [--max-tokens 512]\n"
        "                        [--presence-penalty -2..2] [--frequency-penalty -2..2] [--penalty-window 0|8..65536] [--greedy | --temp 0.6] [--verify-k0]\n"
        "  %s bench-glm53        [--bundle <dir>] [--contexts 512,4096,16384,32768] [--modes lossless,adaptive] [--gen-tokens 64] [--no-think]\n"
        "  %s convert-inkling2        [--hf-dir ~/.cache/strata-apex/inkling-2-mini-hf] [--bundle ~/.cache/strata-apex/inkling-2-mini]\n"
        "  %s convert-inkling2-vision [--hf-dir ~/.cache/strata-apex/inkling-2-mini-hf] [--out ~/.cache/strata-apex/inkling-2-mini-fast/inkling2_vision.strata]\n"
        "  %s repack-inkling2         [--src ~/.cache/strata-apex/inkling-2-mini] [--dst ~/.cache/strata-apex/inkling-2-mini-fast]\n"
        "                             [--q4-unembed] [--q4-mtp] [--threads 8]\n"
        "  %s inspect-inkling2        [--bundle ~/.cache/strata-apex/inkling-2-mini-fast]\n"
        "  %s run-inkling2            [--bundle <dir>] [--image <path>] [--prompt \"<text>\" | --prompt-file <path>] [--no-think | --think]\n"
        "                             [--effort none|minimal|low|medium|high|xhigh] [--draft-k 0..7] [--adaptive-k | --no-adaptive-k]\n"
        "                             [--frspec | --no-frspec] [--tree-spec | --no-tree-spec] [--bitplane-head | --no-bitplane-head]\n"
        "                             [--max-tokens 256] [--max-seq-len 65536..1048576] [--kv-cache f16|q8] [--verify-k0]\n"
        "  %s bench-inkling2          [--bundle <dir>] [--contexts 512,2048,4096,16384,32768] [--draft-ks 0,7] [--gen-tokens 128]\n"
        "                             [--adaptive-k | --no-adaptive-k] [--frspec | --no-frspec] [--tree-spec | --no-tree-spec] [--bitplane-head | --no-bitplane-head]\n"
        "  %s chat                    [--bundle <dir>] [--draft-k 0..4] [--think | --no-think] [--greedy | --temp 0.6]\n"
        "                             [--max-tokens 8192] [--max-seq-len 263168] [--system \"<text>\"] [--script <path>] [--verify-multiturn]\n"
        "  %s chat-glm53              [--bundle <dir>] [--mode lossless|adaptive] [--draft-k 0..4] [--verify-multiturn]\n"
        "  %s chat-flash              [--bundle ~/.cache/strata-apex/qwen3.8-flash-next] [--draft-k 3] [--max-seq-len 263168] [--verify-multiturn]\n"
        "                             [--compact-mode off|shadow|paged|hybrid] [--compact-pool 32768] [--compact-window 32768] [--compact-watermark 0.80]\n"
        "                             [--verify-paged-kv] [--verify-shadow-swap]\n"
        "  %s chat-hf                 [--bundle ~/.cache/strata-apex/qwen3.8-27b] [--verify-multiturn]\n"
        "  %s chat-inkling2           [--bundle <dir>] [--kv-cache f16|q8] [--draft-k 0..7] [--max-seq-len 65536..1048576]\n"
        "  %s serve-inkling2          [--bundle <dir>] [--port 8080] [--effort medium] [--draft-k 7] [--max-batch 1..64] [--max-seq-len 65536..1048576] [--no-refusal]\n",
        STRATA_APEX_VERSION,
        prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog, prog, prog, prog);
}

static int cmd_synth(int argc, char** argv) {
    std::string out_path;
    std::string preset = "mini-deepseek";
    uint64_t seed      = 42ULL;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--out" || a == "-o") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--preset" && i + 1 < argc) {
            preset = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for synth: %s\n", a.c_str());
            return 1;
        }
    }

    if (out_path.empty()) {
        std::fprintf(stderr, "Error: `synth` requires `--out <dir_or_safetensors>`\n");
        return 1;
    }

    int rc = strata_synth_safetensors_checkpoint(out_path.c_str(), preset.c_str(), seed);
    if (rc != 0) {
        std::fprintf(stderr, "[strata_cli] synth failed (rc=%d)\n", rc);
        return 1;
    }

    std::string st_file = out_path;
    if (st_file.size() < 12U || st_file.substr(st_file.size() - 12U) != ".safetensors") {
        st_file += "/model.safetensors";
    }
    uint32_t num_tensors = 0U;
    uint64_t payload_bytes = 0ULL;
    (void)strata_pack_inspect_safetensors_header(st_file.c_str(), &num_tensors, &payload_bytes);

    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 `.safetensors` Checkpoint Synthesized\n");
    std::printf("================================================================================\n");
    std::printf("  Preset / Seed    : %s (seed=%llu)\n", preset.c_str(), static_cast<unsigned long long>(seed));
    std::printf("  Checkpoint Path  : %s\n", st_file.c_str());
    std::printf("  Tensors / Payload: %u tensors | %.3f MiB\n",
                num_tensors, static_cast<double>(payload_bytes) / (1024.0 * 1024.0));
    std::printf("================================================================================\n");
    return 0;
}

static int cmd_convert(int argc, char** argv) {
    std::string st_path;
    std::string bundle_dir;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--safetensors" || a == "--in") && i + 1 < argc) {
            st_path = argv[++i];
        } else if ((a == "--bundle" || a == "--out") && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for convert: %s\n", a.c_str());
            return 1;
        }
    }

    if (st_path.empty() || bundle_dir.empty()) {
        std::fprintf(stderr, "Error: `convert` requires `--safetensors <path> --bundle <dir>`\n");
        return 1;
    }

    StrataConvertStats stats{};
    int rc = strata_convert_safetensors_to_bundle(st_path.c_str(), bundle_dir.c_str(), &stats);
    if (rc != 0) {
        std::fprintf(stderr, "[strata_cli] convert failed (rc=%d)\n", rc);
        return 1;
    }

    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 4-Axis Disaggregation Complete -> %s\n", bundle_dir.c_str());
    std::printf("================================================================================\n");
    std::printf("  Source .safetensors  : %u tensors (%.3f MiB)\n",
                stats.num_safetensors_tensors,
                static_cast<double>(stats.safetensors_payload_bytes) / (1024.0 * 1024.0));
    std::printf("  Axis A (engram.ngram): %.3f MiB (4 KiB F_NOCACHE + ZT0 LUTI4 pages)\n",
                static_cast<double>(stats.axis_a_ngram_bytes) / (1024.0 * 1024.0));
    std::printf("  Axis B (spine.bin)   : %.3f MiB (32-way Tile-rANS E4M3 wCT + FP16 norms/MTP)\n",
                static_cast<double>(stats.axis_b_spine_bytes) / (1024.0 * 1024.0));
    std::printf("  Axis C (uma_hot.bin) : %.3f MiB (%u hot experts + Barycenter/QTIP + W6A16 SVD)\n",
                static_cast<double>(stats.axis_c_uma_hot_bytes) / (1024.0 * 1024.0),
                stats.num_hot_experts_total);
    std::printf("  Axis D (cold.strata) : %.3f MiB (%u cold extents, 256 KiB LBA-aligned)\n",
                static_cast<double>(stats.axis_d_cold_strata_bytes) / (1024.0 * 1024.0),
                stats.num_cold_experts_total);
    std::printf("  Verification Gates   : Tile-rANS Round-Trip=%s | 256 KiB LBA Align=%s\n",
                stats.rans_roundtrip_verified ? "PASS" : "FAIL",
                stats.lba_alignment_verified ? "PASS" : "FAIL");
    std::printf("================================================================================\n");
    return (stats.rans_roundtrip_verified && stats.lba_alignment_verified) ? 0 : 1;
}

static int cmd_inspect(int argc, char** argv) {
    std::string bundle_dir;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for inspect: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty()) {
        std::fprintf(stderr, "Error: `inspect` requires `--bundle <dir>`\n");
        return 1;
    }
    if (is_inkling2_bundle_dir(bundle_dir)) {
        return strata_inkling2_cmd_inspect(argc, argv);
    }
    if (is_glm53_bundle_dir(bundle_dir)) {
        return strata_glm53_cmd_inspect(argc, argv);
    }
    if (is_qwen_flash_bundle_dir(bundle_dir)) {
        return strata_qwen_flash_cmd_inspect(argc, argv);
    }

    StrataCheckpointBundle bdl{};
    int rc = strata_bundle_load(bundle_dir.c_str(), &bdl);
    if (rc != 0) {
        std::fprintf(stderr, "[strata_cli] inspect failed to load bundle `%s` (rc=%d)\n", bundle_dir.c_str(), rc);
        return 1;
    }

    uint64_t spine_rans_bytes = bdl.lm_head_rans_bytes + bdl.mtp_head_rans_bytes;
    uint64_t hot_bytes = 0ULL, base_bytes = 0ULL, svd_bytes = 0ULL;
    uint32_t total_hot_slots = 0U;
    for (uint32_t l = 0U; l < bdl.cfg.num_layers; ++l) {
        const StrataLayerWeights& lw = bdl.layers[l];
        spine_rans_bytes += lw.q_rans_bytes + lw.k_rans_bytes + lw.v_rans_bytes + lw.o_rans_bytes;
        hot_bytes        += lw.hot_pool_bytes;
        base_bytes       += lw.base_pool_bytes;
        svd_bytes        += lw.svd_u_w6_bytes;
        for (uint32_t e = 0U; e < bdl.cfg.num_experts; ++e) {
            if (lw.is_uma_hot[e]) ++total_hot_slots;
        }
    }

    StrataContainerHeader chdr{};
    std::vector<StrataExtentRecord> cext(std::max<uint32_t>(bdl.num_cold_extents, 1U));
    const int mrc = strata_pack_read_manifest(
        bdl.cold_strata_path, &chdr, cext.data(), static_cast<uint32_t>(cext.size()));

    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 Bundle Manifest Inspection: %s\n", bundle_dir.c_str());
    std::printf("================================================================================\n");
    std::printf("  Architecture      : %s\n", bdl.arch_name);
    std::printf("  Capacity Regime   : %s\n", regime_to_str(bdl.cfg.regime));
    std::printf("  Topology          : L=%u (enc=%u, pinned=%u) | d_model=%u | d_ff=%u | GQA %uQ/%uKV (hd=%u)\n",
                bdl.cfg.num_layers, bdl.cfg.encoder_layers, bdl.cfg.pinned_shallow_layers,
                bdl.cfg.d_model, bdl.cfg.d_ff, bdl.n_heads, bdl.n_kv_heads, bdl.head_dim);
    std::printf("  MoE & Speculative : E=%u, top_k=%u, svd_rank=%u | V=%u, FR-Spec shortlist=%u\n",
                bdl.cfg.num_experts, bdl.cfg.top_k, bdl.cfg.svd_rank,
                bdl.cfg.vocab_size, bdl.cfg.shortlist_size);
    std::printf("  Routing Policies  : Mode A tau_tok=%.4f | Mode B cold_cap=%u/layer | ANE probe r=%u\n",
                bdl.cfg.mode_a_tau_tok, bdl.cfg.mode_b_cold_cap, bdl.ane_probe_rank);
    std::printf("  Axis A (N-Gram)   : %s (layer=%u, heads=%u, %.2f KiB)\n",
                bdl.cfg.has_ngram_table ? "ENABLED" : "DISABLED",
                bdl.ngram_layer_idx, bdl.cfg.ngram_heads_per_tok,
                static_cast<double>(bdl.ngram_file_bytes) / 1024.0);
    std::printf("  Axis B (Spine)    : %.2f KiB 32-way Tile-rANS E4M3 (Q/K/V/O + LM Head + MTP Head)\n",
                static_cast<double>(spine_rans_bytes) / 1024.0);
    std::printf("  Axis C (UMA Hot)  : %u hot expert slots (%.2f KiB) | Base %.2f KiB | W6A16 SVD %.2f KiB\n",
                total_hot_slots,
                static_cast<double>(hot_bytes) / 1024.0,
                static_cast<double>(base_bytes) / 1024.0,
                static_cast<double>(svd_bytes) / 1024.0);
    std::printf("  Axis D (NVMe Cold): %u extents | data_start=%llu B | total=%.3f MiB | 256 KiB Align=%s\n",
                bdl.num_cold_extents,
                static_cast<unsigned long long>(chdr.data_start_offset),
                static_cast<double>(chdr.total_file_bytes) / (1024.0 * 1024.0),
                (mrc == 0) ? "PASS" : "FAIL");
    std::printf("================================================================================\n");

    strata_bundle_free(&bdl);
    return (mrc == 0) ? 0 : 1;
}

static void cli_stream_token_cb(uint32_t /*token_id*/, const char* piece_utf8, void* /*user_ctx*/) {
    if (piece_utf8 && piece_utf8[0] != '\0') {
        std::printf("%s", piece_utf8);
        std::fflush(stdout);
    }
}

static int cmd_generate(int argc, char** argv) {
    std::string bundle_dir;
    std::string prompt = "STRATA executes";
    StrataInferOptions opts = strata_infer_default_options();

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "--max-tokens" && i + 1 < argc) {
            opts.max_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "stationary" || m == "mode-a" || m == "a" || m == "A") {
                opts.mode = STRATA_MODE_A_STATIONARY;
            } else if (m == "zero-stall" || m == "zerostall" || m == "mode-b" || m == "b" || m == "B") {
                opts.mode = STRATA_MODE_B_ZERO_STALL;
            } else {
                std::fprintf(stderr, "Invalid `--mode %s` (expected `stationary` or `zero-stall`)\n", m.c_str());
                return 1;
            }
        } else if (a == "--draft-k" && i + 1 < argc) {
            opts.draft_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--greedy") {
            opts.greedy      = true;
            opts.temperature = 0.0f;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            opts.temperature = std::strtof(argv[++i], nullptr);
            opts.greedy      = (opts.temperature <= 1e-6f);
        } else if (a == "--seed" && i + 1 < argc) {
            opts.seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--no-eos-stop") {
            opts.stop_on_eos = false;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for generate: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty()) {
        std::fprintf(stderr, "Error: `generate` requires `--bundle <dir>`\n");
        return 1;
    }

    StrataInferEngine* eng = strata_infer_create(bundle_dir.c_str());
    if (!eng) {
        std::fprintf(stderr, "[strata_cli] Failed to initialize Tri-Engine runtime from `%s`\n", bundle_dir.c_str());
        return 1;
    }

    const StrataCheckpointBundle* bdl = strata_infer_get_bundle(eng);
    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 v2.0 Live Tri-Engine Autoregressive Speculative Generation\n");
    std::printf("  Model: %s | Mode: %s | Draft K=%u (B=%u) | %s\n",
                bdl ? bdl->arch_name : "unknown",
                (opts.mode == STRATA_MODE_A_STATIONARY) ? "Mode A (Stationary)" : "Mode B (Zero-Stall)",
                opts.draft_k,
                std::min<uint32_t>(opts.draft_k, STRATA_MAX_SPEC_BATCH - 1U) + 1U,
                opts.greedy ? "Greedy (T=0)" : "Stochastic Leviathan");
    std::printf("================================================================================\n");
    std::printf("[Prompt] %s\n[Output]", prompt.c_str());
    std::fflush(stdout);

    std::vector<uint32_t> gen_ids(std::max<uint32_t>(opts.max_tokens, 1U), 0U);
    uint32_t num_gen = 0U;
    StrataInferTelemetry tel{};

    int rc = strata_infer_generate(
        eng,
        prompt.c_str(),
        &opts,
        gen_ids.data(),
        opts.max_tokens,
        &num_gen,
        cli_stream_token_cb,
        nullptr,
        &tel);
    std::printf("\n");

    if (rc != 0) {
        std::fprintf(stderr, "[strata_cli] Generation failed (rc=%d)\n", rc);
        strata_infer_destroy(eng);
        return 1;
    }

    const double avg_spine_us = (tel.gpu_spine_calls > 0ULL)
        ? (tel.gpu_spine_us / static_cast<double>(tel.gpu_spine_calls))
        : 0.0;
    const double avg_moe_us = (tel.gpu_moe_calls > 0ULL)
        ? (tel.gpu_moe_us / static_cast<double>(tel.gpu_moe_calls))
        : 0.0;
    const double avg_ane_us = (tel.ane_probe_calls > 0ULL)
        ? (tel.ane_probe_us / static_cast<double>(tel.ane_probe_calls))
        : 0.0;

    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  Tri-Engine Silicon Telemetry (%s)\n", tel.gpu_device_name);
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  Throughput & Decode  : %7.2f tok/s | %u toks in %u steps (%.2f ms decode, %.2f ms prefill)\n",
                tel.tokens_per_sec, tel.generated_tokens, tel.decode_steps,
                tel.decode_wall_ms, tel.prefill_wall_ms);
    std::printf("  Speculative FR-Spec  : %6.2f accepted/step | Draft hit %u/%u (%.1f%%)\n",
                tel.avg_accepted_per_step,
                tel.accepted_draft_tokens, tel.drafted_tokens_total,
                100.0 * tel.draft_accept_rate);
    std::printf("  40c NAX GPU (Spine)  : %6llu calls | %8.1f us total (%5.1f us/call, Tile-rANS E4M3->NAX)\n",
                static_cast<unsigned long long>(tel.gpu_spine_calls),
                tel.gpu_spine_us, avg_spine_us);
    std::printf("  40c NAX GPU (MoE)    : %6llu calls | %8.1f us total (%5.1f us/call | hot=%llu, ring=%llu, fb=%llu)\n",
                static_cast<unsigned long long>(tel.gpu_moe_calls),
                tel.gpu_moe_us, avg_moe_us,
                static_cast<unsigned long long>(tel.moe_hot_hits),
                static_cast<unsigned long long>(tel.moe_ring_hits),
                static_cast<unsigned long long>(tel.moe_svd_qtip_fallbacks));
    std::printf("  16c ANE Window Probe : %6llu calls | %8.1f us total (%5.1f us/call | ran_on_ane=%d, plan=%d, %.1f dB)\n",
                static_cast<unsigned long long>(tel.ane_probe_calls),
                tel.ane_probe_us, avg_ane_us,
                tel.ran_on_ane ? 1 : 0,
                tel.compute_plan_all_ane ? 1 : 0,
                tel.ane_vs_sme2_sqnr_db);
    std::printf("  3x SME2 Phase-1      : %6llu calls | I/O %6.1f us + SME2 %6.1f us (LFU=%llu, 4K_IO=%llu, hw=%d)\n",
                static_cast<unsigned long long>(tel.sme2_phase1_calls),
                tel.sme2_io_us, tel.sme2_compute_us,
                static_cast<unsigned long long>(tel.ngram_lfu_hits),
                static_cast<unsigned long long>(tel.ngram_nvme_ios),
                tel.hardware_sme2_active ? 1 : 0);
    std::printf("  ANS NVMe Slot Ring   : %6llu exts  | %6.2f MiB async DMA (%6.1f us) | waits=%llu (%.1f us)\n",
                static_cast<unsigned long long>(tel.ring_extents_streamed),
                static_cast<double>(tel.ring_bytes_streamed) / (1024.0 * 1024.0),
                tel.ring_dma_us,
                static_cast<unsigned long long>(tel.ring_wait_calls),
                tel.ring_wait_us);
    std::printf("  Memory & Fault Gates : Peak RSS %.3f GiB | Steady MajFlt=%ld (Init=%ld) | NaN/Inf=%llu\n",
                tel.peak_rss_gib,
                tel.steady_major_faults,
                tel.init_major_faults,
                static_cast<unsigned long long>(tel.logits_nan_inf_count));
    std::printf("================================================================================\n");

    strata_infer_destroy(eng);
    return (tel.steady_major_faults == 0 && tel.logits_nan_inf_count == 0ULL) ? 0 : 1;
}

static bool cli_file_exists(const std::string& p) {
    struct stat st{};
    return (!p.empty() && ::stat(p.c_str(), &st) == 0);
}

static std::string cli_sysctl_str(const char* key) {
    char buf[256] = {};
    size_t sz = sizeof(buf) - 1U;
    if (sysctlbyname(key, buf, &sz, nullptr, 0U) == 0 && buf[0] != '\0') {
        return std::string(buf);
    }
    return "Unknown";
}

static uint64_t cli_sysctl_u64(const char* key) {
    uint64_t v = 0ULL;
    size_t sz = sizeof(v);
    if (sysctlbyname(key, &v, &sz, nullptr, 0U) == 0) return v;
    uint32_t v32 = 0U;
    size_t sz32 = sizeof(v32);
    if (sysctlbyname(key, &v32, &sz32, nullptr, 0U) == 0) return static_cast<uint64_t>(v32);
    return 0ULL;
}

static int cmd_doctor() {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    const std::string hw_model = cli_sysctl_str("hw.model");
    const std::string chip     = cli_sysctl_str("machdep.cpu.brand_string");
    const uint64_t    mem_b    = cli_sysctl_u64("hw.memsize");
    const uint64_t    ncpu     = cli_sysctl_u64("hw.ncpu");
    const uint64_t    has_sme2 = cli_sysctl_u64("hw.optional.arm.FEAT_SME2");

    std::string gpu_name = "Unavailable";
    double rec_ws_gib = 0.0, max_buf_gib = 0.0;
    bool has_uma = false, has_m4 = false;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev) {
            gpu_name    = [[dev name] UTF8String];
            has_uma     = [dev hasUnifiedMemory];
            rec_ws_gib  = static_cast<double>([dev recommendedMaxWorkingSetSize]) / kGiB;
            max_buf_gib = static_cast<double>([dev maxBufferLength]) / kGiB;
            if (@available(macOS 26.0, *)) has_m4 = true;
        }
    }
    const double mem_gib  = static_cast<double>(mem_b) / kGiB;
    const double safe_gib = (rec_ws_gib > 0.0) ? (rec_ws_gib * 0.893) : (mem_gib * 0.75);

    std::printf("================================================================================\n");
    std::printf("  Strata Apex v%s — Apple Silicon Readiness & Environment Doctor\n", STRATA_APEX_VERSION);
    std::printf("================================================================================\n");
    std::printf("  Hardware       : %s (%s, %llu CPU cores, %.0f GiB UMA)\n",
                chip.c_str(), hw_model.c_str(), static_cast<unsigned long long>(ncpu), mem_gib);
    std::printf("  Metal GPU      : %s | UMA=%s | Metal 4 NAX=%s | SME2=%s\n",
                gpu_name.c_str(), has_uma ? "YES" : "NO",
                has_m4 ? "AVAILABLE" : "FALLBACK (Metal 3.1)",
                has_sme2 ? "YES" : "NO");
    std::printf("  Metal Limits   : RecMaxWorkingSet %.2f GiB | MaxBuffer %.2f GiB | Safe Budget %.1f GiB\n",
                rec_ws_gib, max_buf_gib, safe_gib);
    std::printf("  Cache Root     : %s\n", strata_apex_cache_root());
    std::printf("  Metal Cache    : %s\n", strata_apex_metal_cache_dir());
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  Installed Model Bundles:\n");

    struct BundleProbe { const char* label; const char* rel_dir; const char* marker; };
    const BundleProbe probes[] = {
        {"Inkling-2-Mini (fast)",       "inkling-2-mini-fast",           "inkling2_uma.strata"},
        {"Inkling-2-Mini (v1)",         "inkling-2-mini",                "inkling2_uma.strata"},
        {"Inkling-2-Mini Vision",       "inkling-2-mini-fast",           "inkling2_vision.strata"},
        {"Inkling-2-Mini No-Refusal",   "inkling-2-mini-norefusal-fast", "inkling2_uma.strata"},
        {"Qwen3.8-Flash-Next (fast)",   "qwen3.8-flash-next-fast",       "uma_part0.strata"},
        {"Qwen3.8-Flash-Next (v1)",     "qwen3.8-flash-next",            "uma_part0.strata"},
        {"GLM-5.3-Flash (fast RAM)",    "glm-5.3-flash-fast",            "experts_uma_part0.strata"},
        {"GLM-5.3-Flash (NVMe)",        "glm-5.3-flash",                 "spine.strata"},
    };
    int found_any = 0;
    for (const auto& p : probes) {
        const std::string dir = strata_apex_bundle_path(p.rel_dir);
        const std::string fp  = dir + "/" + p.marker;
        if (cli_file_exists(fp)) {
            struct stat st{};
            double gib = 0.0;
            if (::stat(fp.c_str(), &st) == 0) gib = static_cast<double>(st.st_size) / kGiB;
            std::printf("    [READY] %-27s  %6.2f GiB  (%s)\n", p.label, gib, dir.c_str());
            ++found_any;
        }
    }
    if (found_any == 0) {
        std::printf("    (none found yet — run `strata-apex pull inkling-2-mini` to get started)\n");
    }
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  Recommended Inkling-2-Mini Launch Presets for %.0f GiB UMA:\n", mem_gib);
    if (mem_gib >= 96.0) {
        std::printf("    Solo (64K-1M)  : strata-apex serve inkling-2-mini --max-seq-len 1048576 --kv-cache q8\n");
        std::printf("    Swarm (8-16x)  : strata-apex serve inkling-2-mini --max-batch 16 --max-seq-len 16384\n");
    } else if (mem_gib >= 48.0) {
        std::printf("    Solo (64K-256K): strata-apex serve inkling-2-mini --max-seq-len 262144 --kv-cache q8\n");
        std::printf("    Swarm (4-8x)   : strata-apex serve inkling-2-mini --max-batch 8 --max-seq-len 16384 --kv-cache q8\n");
    } else if (mem_gib >= 24.0) {
        std::printf("    Solo (16K-64K) : strata-apex serve inkling-2-mini --max-seq-len 32768 --kv-cache q8\n");
        std::printf("    Swarm (2-4x)   : strata-apex serve inkling-2-mini --max-batch 4 --max-seq-len 8192 --kv-cache q8\n");
    } else {
        std::printf("    Note: Inkling-2-Mini needs ~17 GiB minimum UMA (24 GiB+ Mac recommended).\n");
    }
    std::printf("================================================================================\n");
    return 0;
}

static std::string cli_shell_quote(const std::string& s) {
    std::string out = "'";
    for (char ch : s) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out.push_back('\'');
    return out;
}

static int cmd_pull(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[2], "-h") == 0 || std::strcmp(argv[2], "--help") == 0) {
        std::printf(
            "Usage: %s pull <model> [--bundle-repo <hf-repo>] [--hf-repo <hf-repo>] [--no-vision] [--keep-hf]\n"
            "\n"
            "Models:\n"
            "  inkling-2-mini      Inkling-2-Mini 26B Looped (~14.2 GiB fast bundle + 0.6 GiB Vision)\n"
            "  qwen3.8-flash-next  Qwen3.8-Flash-Next 180B MoE (~67.7 GiB UMA + 27.4 GiB NVMe PLE)\n"
            "  glm-5.3-flash       GLM-5.3-Flash 320B MoE (~93.2 GiB RAM-resident E8S-18 fast bundle)\n"
            "\n"
            "Tip: If you publish pre-packed `.strata` files on HuggingFace Hub, pass `--bundle-repo <org/repo>`\n"
            "     to download them directly without local conversion.\n",
            argv[0]);
        return (argc < 3) ? 1 : 0;
    }

    const std::string model = argv[2];
    std::string bundle_repo;
    std::string hf_repo;
    bool with_vision = true;
    bool keep_hf     = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bundle-repo" && i + 1 < argc) bundle_repo = argv[++i];
        else if (a == "--hf-repo" && i + 1 < argc) hf_repo = argv[++i];
        else if (a == "--no-vision") with_vision = false;
        else if (a == "--keep-hf") keep_hf = true;
        else {
            std::fprintf(stderr, "[strata-apex pull] Unknown option `%s`\n", a.c_str());
            return 1;
        }
    }

    auto find_hf_cli = []() -> std::string {
        if (std::system("command -v hf >/dev/null 2>&1") == 0) return "hf";
        if (std::system("command -v huggingface-cli >/dev/null 2>&1") == 0) return "huggingface-cli";
        return {};
    };

    if (model == "inkling-2-mini" || model == "inkling2" || model == "inkling") {
        const std::string cache_root = strata_apex_cache_root();
        const std::string fast_dir   = cache_root + "/inkling-2-mini-fast";
        const std::string v1_dir     = cache_root + "/inkling-2-mini";
        const std::string hf_dir     = cache_root + "/inkling-2-mini-hf";

        if (!bundle_repo.empty()) {
            const std::string hf_bin = find_hf_cli();
            if (hf_bin.empty()) {
                std::fprintf(stderr,
                             "[strata-apex pull] HuggingFace CLI (`hf` or `huggingface-cli`) not found.\n"
                             "  Install with: brew install huggingface-cli  (or: pipx install huggingface_hub)\n");
                return 1;
            }
            const std::string dl_cmd = hf_bin + " download " + cli_shell_quote(bundle_repo) +
                                       " --local-dir " + cli_shell_quote(fast_dir);
            std::printf("[strata-apex pull] Downloading pre-packed bundle `%s` -> `%s`...\n",
                        bundle_repo.c_str(), fast_dir.c_str());
            if (std::system(dl_cmd.c_str()) != 0) {
                std::fprintf(stderr, "[strata-apex pull] Download failed.\n");
                return 1;
            }
            std::printf("[strata-apex pull] Ready! Launch with:\n  strata-apex serve inkling-2-mini --max-batch 8\n");
            return 0;
        }

        const std::string repo = hf_repo.empty() ? std::string("thinkingmachines/Inkling-2-Mini") : hf_repo;
        const std::string hf_bin = find_hf_cli();
        if (hf_bin.empty()) {
            std::fprintf(stderr,
                         "[strata-apex pull] HuggingFace CLI (`hf` or `huggingface-cli`) not found.\n"
                         "  Install with: brew install huggingface-cli  (or: pipx install huggingface_hub)\n"
                         "  Then re-run:  strata-apex pull inkling-2-mini\n");
            return 1;
        }

        std::printf("[strata-apex pull] Step 1/3: Downloading `%s` -> `%s`...\n", repo.c_str(), hf_dir.c_str());
        const std::string dl_cmd = hf_bin + " download " + cli_shell_quote(repo) +
                                   " --local-dir " + cli_shell_quote(hf_dir);
        if (std::system(dl_cmd.c_str()) != 0) {
            std::fprintf(stderr, "[strata-apex pull] HuggingFace download failed.\n");
            return 1;
        }

        std::printf("[strata-apex pull] Step 2/3: Converting to native .strata UMA bundle...\n");
        {
            std::vector<std::string> a = {argv[0], "convert-inkling2", "--hf-dir", hf_dir, "--bundle", v1_dir};
            std::vector<char*> av;
            for (auto& s : a) av.push_back(s.data());
            if (strata_inkling2_cmd_convert(static_cast<int>(av.size()), av.data()) != 0) return 1;
        }
        std::printf("[strata-apex pull] Step 3/3: Repacking fast bundle (~14.2 GiB)%s...\n",
                    with_vision ? " + Vision sidecar" : "");
        {
            std::vector<std::string> a = {argv[0], "repack-inkling2", "--src", v1_dir, "--dst", fast_dir};
            std::vector<char*> av;
            for (auto& s : a) av.push_back(s.data());
            if (strata_inkling2_cmd_repack(static_cast<int>(av.size()), av.data()) != 0) return 1;
        }
        if (with_vision) {
            const std::string vis_out = fast_dir + "/inkling2_vision.strata";
            std::vector<std::string> a = {argv[0], "convert-inkling2-vision", "--hf-dir", hf_dir, "--out", vis_out};
            std::vector<char*> av;
            for (auto& s : a) av.push_back(s.data());
            if (strata_inkling2_cmd_convert_vision(static_cast<int>(av.size()), av.data()) != 0) return 1;
        }
        if (!keep_hf) {
            std::printf("[strata-apex pull] Cleaning up intermediate HF & v1 staging dirs...\n");
            const std::string rm_cmd = "rm -rf " + cli_shell_quote(hf_dir) + " " + cli_shell_quote(v1_dir);
            (void)std::system(rm_cmd.c_str());
        }
        std::printf("\n[strata-apex pull] Done! Ready at `%s`.\n"
                    "  Solo : strata-apex serve inkling-2-mini\n"
                    "  Swarm: strata-apex serve inkling-2-mini --max-batch 8 --max-seq-len 16384\n",
                    fast_dir.c_str());
        return 0;
    }

    if (model == "qwen3.8-flash-next" || model == "flash" || model == "qwen") {
        const std::string repo = hf_repo.empty() ? std::string("Qwen/Qwen3.8-Flash-Next") : hf_repo;
        const std::string out  = std::string(strata_apex_cache_root()) + "/qwen3.8-flash-next";
        std::vector<std::string> a = {argv[0], "convert-flash", "--hf-dir", repo, "--bundle", out};
        std::vector<char*> av;
        for (auto& s : a) av.push_back(s.data());
        return strata_qwen_flash_cmd_convert(static_cast<int>(av.size()), av.data());
    }

    if (model == "glm-5.3-flash" || model == "glm53" || model == "glm") {
        const std::string repo = hf_repo.empty() ? std::string("zai-org/GLM-5.3-Flash") : hf_repo;
        const std::string out  = std::string(strata_apex_cache_root()) + "/glm-5.3-flash-fast";
        std::vector<std::string> a = {argv[0], "convert-glm53-fast", "--hf-repo", repo, "--out", out};
        std::vector<char*> av;
        for (auto& s : a) av.push_back(s.data());
        return strata_glm53_cmd_convert_fast(static_cast<int>(av.size()), av.data());
    }

    std::fprintf(stderr, "[strata-apex pull] Unknown model `%s` (use inkling-2-mini, qwen3.8-flash-next, or glm-5.3-flash)\n",
                 model.c_str());
    return 1;
}

static int cmd_setup_opencode(int argc, char** argv) {
    int port = 8080;
    int ctx_limit = 432768; /* Triggers OpenCode auto-compaction around ~400K usable input */
    int out_limit = 32768;
    std::string provider_id = "strata-apex";
    bool dry_run = false;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            std::printf("Usage: %s setup-opencode [--port 8080] [--context 432768] [--output 32768] [--provider strata-apex] [--dry-run]\n", argv[0]);
            return 0;
        } else if (a == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (a == "--context" && i + 1 < argc) {
            ctx_limit = std::atoi(argv[++i]);
        } else if (a == "--output" && i + 1 < argc) {
            out_limit = std::atoi(argv[++i]);
        } else if (a == "--provider" && i + 1 < argc) {
            provider_id = argv[++i];
        } else if (a == "--dry-run") {
            dry_run = true;
        } else {
            std::fprintf(stderr, "[setup-opencode] Unknown option `%s`\n", a.c_str());
            return 1;
        }
    }
    if (port < 1 || port > 65535 || ctx_limit < 4096 || out_limit < 512) {
        std::fprintf(stderr, "[setup-opencode] Invalid --port / --context / --output value.\n");
        return 1;
    }

    @autoreleasepool {
        const char* home_c = std::getenv("HOME");
        if (!home_c || home_c[0] == '\0') {
            std::fprintf(stderr, "[setup-opencode] HOME environment variable is not set.\n");
            return 1;
        }
        NSString* dir_path = [NSString stringWithFormat:@"%s/.config/opencode", home_c];
        NSString* cfg_path = [dir_path stringByAppendingPathComponent:@"opencode.json"];

        NSMutableDictionary* root = nil;
        NSData* existing = [NSData dataWithContentsOfFile:cfg_path];
        if (existing) {
            NSError* perr = nil;
            id obj = [NSJSONSerialization JSONObjectWithData:existing
                                                     options:NSJSONReadingMutableContainers
                                                       error:&perr];
            if ([obj isKindOfClass:[NSMutableDictionary class]]) {
                root = static_cast<NSMutableDictionary*>(obj);
            } else {
                std::fprintf(stderr, "[setup-opencode] Existing `%s` is not a valid JSON object: %s\n",
                             [cfg_path UTF8String], perr ? [[perr localizedDescription] UTF8String] : "parse error");
                return 1;
            }
        } else {
            root = [NSMutableDictionary dictionary];
            root[@"$schema"] = @"https://opencode.ai/config.json";
            root[@"model"]   = [NSString stringWithFormat:@"%s/inkling-2-mini", provider_id.c_str()];
        }

        NSMutableDictionary* providers = [root[@"provider"] isKindOfClass:[NSMutableDictionary class]]
            ? root[@"provider"]
            : [NSMutableDictionary dictionary];
        root[@"provider"] = providers;

        auto make_model = [&](NSString* mid, NSString* name, int ctx, int out_tok) -> NSDictionary* {
            return @{
                @"id": mid,
                @"name": name,
                @"tool_call": @YES,
                @"reasoning": @YES,
                @"interleaved": @{@"field": @"reasoning_content"},
                @"variants": @{
                    @"low":    @{@"reasoningEffort": @"low"},
                    @"medium": @{@"reasoningEffort": @"medium"},
                    @"high":   @{@"reasoningEffort": @"high"},
                    @"xhigh":  @{@"reasoningEffort": @"xhigh"},
                },
                @"limit": @{@"context": @(ctx), @"output": @(out_tok)},
            };
        };

        NSString* base_url = [NSString stringWithFormat:@"http://127.0.0.1:%d/v1", port];
        NSString* pid_ns   = [NSString stringWithUTF8String:provider_id.c_str()];
        providers[pid_ns] = @{
            @"npm": @"@ai-sdk/openai-compatible",
            @"name": @"Strata Apex Local (Apple Silicon Metal 4)",
            @"options": @{@"baseURL": base_url, @"apiKey": @"strata-apex-local"},
            @"models": @{
                @"inkling-2-mini": make_model(@"inkling-2-mini", @"Inkling-2-Mini (26B Looped Strata Apex)", ctx_limit, out_limit),
                @"inkling-2-mini-norefusal": make_model(@"inkling-2-mini-norefusal", @"Inkling-2-Mini No-Refusal (26B Strata Apex)", ctx_limit, out_limit),
                @"qwen3.8-flash-next": make_model(@"qwen3.8-flash-next", @"Qwen3.8-Flash-Next (180B Strata Apex)", 262144, 65536),
                @"glm-5.3-flash": make_model(@"glm-5.3-flash", @"GLM-5.3-Flash (320B Strata Apex)", 131072, 32768),
            },
        };

        NSError* serr = nil;
        NSData* json_data = [NSJSONSerialization dataWithJSONObject:root
                                                            options:(NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys)
                                                              error:&serr];
        if (!json_data) {
            std::fprintf(stderr, "[setup-opencode] Failed to serialize JSON: %s\n",
                         serr ? [[serr localizedDescription] UTF8String] : "unknown");
            return 1;
        }
        NSString* json_str = [[NSString alloc] initWithData:json_data encoding:NSUTF8StringEncoding];
        if (dry_run) {
            std::printf("[setup-opencode] Dry-run preview for `%s`:\n%s\n", [cfg_path UTF8String], [json_str UTF8String]);
            return 0;
        }

        NSError* merr = nil;
        [[NSFileManager defaultManager] createDirectoryAtPath:dir_path
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&merr];
        if (![json_data writeToFile:cfg_path options:NSDataWritingAtomic error:&merr]) {
            std::fprintf(stderr, "[setup-opencode] Failed writing `%s`: %s\n",
                         [cfg_path UTF8String], merr ? [[merr localizedDescription] UTF8String] : "unknown");
            return 1;
        }
        std::printf("[setup-opencode] Updated `%s` with provider `%s` (%s, Inkling context=%d / output=%d).\n",
                    [cfg_path UTF8String], provider_id.c_str(), [base_url UTF8String], ctx_limit, out_limit);
    }
    return 0;
}

} /* namespace */

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }
        std::string sub = argv[1];
        if (sub == "--version" || sub == "-V" || sub == "version") {
            std::printf("strata-apex %s (Apple Silicon Metal 4 Engine)\n", STRATA_APEX_VERSION);
            return 0;
        }
        if (sub == "doctor")         return cmd_doctor();
        if (sub == "pull")           return cmd_pull(argc, argv);
        if (sub == "setup-opencode") return cmd_setup_opencode(argc, argv);
        if (sub == "synth")          return cmd_synth(argc, argv);
        if (sub == "convert")       return cmd_convert(argc, argv);
        if (sub == "inspect")       return cmd_inspect(argc, argv);
        if (sub == "generate")      return cmd_generate(argc, argv);
        if (sub == "convert-flash") return strata_qwen_flash_cmd_convert(argc, argv);
        if (sub == "repack-flash")  return strata_qwen_flash_cmd_repack(argc, argv);
        if (sub == "inspect-flash") return strata_qwen_flash_cmd_inspect(argc, argv);
        if (sub == "run-flash")     return strata_qwen_flash_cmd_run(argc, argv);
        if (sub == "bench-flash")   return strata_qwen_flash_cmd_bench(argc, argv);
        if (sub == "convert-glm53")      return strata_glm53_cmd_convert(argc, argv);
        if (sub == "convert-glm53-fast") return strata_glm53_cmd_convert_fast(argc, argv);
        if (sub == "inspect-glm53")      return strata_glm53_cmd_inspect(argc, argv);
        if (sub == "run-glm53")          return strata_glm53_cmd_run(argc, argv);
        if (sub == "bench-glm53")        return strata_glm53_cmd_bench(argc, argv);
        if (sub == "convert-inkling2")        return strata_inkling2_cmd_convert(argc, argv);
        if (sub == "convert-inkling2-vision") return strata_inkling2_cmd_convert_vision(argc, argv);
        if (sub == "repack-inkling2")         return strata_inkling2_cmd_repack(argc, argv);
        if (sub == "inspect-inkling2")        return strata_inkling2_cmd_inspect(argc, argv);
        if (sub == "run-inkling2")            return strata_inkling2_cmd_run(argc, argv);
        if (sub == "bench-inkling2")          return strata_inkling2_cmd_bench(argc, argv);
        if (sub == "chat")                    return strata_cmd_chat(argc, argv);
        if (sub == "chat-glm53")              return strata_cmd_chat_glm53(argc, argv);
        if (sub == "chat-flash")              return strata_cmd_chat_flash(argc, argv);
        if (sub == "chat-hf")                 return strata_cmd_chat_hf(argc, argv);
        if (sub == "chat-inkling2")           return strata_cmd_chat_inkling2(argc, argv);
        if (sub == "serve" || sub == "serve-flash" || sub == "serve-glm53" ||
            sub == "serve-hf" || sub == "serve-qwen38" ||
            sub == "serve-inkling2" || sub == "serve-inkling") {
            return strata_cmd_serve(argc, argv);
        }
        if (sub == "convert-hf") {
            const std::string hf_src = peek_flag_value(argc, argv, "--hf-dir", "--repo");
            if (is_inkling2_hf_source(hf_src)) return strata_inkling2_cmd_convert(argc, argv);
            if (is_glm53_hf_source(hf_src)) return strata_glm53_cmd_convert(argc, argv);
            return is_qwen_flash_hf_source(hf_src)
                ? strata_qwen_flash_cmd_convert(argc, argv)
                : strata_qwen38_cmd_convert_hf(argc, argv);
        }
        if (sub == "run-hf") {
            const std::string bdl = peek_flag_value(argc, argv, "--bundle");
            if (is_inkling2_bundle_dir(bdl)) return strata_inkling2_cmd_run(argc, argv);
            if (is_glm53_bundle_dir(bdl)) return strata_glm53_cmd_run(argc, argv);
            return is_qwen_flash_bundle_dir(bdl)
                ? strata_qwen_flash_cmd_run(argc, argv)
                : strata_qwen38_cmd_run_hf(argc, argv);
        }
        if (sub == "bench-hf") {
            const std::string bdl = peek_flag_value(argc, argv, "--bundle");
            if (is_inkling2_bundle_dir(bdl)) return strata_inkling2_cmd_bench(argc, argv);
            if (is_glm53_bundle_dir(bdl)) return strata_glm53_cmd_bench(argc, argv);
            return is_qwen_flash_bundle_dir(bdl)
                ? strata_qwen_flash_cmd_bench(argc, argv)
                : strata_qwen38_cmd_bench_hf(argc, argv);
        }
        if (sub == "--help" || sub == "-h" || sub == "help") {
            print_usage(argv[0]);
            return 0;
        }
        std::fprintf(stderr, "Unknown subcommand `%s`\n\n", sub.c_str());
        print_usage(argv[0]);
        return 1;
    }
}
