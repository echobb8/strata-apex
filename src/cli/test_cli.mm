/*
 * STRATA-M5 v2.0 — End-to-End 4-Axis Converter & Live Tri-Engine Speculative CLI Test Suite
 * File: src/cli/test_cli.mm
 *
 * Verifies:
 *   [1/4] `.safetensors` Synthesis (`mini-deepseek`, `mini-qwen`, `mini-glm`), 4-Axis
 *         `.strata` Disaggregation, `256 KiB` LBA Alignment & `Tile-rANS` Exact Round-Trip.
 *   [2/4] Exactness Gate 1 (Greedy Equivalence): `K=0` single-token greedy decode vs
 *         `K=5` `STRATA_MODE_A_STATIONARY` speculative decode for `24` tokens -> asserts
 *         100% bit-identical token ID sequence!
 *   [3/4] Exactness Gate 2 (Tri-Engine Co-Execution & Mode B Zero-Stall): asserts physical
 *         `16-Core ANE` (`ran_on_ane == true`, `MLComputePlan` verified), `ARMv9.2 SME2`
 *         Phase-1 execution, `MTLIOCommandQueue` (`F_NOCACHE`) cold extent streaming with
 *         `Steady MajFlt == 0`, `0` NaNs/Infs in logits, Mode B `0` stalls (`ring_wait_calls == 0`),
 *         and stochastic FR-Spec + Leviathan rejection sampling stability.
 *   [4/4] Safety & Disk Hygiene Gate: `Peak RSS <= 4.0 GiB` and `0` leftover bytes in `bench_data/`.
 */

#import <Foundation/Foundation.h>

#include "../../include/strata_checkpoint.h"
#include "../../include/strata_glm53.h"
#include "../../include/strata_inkling2.h"
#include "../../include/strata_inkling2_vision.h"
#include "../../include/strata_qwen38.h"
#include "../../include/strata_qwen_flash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
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

int strata_glm53_cmd_convert(int argc, char** argv);
int strata_glm53_cmd_convert_fast(int argc, char** argv);
int strata_glm53_cmd_inspect(int argc, char** argv);
int strata_glm53_cmd_run(int argc, char** argv);
int strata_glm53_cmd_bench(int argc, char** argv);
int strata_qwen_flash_cmd_run(int argc, char** argv);
int strata_qwen_flash_cmd_bench(int argc, char** argv);

int strata_cmd_chat(int argc, char** argv);
int strata_cmd_chat_flash(int argc, char** argv);
int strata_cmd_chat_hf(int argc, char** argv);
int strata_cmd_chat_glm53(int argc, char** argv);
int strata_cmd_chat_inkling2(int argc, char** argv);
int strata_chat_repl_unit_tests(void);
int strata_cmd_serve(int argc, char** argv);
int strata_openai_server_unit_tests(void);
int strata_inkling2_cmd_convert_vision(int argc, char** argv);
int strata_inkling2_cmd_run(int argc, char** argv);
int strata_inkling2_cli_unit_tests(void);

} /* extern "C" */

namespace {

static void remove_tree(const std::string& path) {
    struct stat st{};
    if (::lstat(path.c_str(), &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR* d = ::opendir(path.c_str());
        if (d) {
            struct dirent* ent = nullptr;
            while ((ent = ::readdir(d)) != nullptr) {
                if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) {
                    continue;
                }
                remove_tree(path + "/" + ent->d_name);
            }
            ::closedir(d);
        }
        ::rmdir(path.c_str());
    } else {
        ::unlink(path.c_str());
    }
}

static uint64_t count_dir_entries_and_bytes(const std::string& dir_path, uint32_t* files_out) {
    uint64_t total_bytes = 0ULL;
    DIR* d = ::opendir(dir_path.c_str());
    if (!d) return 0ULL;
    struct dirent* ent = nullptr;
    while ((ent = ::readdir(d)) != nullptr) {
        if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        std::string sub = dir_path + "/" + ent->d_name;
        struct stat st{};
        if (::lstat(sub.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                total_bytes += count_dir_entries_and_bytes(sub, files_out);
            } else {
                if (files_out) (*files_out)++;
                total_bytes += static_cast<uint64_t>(st.st_size);
            }
        }
    }
    ::closedir(d);
    return total_bytes;
}

static std::string decode_tokens_to_text(
    const StrataCheckpointBundle* bdl,
    const uint32_t* ids,
    uint32_t n
) {
    std::string s;
    for (uint32_t i = 0U; i < n; ++i) {
        s += strata_detokenize_token(bdl, ids[i]);
    }
    return s;
}

static bool verify_bundle_lba_and_rans(const std::string& bundle_dir, bool expect_cold_extents) {
    StrataCheckpointBundle bdl{};
    if (strata_bundle_load(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "  [FAIL] strata_bundle_load failed for %s\n", bundle_dir.c_str());
        return false;
    }

    bool ok = true;

    /* 1. Verify 256 KiB LBA alignment on cold_experts.strata */
    StrataContainerHeader chdr{};
    std::vector<StrataExtentRecord> exts(std::max<uint32_t>(bdl.num_cold_extents, 1U));
    if (strata_pack_read_manifest(
            bdl.cold_strata_path, &chdr, exts.data(), static_cast<uint32_t>(exts.size())) != 0) {
        std::fprintf(stderr, "  [FAIL] strata_pack_read_manifest failed on %s\n", bdl.cold_strata_path);
        ok = false;
    }
    if ((chdr.data_start_offset % STRATA_LBA_ALIGN_BYTES) != 0ULL) {
        std::fprintf(stderr, "  [FAIL] data_start_offset not 256 KiB aligned\n");
        ok = false;
    }
    if (expect_cold_extents && chdr.num_extents == 0U) {
        std::fprintf(stderr, "  [FAIL] expected >0 cold extents in %s\n", bdl.cold_strata_path);
        ok = false;
    }
    for (uint32_t i = 0U; i < chdr.num_extents; ++i) {
        if ((exts[i].offset_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
            (exts[i].aligned_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
            exts[i].aligned_bytes == 0ULL) {
            std::fprintf(stderr, "  [FAIL] extent %u violates 256 KiB LBA alignment\n", i);
            ok = false;
            break;
        }
    }

    /* 2. Independently verify 32-way Tile-rANS lossless decode on Spine Q-proj & LM Head */
    const StrataLayerWeights& lw0 = bdl.layers[0];
    const size_t q_cnt = static_cast<size_t>(lw0.n_heads * lw0.head_dim) * bdl.cfg.d_model;
    std::vector<uint8_t> q_syms(q_cnt, 0U);
    if (strata_codec_rans_decode_32way_cpu_ref(lw0.q_rans_blob, lw0.q_rans_bytes, q_syms.data(), q_cnt) != 0) {
        std::fprintf(stderr, "  [FAIL] Tile-rANS CPU decode failed on layer 0 q_rans_blob\n");
        ok = false;
    }

    if (bdl.lm_head_rans_blob != nullptr) {
        const size_t lm_cnt = static_cast<size_t>(bdl.cfg.vocab_size) * bdl.cfg.d_model;
        std::vector<uint8_t> lm_syms(lm_cnt, 0U);
        if (strata_codec_rans_decode_32way_cpu_ref(
                bdl.lm_head_rans_blob, bdl.lm_head_rans_bytes, lm_syms.data(), lm_cnt) != 0) {
            std::fprintf(stderr, "  [FAIL] Tile-rANS CPU decode failed on lm_head_rans_blob\n");
            ok = false;
        }
    }

    strata_bundle_free(&bdl);
    return ok;
}

static uint64_t tcli_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

static float tcli_rand_sym(uint64_t* state, float amp) {
    const uint64_t r = tcli_splitmix64(state);
    const double u = static_cast<double>(r >> 11U) * (1.0 / 9007199254740992.0);
    return static_cast<float>((2.0 * u - 1.0) * static_cast<double>(amp));
}

static void fill_synth_q4(uint8_t* base, const StrataGlm53QuantMatrix& qm, uint64_t* rng, float scale_f) {
    if (qm.weight_bytes == 0ULL) return;
    auto* w = reinterpret_cast<uint32_t*>(base + qm.weight_off);
    const size_t n_u32 = static_cast<size_t>(qm.weight_bytes / sizeof(uint32_t));
    for (size_t i = 0U; i < n_u32; ++i) {
        w[i] = static_cast<uint32_t>(tcli_splitmix64(rng));
    }
    auto* sc = reinterpret_cast<uint16_t*>(base + qm.scales_off);
    auto* bi = reinterpret_cast<uint16_t*>(base + qm.biases_off);
    const size_t n_s = static_cast<size_t>(qm.scales_bytes / sizeof(uint16_t));
    const uint16_t sc_h = sg53_f32_to_f16(scale_f);
    const uint16_t bi_h = sg53_f32_to_f16(-7.5f * scale_f);
    for (size_t i = 0U; i < n_s; ++i) {
        sc[i] = sc_h;
        bi[i] = bi_h;
    }
}

static void fill_synth_f32_const(uint8_t* base, const StrataGlm53TensorSlice& sl, float val) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<float*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = val;
}

static void fill_synth_f32_rand(uint8_t* base, const StrataGlm53TensorSlice& sl, uint64_t* rng, float amp) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<float*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = tcli_rand_sym(rng, amp);
}

static void fill_synth_f16_rand(uint8_t* base, const StrataGlm53TensorSlice& sl, uint64_t* rng, float amp) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<uint16_t*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = sg53_f32_to_f16(tcli_rand_sym(rng, amp));
}

static bool write_all_bytes(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    if (!buf.empty()) {
        ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }
    return static_cast<bool>(ofs);
}

static bool synthesize_mini_glm53_v2_bundle(const std::string& out_dir) {
    ::mkdir(out_dir.c_str(), 0755);

    StrataGlm53Config cfg{};
    strata_glm53_default_config(&cfg);
    cfg.num_hidden_layers        = 4U;   /* L0 dense KDA, L1..L2 MoE KDA, L3 MoE DSA */
    cfg.num_nextn_predict_layers = 1U;   /* L4 MTP DSA + MoE */
    cfg.first_k_dense_replace    = 1U;   /* 3 main MoE layers -> part0=2, part1=1 */
    cfg.hidden_size              = 256U;
    cfg.vocab_size               = 256U;
    cfg.intermediate_size        = 256U;
    cfg.num_kda_layers           = 3U;
    cfg.num_dsa_layers           = 1U;
    cfg.mhc_enabled              = 1U;
    cfg.hc_mult                  = 4U;
    cfg.hc_sinkhorn_iters        = 20U;
    cfg.linear_num_heads         = 2U;
    cfg.linear_head_dim          = 128U;
    cfg.num_attention_heads      = 4U;
    cfg.num_key_value_heads      = 4U;
    cfg.q_lora_rank              = 128U;
    cfg.kv_lora_rank             = 128U;
    cfg.qk_nope_head_dim         = 128U;
    cfg.qk_rope_head_dim         = 0U;
    cfg.v_head_dim               = 128U;
    cfg.index_n_heads            = 4U;
    cfg.index_head_dim           = 128U;
    cfg.index_kpool              = 4U;
    cfg.index_topk               = 64U;
    cfg.n_routed_experts         = 16U;
    cfg.num_experts_per_tok      = 4U;
    cfg.n_shared_experts         = 1U;
    cfg.moe_intermediate_size    = 256U;
    cfg.quant_group_size         = 64U;
    cfg.pad_token_id             = 0U;
    cfg.bos_gmask_token_id       = 1U;
    cfg.bos_sop_token_id         = 2U;
    cfg.eos_token_ids[0]         = 255U;
    cfg.num_eos_token_ids        = 1U;
    for (uint32_t l = 0U; l < STRATA_GLM53_MAX_LAYERS; ++l) {
        cfg.layer_attn_types[l] = (l == 3U) ? STRATA_GLM53_ATTN_DSA_MLA : STRATA_GLM53_ATTN_KDA_LINEAR;
        cfg.layer_mlp_types[l]  = (l < cfg.first_k_dense_replace) ? STRATA_GLM53_MLP_DENSE : STRATA_GLM53_MLP_SPARSE_MOE;
    }

    auto hdr = std::make_unique<StrataGlm53SpineHeader>();
    strata_glm53_init_layout_v2(&cfg, hdr.get());

    std::vector<uint8_t> spine(static_cast<size_t>(hdr->total_spine_bytes), 0U);
    std::memcpy(spine.data(), hdr.get(), sizeof(*hdr));

    uint64_t rng = 0x5354524154415632ULL;
    fill_synth_q4(spine.data(), hdr->embed_tokens, &rng, 0.03f);
    fill_synth_f32_const(spine.data(), hdr->final_norm, 1.0f);
    fill_synth_q4(spine.data(), hdr->lm_head, &rng, 0.04f);

    auto fill_layer = [&](const StrataGlm53LayerDesc& ld) {
        if (ld.has_mhc) {
            fill_synth_f32_rand(spine.data(), ld.hc_attn.fn, &rng, 0.005f);
            fill_synth_f32_const(spine.data(), ld.hc_attn.base, 0.0f);
            fill_synth_f32_const(spine.data(), ld.hc_attn.scale, 0.1f);
            fill_synth_f32_rand(spine.data(), ld.hc_ffn.fn, &rng, 0.005f);
            fill_synth_f32_const(spine.data(), ld.hc_ffn.base, 0.0f);
            fill_synth_f32_const(spine.data(), ld.hc_ffn.scale, 0.1f);
        }
        fill_synth_f32_const(spine.data(), ld.input_layernorm, 1.0f);
        fill_synth_f32_const(spine.data(), ld.post_attention_layernorm, 1.0f);
        if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
            fill_synth_q4(spine.data(), ld.lin_qkv_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.lin_fbg_a_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.lin_f_b_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.lin_g_b_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.lin_o_proj, &rng, 0.015f);
            fill_synth_f32_rand(spine.data(), ld.lin_qkv_conv1d, &rng, 0.15f);
            fill_synth_f32_const(spine.data(), ld.lin_A_log, -1.0f);
            fill_synth_f32_const(spine.data(), ld.lin_dt_bias, 0.0f);
            fill_synth_f32_const(spine.data(), ld.lin_o_norm, 1.0f);
        } else {
            fill_synth_q4(spine.data(), ld.mla_qkv_a_proj, &rng, 0.015f);
            fill_synth_f32_const(spine.data(), ld.mla_q_a_layernorm, 1.0f);
            fill_synth_q4(spine.data(), ld.mla_q_b_proj, &rng, 0.015f);
            fill_synth_f32_const(spine.data(), ld.mla_kv_a_layernorm, 1.0f);
            fill_synth_f16_rand(spine.data(), ld.mla_w_uk, &rng, 0.02f);
            fill_synth_f16_rand(spine.data(), ld.mla_w_uv, &rng, 0.02f);
            fill_synth_q4(spine.data(), ld.mla_o_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.indexer_wq_b, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.indexer_wk, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.indexer_weights_proj, &rng, 0.015f);
            fill_synth_f32_const(spine.data(), ld.indexer_k_norm_weight, 1.0f);
            fill_synth_f32_const(spine.data(), ld.indexer_k_norm_bias, 0.0f);
            fill_synth_f32_rand(spine.data(), ld.indexer_compress_ape, &rng, 0.05f);
            fill_synth_f32_rand(spine.data(), ld.indexer_compress_gate, &rng, 0.02f);
        }
        if (ld.mlp_type == STRATA_GLM53_MLP_DENSE) {
            fill_synth_q4(spine.data(), ld.dense_gate_up_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.dense_down_proj, &rng, 0.015f);
        } else {
            fill_synth_f16_rand(spine.data(), ld.moe_router_gate, &rng, 0.05f);
            fill_synth_f32_const(spine.data(), ld.moe_e_score_bias, 0.0f);
            fill_synth_q4(spine.data(), ld.shared_gate_up_proj, &rng, 0.015f);
            fill_synth_q4(spine.data(), ld.shared_down_proj, &rng, 0.015f);
        }
    };

    for (uint32_t l = 0U; l < cfg.num_hidden_layers; ++l) {
        fill_layer(hdr->layers[l]);
    }
    fill_synth_f32_const(spine.data(), hdr->mtp.enorm, 1.0f);
    fill_synth_f32_const(spine.data(), hdr->mtp.hnorm, 1.0f);
    fill_synth_q4(spine.data(), hdr->mtp.eh_proj, &rng, 0.015f);
    fill_layer(hdr->mtp.layer);
    fill_synth_f32_const(spine.data(), hdr->mtp.shared_head_norm, 1.0f);

    /* Populate v2 E8S-18-V4 main experts in `part0` and `part1` with valid orbits, H8 signs, V4 bits & E4M4 scales */
    auto fill_e8_part = [&](std::vector<uint8_t>* buf, uint32_t num_layers_in_part) {
        const uint8_t sc_u8 = sg53_e4m4_encode_f32(0.012f);
        const uint32_t n_exp = num_layers_in_part * cfg.n_routed_experts;
        for (uint32_t e = 0U; e < n_exp; ++e) {
            uint8_t* rec = buf->data() + static_cast<size_t>(e) * static_cast<size_t>(hdr->e8_record_bytes);
            for (uint32_t p = 0U; p < 3U; ++p) {
                uint8_t* prj = rec + static_cast<size_t>(p) * static_cast<size_t>(hdr->e8_proj_bytes);
                auto* lo16 = reinterpret_cast<uint16_t*>(prj);
                auto* hi2  = reinterpret_cast<uint16_t*>(prj + hdr->e8_lo16_bytes);
                uint8_t* sc = prj + hdr->e8_lo16_bytes + hdr->e8_hi2_bytes;
                const size_t n_lo = static_cast<size_t>(hdr->e8_lo16_bytes / sizeof(uint16_t));
                const size_t n_hi = static_cast<size_t>(hdr->e8_hi2_bytes / sizeof(uint16_t));
                const size_t n_sc = static_cast<size_t>(hdr->e8_scale_bytes);
                for (size_t i = 0U; i < n_lo; ++i) {
                    lo16[i] = static_cast<uint16_t>(tcli_splitmix64(&rng) & 0xFFFFU);
                }
                for (size_t i = 0U; i < n_hi; ++i) {
                    hi2[i] = static_cast<uint16_t>(tcli_splitmix64(&rng) & 0xFFFFU);
                }
                std::memset(sc, sc_u8, n_sc);
            }
        }
    };

    std::vector<uint8_t> part0(static_cast<size_t>(hdr->part0_bytes), 0U);
    std::vector<uint8_t> part1(static_cast<size_t>(hdr->part1_bytes), 0U);
    fill_e8_part(&part0, hdr->part0_moe_layers);
    fill_e8_part(&part1, hdr->part1_moe_layers);

    /* Populate Layer-45 MTP Q4_A experts in `mtp_experts.strata` */
    std::vector<uint8_t> mtp_exp(static_cast<size_t>(hdr->mtp_experts_bytes), 0U);
    const uint16_t mtp_sc_h = sg53_f32_to_f16(0.015f);
    const uint16_t mtp_bi_h = sg53_f32_to_f16(-7.5f * 0.015f);
    for (uint32_t e = 0U; e < cfg.n_routed_experts; ++e) {
        uint8_t* rec = mtp_exp.data() + static_cast<size_t>(e) * static_cast<size_t>(hdr->expert_record_bytes);
        for (uint32_t p = 0U; p < 3U; ++p) {
            uint8_t* prj = rec + static_cast<size_t>(p) * static_cast<size_t>(hdr->expert_proj_bytes);
            auto* w  = reinterpret_cast<uint32_t*>(prj);
            auto* sc = reinterpret_cast<uint16_t*>(prj + hdr->expert_weight_bytes);
            auto* bi = reinterpret_cast<uint16_t*>(prj + hdr->expert_weight_bytes + hdr->expert_scales_bytes);
            const size_t n_w = static_cast<size_t>(hdr->expert_weight_bytes / sizeof(uint32_t));
            const size_t n_s = static_cast<size_t>(hdr->expert_scales_bytes / sizeof(uint16_t));
            for (size_t i = 0U; i < n_w; ++i) w[i] = static_cast<uint32_t>(tcli_splitmix64(&rng));
            for (size_t i = 0U; i < n_s; ++i) { sc[i] = mtp_sc_h; bi[i] = mtp_bi_h; }
        }
    }

    return write_all_bytes(out_dir + "/" STRATA_GLM53_SPINE_FILENAME, spine) &&
           write_all_bytes(out_dir + "/" STRATA_GLM53_EXPERTS_PART0_FILENAME, part0) &&
           write_all_bytes(out_dir + "/" STRATA_GLM53_EXPERTS_PART1_FILENAME, part1) &&
           write_all_bytes(out_dir + "/" STRATA_GLM53_MTP_EXPERTS_FILENAME, mtp_exp);
}

static bool run_glm53_v2_ram_resident_cli_test(const std::string& bundle_dir) {
    if (!synthesize_mini_glm53_v2_bundle(bundle_dir)) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: failed to synthesize mini v2 bundle\n");
        return false;
    }

    /* 1. CLI inspect-glm53 on v2 RAM-resident bundle */
    const char* insp_av[] = {"strata_cli", "inspect-glm53", "--bundle", bundle_dir.c_str()};
    if (strata_glm53_cmd_inspect(4, const_cast<char**>(insp_av)) != 0) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: inspect-glm53 failed\n");
        return false;
    }

    /* 2. Open bundle & verify v2 zero-copy UMA pointers / zero NVMe FDs */
    StrataGlm53Bundle bdl{};
    if (strata_glm53_bundle_open(bundle_dir.c_str(), &bdl) != 0) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: strata_glm53_bundle_open failed\n");
        return false;
    }
    bool ok = (bdl.is_ram_resident &&
               bdl.header &&
               bdl.header->version == STRATA_GLM53_FORMAT_VERSION_V2 &&
               bdl.header->is_ram_resident == 1U &&
               bdl.header->main_expert_dtype == STRATA_GLM53_EXPERT_DTYPE_E8S18_V4 &&
               bdl.part0_base != nullptr && bdl.part0_bytes > 0U &&
               bdl.part1_base != nullptr && bdl.part1_bytes > 0U &&
               bdl.mtp_experts_base != nullptr && bdl.mtp_experts_bytes > 0U &&
               bdl.num_experts_fds == 0U && bdl.experts_nvme_bytes == 0ULL);
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: bundle metadata mismatch\n");
        strata_glm53_bundle_close(&bdl);
        return false;
    }

    /* 3. Create v2 RAM-Resident GPU engine & verify zero hot-cache / zero NVMe allocation */
    StrataGlm53GpuEngine* eng = strata_glm53_gpu_create(&bdl, 256U, 0.0f, STRATA_GLM53_MODE_LOSSLESS);
    if (!eng) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: strata_glm53_gpu_create failed\n");
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    StrataGlm53GpuInfo info{};
    strata_glm53_gpu_get_info(eng, &info);
    if (info.is_ram_resident != 1U ||
        info.main_expert_dtype != STRATA_GLM53_EXPERT_DTYPE_E8S18_V4 ||
        info.hot_cache_bytes != 0ULL ||
        info.ring_dma_bytes != 0ULL ||
        info.experts_nvme_file_bytes != 0ULL ||
        info.num_nvme_fds != 0U ||
        info.uma_part0_mapped_bytes != bdl.part0_bytes ||
        info.uma_part1_mapped_bytes != bdl.part1_bytes ||
        info.mtp_experts_mapped_bytes != bdl.mtp_experts_bytes) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: StrataGlm53GpuInfo v2 fields mismatch\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }

    const uint32_t prompt[16] = {
        5U, 18U, 31U, 44U, 57U, 70U, 83U, 96U,
        109U, 122U, 135U, 148U, 161U, 174U, 187U, 200U
    };
    uint32_t prompt40[40];
    for (uint32_t i = 0U; i < 40U; ++i) {
        prompt40[i] = 5U + ((i * 7U) % 240U);
    }
    constexpr uint32_t kGenLen = 16U;

    auto run_seq_prompt = [&](const uint32_t* p_ids, uint32_t p_len,
                              uint32_t draft_k, const StrataGlm53SamplingParams& samp,
                              std::vector<uint32_t>* out_ids) -> bool {
        out_ids->clear();
        strata_glm53_gpu_reset(eng);
        strata_glm53_gpu_penalties_reset(eng);
        strata_glm53_gpu_reset_counters(eng);
        uint32_t first_tok = 0U;
        if (strata_glm53_gpu_prefill(eng, p_ids, p_len, &samp, &first_tok, nullptr) != 0) return false;
        if (first_tok >= bdl.cfg.vocab_size) return false;
        out_ids->push_back(first_tok);
        uint32_t pending = first_tok;
        while (out_ids->size() < kGenLen) {
            if (draft_k == 0U) {
                uint32_t nxt = 0U;
                if (strata_glm53_gpu_decode_step(eng, pending, &samp, &nxt, nullptr) != 0) return false;
                if (nxt >= bdl.cfg.vocab_size) return false;
                out_ids->push_back(nxt);
                pending = nxt;
            } else {
                uint32_t em[STRATA_GLM53_MAX_SPEC_DRAFT_K + 2U] = {};
                uint32_t ne = 0U, na = 0U;
                if (strata_glm53_gpu_spec_decode_step(eng, pending, draft_k, &samp, em, &ne, &na) != 0 || ne == 0U) {
                    return false;
                }
                for (uint32_t i = 0U; i < ne && out_ids->size() < kGenLen; ++i) {
                    if (em[i] >= bdl.cfg.vocab_size) return false;
                    out_ids->push_back(em[i]);
                }
                pending = em[ne - 1U];
            }
        }
        return out_ids->size() == kGenLen;
    };
    auto run_seq = [&](uint32_t draft_k, const StrataGlm53SamplingParams& samp, std::vector<uint32_t>* out_ids) -> bool {
        return run_seq_prompt(prompt, 16U, draft_k, samp, out_ids);
    };

    /* 4. Warmup once (both FP16 and INT8 M>=32 prefill paths) before checking Steady MajFlt */
    StrataGlm53SamplingParams greedy{};
    greedy.greedy             = true;
    greedy.temperature        = 0.0f;
    greedy.repetition_penalty = 1.0f;
    greedy.rng_seed           = 42ULL;
    std::vector<uint32_t> warm_ids;
    if (!run_seq(4U, greedy, &warm_ids)) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: warmup run failed\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    strata_glm53_gpu_set_int8_prefill(eng, true);
    if (!run_seq_prompt(prompt40, 40U, 0U, greedy, &warm_ids)) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: int8_prefill warmup run failed\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    strata_glm53_gpu_set_int8_prefill(eng, false);

    struct rusage ru0{};
    getrusage(RUSAGE_SELF, &ru0);

    /* 5. Pure Greedy MTP K=4 == K=0 Bit-Exact Verification */
    std::vector<uint32_t> k0_ids, k4_ids;
    if (!run_seq(0U, greedy, &k0_ids) || !run_seq(4U, greedy, &k4_ids) || k0_ids != k4_ids) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: greedy MTP K=4 != K=0 mismatch\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    StrataGlm53PerfCounters cnt{};
    strata_glm53_gpu_get_counters(eng, &cnt);
    if (cnt.nvme_cold_reads != 0ULL || cnt.nvme_bytes_read != 0ULL) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: non-zero NVMe reads (%llu) on RAM-resident bundle\n",
                     static_cast<unsigned long long>(cnt.nvme_cold_reads));
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }

    /* 5b. Verify `--int8-prefill` (M=40 >= 32), `--no-adaptive-k`, and `--no-frspec` toggles & K=4==K=0 exactness */
    strata_glm53_gpu_set_int8_prefill(eng, true);
    strata_glm53_gpu_set_adaptive_k(eng, false);
    strata_glm53_gpu_set_frspec(eng, false);
    StrataGlm53GpuInfo info_tog{};
    strata_glm53_gpu_get_info(eng, &info_tog);
    if (info_tog.int8_prefill_enabled != 1U ||
        info_tog.adaptive_k_enabled != 0U ||
        info_tog.frspec_enabled != 0U) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: toggle state mismatch (int8=%u, ak=%u, fr=%u)\n",
                     info_tog.int8_prefill_enabled, info_tog.adaptive_k_enabled, info_tog.frspec_enabled);
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    std::vector<uint32_t> i8_k0, i8_k4;
    if (!run_seq_prompt(prompt40, 40U, 0U, greedy, &i8_k0) ||
        !run_seq_prompt(prompt40, 40U, 4U, greedy, &i8_k4) ||
        i8_k0 != i8_k4) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: int8_prefill + no-adaptive-k + no-frspec K=4 != K=0 mismatch\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }
    strata_glm53_gpu_set_int8_prefill(eng, false);
    strata_glm53_gpu_set_adaptive_k(eng, true);
    strata_glm53_gpu_set_frspec(eng, true);
    strata_glm53_gpu_get_info(eng, &info_tog);
    if (info_tog.int8_prefill_enabled != 0U || info_tog.adaptive_k_enabled != 1U) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: toggle restore mismatch (int8=%u, ak=%u)\n",
                     info_tog.int8_prefill_enabled, info_tog.adaptive_k_enabled);
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }

    /* 6. Sliding-Window Penalties (`window=8, pres=1.2, freq=0.25, rep=1.1`) MTP K=4 == K=0 Bit-Exact */
    StrataGlm53SamplingParams pen_samp = greedy;
    pen_samp.presence_penalty   = 1.2f;
    pen_samp.frequency_penalty  = 0.25f;
    pen_samp.repetition_penalty = 1.10f;
    pen_samp.penalty_window     = 8U;

    std::vector<uint32_t> pk0_ids, pk4_ids;
    if (!run_seq(0U, pen_samp, &pk0_ids) || !run_seq(4U, pen_samp, &pk4_ids) || pk0_ids != pk4_ids) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: sliding-window penalty MTP K=4 != K=0 mismatch\n");
        strata_glm53_gpu_destroy(eng);
        strata_glm53_bundle_close(&bdl);
        return false;
    }

    struct rusage ru1{};
    getrusage(RUSAGE_SELF, &ru1);
    const long steady_majflt = std::max<long>(0L, ru1.ru_majflt - ru0.ru_majflt);

    strata_glm53_gpu_destroy(eng);

    /* 7. Verify MTLBinaryArchive persistent cache-hit boot on second engine creation */
    {
        const char* dis_env = std::getenv("STRATA_METAL_CACHE_DISABLE");
        const bool cache_on = !(dis_env && std::strcmp(dis_env, "1") == 0);
        StrataGlm53GpuEngine* eng2 = strata_glm53_gpu_create(&bdl, 256U, 0.0f, STRATA_GLM53_MODE_LOSSLESS);
        if (!eng2) {
            std::fprintf(stderr, "  [FAIL] glm53_v2: second strata_glm53_gpu_create failed\n");
            strata_glm53_bundle_close(&bdl);
            return false;
        }
        StrataGlm53GpuInfo info2{};
        strata_glm53_gpu_get_info(eng2, &info2);
        strata_glm53_gpu_destroy(eng2);
        if (cache_on && (info2.metal_cache_hit != 1U || info2.pso_load_ms < 0.0)) {
            std::fprintf(stderr, "  [FAIL] glm53_v2: expected MTLBinaryArchive cache hit on 2nd boot (hit=%u, ms=%.2f)\n",
                         info2.metal_cache_hit, info2.pso_load_ms);
            strata_glm53_bundle_close(&bdl);
            return false;
        }
    }

    strata_glm53_bundle_close(&bdl);

    if (steady_majflt != 0L) {
        std::fprintf(stderr, "  [FAIL] glm53_v2: non-zero steady MajFlt=%ld\n", steady_majflt);
        return false;
    }
    return true;
}

static uint64_t ik2_align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

static StrataInkling2TensorSlice ik2_alloc_slice(
    uint64_t* cur,
    uint32_t dtype,
    uint32_t shape0,
    uint32_t shape1
) {
    StrataInkling2TensorSlice sl{};
    if (dtype == STRATA_INKLING2_DTYPE_NONE || shape0 == 0U || shape1 == 0U) return sl;
    *cur = strata_inkling2_align256(*cur);
    sl.offset     = *cur;
    sl.dtype      = dtype;
    sl.shape0     = shape0;
    sl.shape1     = shape1;
    sl.elem_count = shape0 * shape1;
    const uint64_t bpe = (dtype == STRATA_INKLING2_DTYPE_F32) ? 4ULL : 2ULL;
    sl.bytes = static_cast<uint64_t>(sl.elem_count) * bpe;
    *cur = strata_inkling2_align256(*cur + sl.bytes);
    return sl;
}

static StrataInkling2ExpertMatrix ik2_alloc_experts(
    uint64_t* cur,
    uint32_t E,
    uint32_t rows,
    uint32_t cols,
    uint32_t dtype
) {
    StrataInkling2ExpertMatrix em{};
    em.num_experts     = E;
    em.rows_per_expert = rows;
    em.cols            = cols;
    em.dtype           = dtype;
    if (dtype == STRATA_INKLING2_DTYPE_NVFP4) {
        em.block_size = 16U;
        *cur = strata_inkling2_align256(*cur);
        em.weight_off   = *cur;
        em.weight_bytes = static_cast<uint64_t>(E) * rows * (cols / 2U);
        *cur = strata_inkling2_align256(*cur + em.weight_bytes);
        em.scales_off   = *cur;
        em.scales_bytes = static_cast<uint64_t>(E) * rows * (cols / 16U);
        *cur = strata_inkling2_align256(*cur + em.scales_bytes);
        em.expert_scale_off   = *cur;
        em.expert_scale_bytes = static_cast<uint64_t>(E) * sizeof(float);
        *cur = strata_inkling2_align256(*cur + em.expert_scale_bytes);
    } else {
        em.block_size = 32U;
        *cur = strata_inkling2_align256(*cur);
        em.weight_off   = *cur;
        em.weight_bytes = static_cast<uint64_t>(E) * rows * cols;
        *cur = strata_inkling2_align256(*cur + em.weight_bytes);
        em.scales_off   = *cur;
        em.scales_bytes = static_cast<uint64_t>(E) * rows * (cols / 32U);
        *cur = strata_inkling2_align256(*cur + em.scales_bytes);
    }
    return em;
}

static void ik2_fill_f32_const(uint8_t* base, const StrataInkling2TensorSlice& sl, float val) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<float*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = val;
}

static void ik2_fill_f32_rand(uint8_t* base, const StrataInkling2TensorSlice& sl, uint64_t* rng, float amp) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<float*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = tcli_rand_sym(rng, amp);
}

static void ik2_fill_f16_rand(uint8_t* base, const StrataInkling2TensorSlice& sl, uint64_t* rng, float amp) {
    if (sl.bytes == 0ULL || sl.elem_count == 0U) return;
    auto* p = reinterpret_cast<uint16_t*>(base + sl.offset);
    for (uint32_t i = 0U; i < sl.elem_count; ++i) p[i] = sg53_f32_to_f16(tcli_rand_sym(rng, amp));
}

static void ik2_fill_experts(uint8_t* base, const StrataInkling2ExpertMatrix& em, uint64_t* rng) {
    uint8_t* w = base + em.weight_off;
    uint8_t* s = base + em.scales_off;
    if (em.dtype == STRATA_INKLING2_DTYPE_NVFP4) {
        for (uint64_t i = 0ULL; i < em.weight_bytes; ++i) {
            w[i] = static_cast<uint8_t>(tcli_splitmix64(rng) & 0xFFU);
        }
        /* float8_e4m3fn 0x38 == 1.0f */
        std::memset(s, 0x38, static_cast<size_t>(em.scales_bytes));
        auto* es = reinterpret_cast<float*>(base + em.expert_scale_off);
        for (uint32_t e = 0U; e < em.num_experts; ++e) es[e] = 0.02f;
    } else {
        /* float8_e4m3fn small finite values (+/- 0.5..1.5, never 0x7F/0xFF NaN) */
        static const uint8_t kSafeE4M3[8] = {0x30, 0x34, 0x38, 0x3C, 0xB0, 0xB4, 0xB8, 0xBC};
        for (uint64_t i = 0ULL; i < em.weight_bytes; ++i) {
            w[i] = kSafeE4M3[tcli_splitmix64(rng) & 7ULL];
        }
        /* UE8M0 123 => 2^(123 - 127) = 0.0625f */
        std::memset(s, 123, static_cast<size_t>(em.scales_bytes));
    }
}

static bool synthesize_mini_inkling2_v1_bundle(const std::string& out_dir) {
    ::mkdir(out_dir.c_str(), 0755);

    auto hdr = std::make_unique<StrataInkling2Header>();
    std::memset(hdr.get(), 0, sizeof(*hdr));

    StrataInkling2Config& c = hdr->cfg;
    c.num_phys_layers         = 6U;
    c.num_logical_layers      = 10U; /* 2 prelude + 3*2 core + 2 coda */
    c.hidden_size             = 256U;
    c.vocab_size_padded       = 512U;
    c.vocab_size_unpadded     = 512U;
    c.num_attention_heads     = 6U;
    c.num_key_value_heads     = 2U;
    c.head_dim                = 128U;
    c.d_rel                   = 16U;
    c.rel_extent              = 1024U;
    c.sliding_window_size     = 1024U;
    c.sconv_kernel_size       = 4U;
    c.attn_res_block_size     = 2U;
    c.num_experts             = 32U;
    c.num_experts_per_tok     = 8U;
    c.intermediate_size       = 128U;
    c.loop_core_start         = 2U;
    c.loop_core_end           = 4U;
    c.loop_num_passes         = 3U;
    c.loop_gate_rank          = 64U;
    c.log_scaling_n_floor     = 8192U;
    c.max_position_embeddings = 65536U;
    c.eos_token_id            = 511U;
    c.pad_token_id            = 0U;
    c.rms_norm_eps            = 1e-6f;
    c.qk_l2_clip_threshold    = 22.627417f;
    c.rel_l2_clip_threshold   = 8.0f;
    c.log_scaling_alpha       = 0.1f;
    c.mlp_gate_softcap        = 4.0f;
    c.mlp_up_softcap          = 25.0f;
    c.route_scale             = 8.0f;
    c.final_logit_softcapping = 30.0f;

    const uint32_t phys_is_local[6]  = {1U, 0U, 1U, 0U, 1U, 0U};
    const uint32_t phys_exp_dt[6]    = {
        STRATA_INKLING2_DTYPE_NVFP4, STRATA_INKLING2_DTYPE_NVFP4,
        STRATA_INKLING2_DTYPE_NVFP4, STRATA_INKLING2_DTYPE_NVFP4,
        STRATA_INKLING2_DTYPE_MXFP8, STRATA_INKLING2_DTYPE_MXFP8
    };
    const uint32_t l2p[10]  = {0U, 1U, 2U, 3U, 2U, 3U, 2U, 3U, 4U, 5U};
    const uint32_t lpass[10]= {0U, 0U, 0U, 0U, 1U, 1U, 2U, 2U, 0U, 0U};
    uint32_t local_slot = 0U, global_slot = 0U;
    for (uint32_t il = 0U; il < c.num_logical_layers; ++il) {
        const uint32_t p = l2p[il];
        c.logical_to_phys[il]  = p;
        c.logical_pass_idx[il] = lpass[il];
        c.logical_is_local[il] = phys_is_local[p];
        c.logical_kv_slot[il]  = phys_is_local[p] ? (local_slot++) : (global_slot++);
    }
    c.num_local_visits  = local_slot;
    c.num_global_visits = global_slot;

    hdr->magic               = STRATA_INKLING2_MAGIC_U64;
    hdr->version             = STRATA_INKLING2_FORMAT_VERSION_V1;
    hdr->header_padded_bytes = STRATA_INKLING2_HEADER_PADDED_BYTES;

    const uint32_t H     = c.hidden_size;
    const uint32_t NH    = c.num_attention_heads;
    const uint32_t NKV   = c.num_key_value_heads;
    const uint32_t HD    = c.head_dim;
    const uint32_t DR    = c.d_rel;
    const uint32_t KS    = c.sconv_kernel_size;
    const uint32_t E     = c.num_experts;
    const uint32_t IFF   = c.intermediate_size;
    const uint32_t QDIM  = NH * HD;
    const uint32_t KVDIM = NKV * HD;

    uint64_t cur = STRATA_INKLING2_HEADER_PADDED_BYTES;
    hdr->embed_tokens          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_padded, H);
    hdr->embed_norm            = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
    hdr->final_norm            = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
    hdr->output_res_query      = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
    hdr->unembed               = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_unpadded, H);
    const uint32_t n_cp_rows   = 2U * (c.loop_core_end - c.loop_core_start);
    hdr->cross_pass_queries    = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, n_cp_rows, H);
    hdr->cross_pass_kv_queries = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, n_cp_rows, H);

    for (uint32_t i = 0U; i < c.num_phys_layers; ++i) {
        cur = ik2_align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        StrataInkling2LayerDesc& ld = hdr->layers[i];
        const bool is_local = (phys_is_local[i] != 0U);
        const bool in_core  = (i >= c.loop_core_start && i < c.loop_core_end);
        const uint32_t ext  = is_local ? c.sliding_window_size : c.rel_extent;

        ld.phys_layer_idx = i;
        ld.is_local_swa   = is_local ? 1U : 0U;
        ld.in_loop_core   = in_core ? 1U : 0U;
        ld.expert_dtype   = phys_exp_dt[i];
        ld.layer_page_off = cur;

        if (i > 0U) {
            ld.attn_res_query = is_local
                ? ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U)
                : ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, 3U, H);
        }
        ld.mlp_res_query = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        if (is_local) {
            ld.attn_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        } else {
            ld.q_input_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
            ld.k_input_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
            ld.v_input_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        }
        ld.mlp_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);

        ld.wq             = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
        ld.wk             = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ld.wv             = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ld.wo             = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, QDIM);
        ld.gate_attn      = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
        ld.wr             = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, NH * DR, H);
        ld.attn_sink_proj = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, NH, H);

        ld.q_l2_clip_scale   = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, NH, HD);
        ld.rel_l2_clip_scale = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, NH, DR);
        ld.rel_logits_proj   = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, ext, DR);

        ld.k_sconv_w = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, KVDIM, KS);
        ld.v_sconv_w = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, KVDIM, KS);
        if (!is_local) {
            ld.k_sconv_proj = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, NKV * KS, H);
            ld.v_sconv_proj = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, NKV * KS, H);
        }
        ld.attn_sconv_w = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, KS);
        ld.mlp_sconv_w  = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, KS);

        ld.moe_gate_weight = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, E, H);
        ld.moe_gate_bias   = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, E, 1U);
        ld.experts_w13     = ik2_alloc_experts(&cur, E, 2U * IFF, H, ld.expert_dtype);
        ld.experts_w2      = ik2_alloc_experts(&cur, E, H, IFF, ld.expert_dtype);

        if (in_core) {
            for (uint32_t m = 0U; m < 4U; ++m) {
                ld.controllers[m].down_weight   = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, c.loop_gate_rank, 2U * H);
                ld.controllers[m].gain_weight   = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, c.loop_gate_rank);
                ld.controllers[m].forget_weight = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, c.loop_gate_rank);
            }
        }
        cur = ik2_align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        ld.layer_page_bytes = cur - ld.layer_page_off;
    }

    /* 3-Layer DSpark Block-8 MTP section (FP16 in v1 -> repacked to Q4_A in v2 `-fast`!) */
    cur = ik2_align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
    StrataInkling2MtpDesc& mtp = hdr->mtp;
    mtp.enabled             = 1U;
    mtp.num_layers          = 3U;
    mtp.intermediate_size   = 256U;
    mtp.block_size          = 8U;
    mtp.markov_rank         = 256U;
    mtp.sliding_window      = 1024U;
    mtp.rope_theta          = 1000000.0f;
    mtp.rms_norm_eps        = 1e-6f;
    mtp.target_layer_ids[0] = 2U;
    mtp.target_layer_ids[1] = 4U;
    mtp.target_layer_ids[2] = 6U;
    mtp.target_layer_ids[3] = 10U;
    mtp.mtp_page_off        = cur;

    const uint32_t MR = mtp.markov_rank;
    const uint32_t MI = mtp.intermediate_size;
    mtp.fuse_alpha         = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, mtp.num_layers + 1U, STRATA_INKLING2_MAX_MTP_TAPS);
    mtp.mask_embed         = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, 1U);
    mtp.final_norm         = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
    mtp.markov_dflare_norm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
    mtp.markov_w1          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_padded, MR);
    mtp.markov_rnn_weight  = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, 3U * MR, 2U * MR + 2U * H);
    mtp.markov_rnn_bias    = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, 3U * MR, 1U);
    mtp.markov_hidden_out  = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, MR);
    mtp.markov_hidden_bias = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);

    const uint32_t mtp_local[3] = {1U, 0U, 1U};
    for (uint32_t l = 0U; l < mtp.num_layers; ++l) {
        StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
        ml.layer_idx       = l;
        ml.is_local_swa    = mtp_local[l];
        ml.dflare_norm     = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        ml.input_layernorm = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        ml.post_attn_norm  = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, H, 1U);
        ml.q_norm          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, HD, 1U);
        ml.k_norm          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, HD, 1U);
        ml.k_norm_ctx      = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F32, HD, 1U);
        ml.q_proj          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
        ml.k_proj          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ml.v_proj          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ml.k_proj_ctx      = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ml.v_proj_ctx      = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ml.o_proj          = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, QDIM);
        ml.gate_proj       = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, MI, H);
        ml.up_proj         = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, MI, H);
        ml.down_proj       = ik2_alloc_slice(&cur, STRATA_INKLING2_DTYPE_F16, H, MI);
    }
    cur = ik2_align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
    mtp.mtp_page_bytes = cur - mtp.mtp_page_off;

    hdr->total_file_bytes = ik2_align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
    hdr->active_bytes_per_tok = 1000000ULL;
    hdr->mtp_active_bytes_per_blk = mtp.mtp_page_bytes;

    std::vector<uint8_t> uma(static_cast<size_t>(hdr->total_file_bytes), 0U);
    std::memcpy(uma.data(), hdr.get(), sizeof(*hdr));

    uint64_t rng = 0x494E4B4C494E4732ULL;
    ik2_fill_f16_rand(uma.data(), hdr->embed_tokens, &rng, 0.04f);
    ik2_fill_f32_const(uma.data(), hdr->embed_norm, 1.0f);
    ik2_fill_f32_const(uma.data(), hdr->final_norm, 1.0f);
    ik2_fill_f32_rand(uma.data(), hdr->output_res_query, &rng, 0.05f);
    ik2_fill_f16_rand(uma.data(), hdr->unembed, &rng, 0.05f);
    ik2_fill_f32_rand(uma.data(), hdr->cross_pass_queries, &rng, 0.05f);
    ik2_fill_f32_rand(uma.data(), hdr->cross_pass_kv_queries, &rng, 0.05f);

    for (uint32_t i = 0U; i < c.num_phys_layers; ++i) {
        const StrataInkling2LayerDesc& ld = hdr->layers[i];
        ik2_fill_f32_rand(uma.data(), ld.attn_res_query, &rng, 0.05f);
        ik2_fill_f32_rand(uma.data(), ld.mlp_res_query, &rng, 0.05f);
        ik2_fill_f32_const(uma.data(), ld.attn_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ld.q_input_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ld.k_input_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ld.v_input_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ld.mlp_norm, 1.0f);
        ik2_fill_f16_rand(uma.data(), ld.wq, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.wk, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.wv, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.wo, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.gate_attn, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.wr, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ld.attn_sink_proj, &rng, 0.02f);
        ik2_fill_f32_const(uma.data(), ld.q_l2_clip_scale, 1.0f);
        ik2_fill_f32_const(uma.data(), ld.rel_l2_clip_scale, 1.0f);
        ik2_fill_f32_rand(uma.data(), ld.rel_logits_proj, &rng, 0.05f);
        ik2_fill_f32_rand(uma.data(), ld.k_sconv_w, &rng, 0.15f);
        ik2_fill_f32_rand(uma.data(), ld.v_sconv_w, &rng, 0.15f);
        ik2_fill_f16_rand(uma.data(), ld.k_sconv_proj, &rng, 0.05f);
        ik2_fill_f16_rand(uma.data(), ld.v_sconv_proj, &rng, 0.05f);
        ik2_fill_f32_rand(uma.data(), ld.attn_sconv_w, &rng, 0.15f);
        ik2_fill_f32_rand(uma.data(), ld.mlp_sconv_w, &rng, 0.15f);
        ik2_fill_f32_rand(uma.data(), ld.moe_gate_weight, &rng, 0.05f);
        ik2_fill_f32_const(uma.data(), ld.moe_gate_bias, 0.0f);
        ik2_fill_experts(uma.data(), ld.experts_w13, &rng);
        ik2_fill_experts(uma.data(), ld.experts_w2, &rng);
        if (ld.in_loop_core) {
            for (uint32_t m = 0U; m < 4U; ++m) {
                ik2_fill_f16_rand(uma.data(), ld.controllers[m].down_weight, &rng, 0.02f);
                ik2_fill_f16_rand(uma.data(), ld.controllers[m].gain_weight, &rng, 0.02f);
                ik2_fill_f16_rand(uma.data(), ld.controllers[m].forget_weight, &rng, 0.02f);
            }
        }
    }

    ik2_fill_f32_const(uma.data(), mtp.fuse_alpha, 0.25f);
    ik2_fill_f16_rand(uma.data(), mtp.mask_embed, &rng, 0.03f);
    ik2_fill_f32_const(uma.data(), mtp.final_norm, 1.0f);
    ik2_fill_f32_const(uma.data(), mtp.markov_dflare_norm, 1.0f);
    ik2_fill_f16_rand(uma.data(), mtp.markov_w1, &rng, 0.04f);
    ik2_fill_f16_rand(uma.data(), mtp.markov_rnn_weight, &rng, 0.02f);
    ik2_fill_f32_const(uma.data(), mtp.markov_rnn_bias, 0.0f);
    ik2_fill_f16_rand(uma.data(), mtp.markov_hidden_out, &rng, 0.03f);
    ik2_fill_f32_const(uma.data(), mtp.markov_hidden_bias, 0.0f);
    for (uint32_t l = 0U; l < mtp.num_layers; ++l) {
        const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
        ik2_fill_f32_const(uma.data(), ml.dflare_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ml.input_layernorm, 1.0f);
        ik2_fill_f32_const(uma.data(), ml.post_attn_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ml.q_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ml.k_norm, 1.0f);
        ik2_fill_f32_const(uma.data(), ml.k_norm_ctx, 1.0f);
        ik2_fill_f16_rand(uma.data(), ml.q_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.k_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.v_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.k_proj_ctx, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.v_proj_ctx, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.o_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.gate_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.up_proj, &rng, 0.02f);
        ik2_fill_f16_rand(uma.data(), ml.down_proj, &rng, 0.02f);
    }

    return write_all_bytes(out_dir + "/" STRATA_INKLING2_UMA_FILENAME, uma);
}

static bool run_inkling2_turbo_cli_test(const std::string& v1_dir, const std::string& v2_dir) {
    if (!synthesize_mini_inkling2_v1_bundle(v1_dir)) {
        std::fprintf(stderr, "  [FAIL] inkling2_turbo: failed to synthesize mini v1 bundle\n");
        return false;
    }

    /* 1. CLI repack-inkling2 (default Q4_A 3L DSpark MTP + Q8_0 RNN + FR-Spec) */
    const char* rep_av[] = {
        "strata_cli", "repack-inkling2",
        "--src", v1_dir.c_str(),
        "--dst", v2_dir.c_str(),
        "--threads", "4"
    };
    if (strata_inkling2_cmd_repack(8, const_cast<char**>(rep_av)) != 0) {
        std::fprintf(stderr, "  [FAIL] inkling2_turbo: repack-inkling2 Q4_A MTP failed\n");
        return false;
    }

    /* 1b. Pre-pub P2: verify manifest.json stores only JSON-escaped basename (no $HOME / parent path leak),
     *     including when an absolute source directory path contains a double quote. */
    {
        char cwd_buf[PATH_MAX] = {};
        const std::string abs_prefix = (::getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr)
            ? (std::string(cwd_buf) + "/") : std::string();
        const std::string q_src = abs_prefix + v1_dir + "_p2_\"quoted\"";
        const std::string q_dst = abs_prefix + v2_dir + "_p2_quoted";
        const bool q_syn = synthesize_mini_inkling2_v1_bundle(q_src);
        const char* q_av[] = {
            "strata_cli", "repack-inkling2",
            "--src", q_src.c_str(),
            "--dst", q_dst.c_str(),
            "--threads", "2"
        };
        bool p2_ok = q_syn && (strata_inkling2_cmd_repack(8, const_cast<char**>(q_av)) == 0);
        if (p2_ok) {
            std::ifstream mf(q_dst + "/manifest.json", std::ios::binary);
            const std::string mtxt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
            const char* home = std::getenv("HOME");
            const size_t slash = q_src.find_last_of('/');
            const std::string base = (slash != std::string::npos) ? q_src.substr(slash + 1U) : q_src;
            const std::string parent = (slash != std::string::npos) ? q_src.substr(0U, slash) : std::string();
            std::string esc_base;
            for (char c : base) {
                if (c == '"' || c == '\\') esc_base.push_back('\\');
                esc_base.push_back(c);
            }
            const std::string want_line = "\"source\": \"" + esc_base + "\"";
            if (mtxt.find(want_line) == std::string::npos ||
                (home && home[0] != '\0' && mtxt.find(home) != std::string::npos) ||
                (!parent.empty() && mtxt.find(parent) != std::string::npos)) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo: P2 manifest.json source leak or bad escaping!\n");
                p2_ok = false;
            }
        }
        remove_tree(q_src);
        remove_tree(q_dst);
        if (!p2_ok) return false;
    }

    /* 2. CLI inspect-inkling2 on repacked Q4_A bundle */
    const char* insp_av[] = {"strata_cli", "inspect-inkling2", "--bundle", v2_dir.c_str()};
    if (strata_inkling2_cmd_inspect(4, const_cast<char**>(insp_av)) != 0) {
        std::fprintf(stderr, "  [FAIL] inkling2_turbo: inspect-inkling2 failed\n");
        return false;
    }

    /* 3. Verify Q4_A MTP round-trip header invariants & reduced MTP byte traffic */
    StrataInkling2Bundle bdl_v1{}, bdl_v2{};
    if (strata_inkling2_load_bundle(v1_dir.c_str(), &bdl_v1) != 0 ||
        strata_inkling2_load_bundle(v2_dir.c_str(), &bdl_v2) != 0) {
        std::fprintf(stderr, "  [FAIL] inkling2_turbo: strata_inkling2_load_bundle failed\n");
        strata_inkling2_free_bundle(&bdl_v1);
        strata_inkling2_free_bundle(&bdl_v2);
        return false;
    }
    bool repack_ok = (bdl_v2.hdr.version == STRATA_INKLING2_FORMAT_VERSION &&
                      bdl_v2.hdr.mtp.enabled == 1U &&
                      bdl_v2.hdr.mtp.markov_rnn_weight.dtype == STRATA_INKLING2_DTYPE_Q8_0 &&
                      bdl_v2.hdr.mtp.frspec.enabled == 1U &&
                      bdl_v2.hdr.mtp.frspec.shortlist_size == 512U &&
                      bdl_v2.hdr.mtp_active_bytes_per_blk > 0ULL &&
                      bdl_v2.hdr.mtp_active_bytes_per_blk < bdl_v1.hdr.mtp_active_bytes_per_blk);
    for (uint32_t l = 0U; l < bdl_v2.hdr.mtp.num_layers && repack_ok; ++l) {
        const StrataInkling2MtpLayerDesc& ml = bdl_v2.hdr.mtp.layers[l];
        repack_ok = (ml.q_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.k_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.v_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.k_proj_ctx.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.v_proj_ctx.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.o_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.gate_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.up_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A &&
                     ml.down_proj.dtype == STRATA_INKLING2_DTYPE_Q4_A);
    }
    strata_inkling2_free_bundle(&bdl_v1);
    if (!repack_ok) {
        std::fprintf(stderr, "  [FAIL] inkling2_turbo: Q4_A MTP repack metadata invariant mismatch\n");
        strata_inkling2_free_bundle(&bdl_v2);
        return false;
    }

    /* 4. Verify BOTH `--kv-cache f16` (default lossless) AND `--kv-cache q8`:
     *    - `K=7 == K=0` bit-exact decode (pure greedy + sliding-window penalties)
     *    - Checkpoint save & restore bit-exactness
     *    - `Steady MajFlt == 0`
     */
    const uint32_t prompt[16] = {
        3U, 17U, 31U, 45U, 59U, 73U, 87U, 101U,
        115U, 129U, 143U, 157U, 171U, 185U, 199U, 213U
    };
    uint32_t prompt40[40];
    for (uint32_t i = 0U; i < 40U; ++i) {
        prompt40[i] = 3U + ((i * 11U) % 480U);
    }
    constexpr uint32_t kGenLen = 16U;
    uint64_t f16_global_kv_bytes = 0ULL;
    const char* dis_env = std::getenv("STRATA_METAL_CACHE_DISABLE");
    const bool cache_on = !(dis_env && std::strcmp(dis_env, "1") == 0);

    const uint32_t kv_modes[2] = {STRATA_INKLING2_KV_DTYPE_F16, STRATA_INKLING2_KV_DTYPE_Q8_0};
    for (uint32_t mi = 0U; mi < 2U; ++mi) {
        const uint32_t kv_dt = kv_modes[mi];
        const char* kv_lbl   = (kv_dt == STRATA_INKLING2_KV_DTYPE_Q8_0) ? "q8" : "f16";

        StrataInkling2GpuEngine* eng = strata_inkling2_gpu_create_ex(&bdl_v2, 512U, kv_dt);
        if (!eng) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): strata_inkling2_gpu_create_ex failed\n", kv_lbl);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        StrataInkling2GpuInfo info{};
        strata_inkling2_gpu_get_info(eng, &info);
        if (info.kv_dtype != kv_dt || !info.mtp_enabled || strata_inkling2_slot_memory_bytes(eng) == 0U) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): StrataInkling2GpuInfo kv_dtype/mtp/slot_bytes mismatch\n", kv_lbl);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        if (mi == 1U && cache_on && (info.metal_cache_hit != 1U || info.pso_load_ms < 0.0)) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): expected MTLBinaryArchive cache hit on 2nd boot (hit=%u, ms=%.2f)\n",
                         kv_lbl, info.metal_cache_hit, info.pso_load_ms);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        if (kv_dt == STRATA_INKLING2_KV_DTYPE_F16) {
            f16_global_kv_bytes = info.global_kv_bytes_allocated;
        } else if (info.global_kv_bytes_allocated >= f16_global_kv_bytes) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(q8): expected global_kv_bytes_allocated (%llu) < f16 (%llu)\n",
                         static_cast<unsigned long long>(info.global_kv_bytes_allocated),
                         static_cast<unsigned long long>(f16_global_kv_bytes));
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }

        strata_inkling2_gpu_set_adaptive_k(eng, true);
        strata_inkling2_gpu_set_frspec(eng, true);

        auto run_seq_prompt = [&](const uint32_t* p_ids, uint32_t p_len,
                                  uint32_t draft_k, const StrataInkling2SamplingParams& samp,
                                  std::vector<uint32_t>* out_ids) -> bool {
            out_ids->clear();
            strata_inkling2_gpu_reset(eng);
            strata_inkling2_gpu_penalties_reset(eng);
            strata_inkling2_gpu_reset_counters(eng);
            uint32_t first_tok = 0U;
            if (strata_inkling2_gpu_prefill(eng, p_ids, p_len, &samp, &first_tok, nullptr) != 0) return false;
            if (first_tok >= bdl_v2.hdr.cfg.vocab_size_unpadded) return false;
            out_ids->push_back(first_tok);
            uint32_t pending = first_tok;
            while (out_ids->size() < kGenLen) {
                if (draft_k == 0U) {
                    uint32_t nxt = 0U;
                    if (strata_inkling2_gpu_decode_step(eng, pending, &samp, &nxt, nullptr) != 0) return false;
                    if (nxt >= bdl_v2.hdr.cfg.vocab_size_unpadded) return false;
                    out_ids->push_back(nxt);
                    pending = nxt;
                } else {
                    uint32_t em[STRATA_INKLING2_MAX_SPEC_DRAFT_K + 2U] = {};
                    uint32_t ne = 0U, na = 0U;
                    if (strata_inkling2_gpu_spec_decode_step(eng, pending, draft_k, &samp, em, &ne, &na) != 0 ||
                        ne == 0U) {
                        return false;
                    }
                    for (uint32_t i = 0U; i < ne && out_ids->size() < kGenLen; ++i) {
                        if (em[i] >= bdl_v2.hdr.cfg.vocab_size_unpadded) return false;
                        out_ids->push_back(em[i]);
                    }
                    pending = em[ne - 1U];
                }
            }
            return out_ids->size() == kGenLen;
        };
        auto run_seq = [&](uint32_t draft_k, const StrataInkling2SamplingParams& samp,
                           std::vector<uint32_t>* out_ids) -> bool {
            return run_seq_prompt(prompt, 16U, draft_k, samp, out_ids);
        };

        StrataInkling2SamplingParams greedy{};
        greedy.greedy             = true;
        greedy.temperature        = 0.0f;
        greedy.repetition_penalty = 1.0f;
        greedy.rng_seed           = 42ULL;

        StrataInkling2Checkpoint* ckpt = strata_inkling2_checkpoint_create(eng);
        if (!ckpt) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): checkpoint_create failed\n", kv_lbl);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        /* Warmup once (both FP16 and INT8 M>=32 prefill paths) before steady-state MajFlt measurement */
        std::vector<uint32_t> warm_ids;
        if (!run_seq(STRATA_INKLING2_MAX_SPEC_DRAFT_K, greedy, &warm_ids) ||
            strata_inkling2_checkpoint_save(eng, ckpt) != 0 ||
            strata_inkling2_checkpoint_restore(eng, ckpt) != 0) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): warmup failed\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        if (mi == 0U) {
            strata_inkling2_gpu_set_int8_prefill(eng, true);
            if (!run_seq_prompt(prompt40, 40U, 0U, greedy, &warm_ids)) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): int8_prefill warmup failed\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
            strata_inkling2_gpu_set_int8_prefill(eng, false);
        }

        struct rusage ru0{};
        getrusage(RUSAGE_SELF, &ru0);

        /* 4a. Pure Greedy K=7 == K=0 Bit-Exact */
        std::vector<uint32_t> k0_ids, k7_ids;
        if (!run_seq(0U, greedy, &k0_ids) ||
            !run_seq(STRATA_INKLING2_MAX_SPEC_DRAFT_K, greedy, &k7_ids) ||
            k0_ids != k7_ids) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): greedy K=7 != K=0 mismatch\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }

        /* 4a2. `--int8-prefill` (M=40 >= 32), `--no-adaptive-k`, `--no-frspec` toggles & K=7==K=0 exactness */
        if (mi == 0U) {
            strata_inkling2_gpu_set_int8_prefill(eng, true);
            strata_inkling2_gpu_set_adaptive_k(eng, false);
            strata_inkling2_gpu_set_frspec(eng, false);
            StrataInkling2GpuInfo info_tog{};
            strata_inkling2_gpu_get_info(eng, &info_tog);
            if (info_tog.int8_prefill_enabled != 1U ||
                strata_inkling2_gpu_get_adaptive_k(eng) ||
                strata_inkling2_gpu_get_frspec(eng)) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): toggle state mismatch\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
            std::vector<uint32_t> i8_k0, i8_k7;
            if (!run_seq_prompt(prompt40, 40U, 0U, greedy, &i8_k0) ||
                !run_seq_prompt(prompt40, 40U, STRATA_INKLING2_MAX_SPEC_DRAFT_K, greedy, &i8_k7) ||
                i8_k0 != i8_k7) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): int8_prefill + no-adaptive-k + no-frspec K=7 != K=0 mismatch\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
            strata_inkling2_gpu_set_int8_prefill(eng, false);
            strata_inkling2_gpu_set_adaptive_k(eng, true);
            strata_inkling2_gpu_set_frspec(eng, true);
            strata_inkling2_gpu_get_info(eng, &info_tog);
            if (info_tog.int8_prefill_enabled != 0U ||
                !strata_inkling2_gpu_get_adaptive_k(eng) ||
                !strata_inkling2_gpu_get_frspec(eng)) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): toggle restore mismatch\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
        }

        /* 4b. Sliding-Window Penalties K=7 == K=0 Bit-Exact */
        StrataInkling2SamplingParams pen = greedy;
        pen.presence_penalty   = 1.2f;
        pen.frequency_penalty  = 0.25f;
        pen.repetition_penalty = 1.10f;
        pen.penalty_window     = 8U;
        std::vector<uint32_t> pk0_ids, pk7_ids;
        if (!run_seq(0U, pen, &pk0_ids) ||
            !run_seq(STRATA_INKLING2_MAX_SPEC_DRAFT_K, pen, &pk7_ids) ||
            pk0_ids != pk7_ids) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): penalty K=7 != K=0 mismatch\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }

        /* 4c. Checkpoint Save/Restore Bit-Exact Decode (prefill -> save -> decode K=7 -> restore -> decode K=0) */
        strata_inkling2_gpu_reset(eng);
        strata_inkling2_gpu_penalties_reset(eng);
        uint32_t c_first = 0U;
        if (strata_inkling2_gpu_prefill(eng, prompt, 16U, &greedy, &c_first, nullptr) != 0 ||
            strata_inkling2_checkpoint_save(eng, ckpt) != 0) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): checkpoint save after prefill failed\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        const uint32_t saved_pos = strata_inkling2_gpu_seq_pos(eng);
        std::vector<uint32_t> ck_k7 = {c_first};
        uint32_t pend = c_first;
        while (ck_k7.size() < kGenLen) {
            uint32_t em[STRATA_INKLING2_MAX_SPEC_DRAFT_K + 2U] = {};
            uint32_t ne = 0U, na = 0U;
            if (strata_inkling2_gpu_spec_decode_step(
                    eng, pend, STRATA_INKLING2_MAX_SPEC_DRAFT_K, &greedy, em, &ne, &na) != 0 || ne == 0U) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): post-save K=7 decode failed\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
            for (uint32_t i = 0U; i < ne && ck_k7.size() < kGenLen; ++i) ck_k7.push_back(em[i]);
            pend = em[ne - 1U];
        }
        if (strata_inkling2_checkpoint_restore(eng, ckpt) != 0 ||
            strata_inkling2_gpu_seq_pos(eng) != saved_pos) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): checkpoint restore / seq_pos mismatch\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
        std::vector<uint32_t> ck_k0 = {c_first};
        pend = c_first;
        while (ck_k0.size() < kGenLen) {
            uint32_t nxt = 0U;
            if (strata_inkling2_gpu_decode_step(eng, pend, &greedy, &nxt, nullptr) != 0) {
                std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): post-restore K=0 decode failed\n", kv_lbl);
                strata_inkling2_checkpoint_destroy(ckpt);
                strata_inkling2_gpu_destroy(eng);
                strata_inkling2_free_bundle(&bdl_v2);
                return false;
            }
            ck_k0.push_back(nxt);
            pend = nxt;
        }
        if (ck_k7 != k0_ids || ck_k0 != k0_ids) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): checkpoint restore decode not bit-exact\n", kv_lbl);
            strata_inkling2_checkpoint_destroy(ckpt);
            strata_inkling2_gpu_destroy(eng);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }

        struct rusage ru1{};
        getrusage(RUSAGE_SELF, &ru1);
        const long steady_majflt = std::max<long>(0L, ru1.ru_majflt - ru0.ru_majflt);

        strata_inkling2_checkpoint_destroy(ckpt);
        strata_inkling2_gpu_destroy(eng);

        if (steady_majflt != 0L) {
            std::fprintf(stderr, "  [FAIL] inkling2_turbo(%s): non-zero steady MajFlt=%ld\n", kv_lbl, steady_majflt);
            strata_inkling2_free_bundle(&bdl_v2);
            return false;
        }
    }

    strata_inkling2_free_bundle(&bdl_v2);
    return true;
}

template <size_t K>
static double tcli_ngram_dup_ratio(const std::vector<uint32_t>& ids) {
    if (ids.size() < K) return 0.0;
    const size_t total = ids.size() - K + 1U;
    std::vector<std::array<uint32_t, K>> grams(total);
    for (size_t i = 0U; i < total; ++i) {
        for (size_t j = 0U; j < K; ++j) grams[i][j] = ids[i + j];
    }
    std::sort(grams.begin(), grams.end());
    size_t uniq = 1U;
    for (size_t i = 1U; i < total; ++i) {
        if (grams[i] != grams[i - 1U]) ++uniq;
    }
    return 1.0 - (static_cast<double>(uniq) / static_cast<double>(total));
}

static bool run_shingle_loop_sanity_unit_test() {
    std::vector<uint32_t> short_seq(32U, 7U);
    if (short_seq.size() >= 48U) return false;

    std::vector<uint32_t> diverse(64U);
    for (uint32_t i = 0U; i < 64U; ++i) diverse[i] = 100U + i;
    const double d8_div = tcli_ngram_dup_ratio<8U>(diverse);
    const double d4_div = tcli_ngram_dup_ratio<4U>(diverse);
    if (d8_div > 0.01 || d4_div > 0.01) return false;

    std::vector<uint32_t> looping(64U);
    for (uint32_t i = 0U; i < 64U; ++i) looping[i] = 1000U + (i % 8U);
    const double d8_loop = tcli_ngram_dup_ratio<8U>(looping);
    const double d4_loop = tcli_ngram_dup_ratio<4U>(looping);
    if (!(d8_loop > 0.35) || !(d4_loop > 0.35)) return false;

    return true;
}

} /* namespace */

int main() {
    @autoreleasepool {
        std::printf("================================================================================\n");
        std::printf("  STRATA-M5 v2.0 — End-to-End 4-Axis Converter & Speculative CLI Test Suite\n");
        std::printf("================================================================================\n");

        const std::string bench_root = "bench_data";
        ::mkdir(bench_root.c_str(), 0755);

        const std::string ds_st_dir  = bench_root + "/cli_ds_st";
        const std::string ds_bdl_dir = bench_root + "/cli_ds_bundle";
        const std::string qw_st_dir  = bench_root + "/cli_qw_st";
        const std::string qw_bdl_dir = bench_root + "/cli_qw_bundle";
        const std::string glm_st_dir = bench_root + "/cli_glm_st";
        const std::string glm_bdl_dir = bench_root + "/cli_glm_bundle";
        const std::string glm53_v2_dir = bench_root + "/cli_glm53_v2_bundle";
        const std::string ik2_v1_dir   = bench_root + "/cli_inkling2_v1";
        const std::string ik2_v2_dir   = bench_root + "/cli_inkling2_v2";

        auto cleanup_all = [&]() {
            remove_tree(ds_st_dir);
            remove_tree(ds_bdl_dir);
            remove_tree(qw_st_dir);
            remove_tree(qw_bdl_dir);
            remove_tree(glm_st_dir);
            remove_tree(glm_bdl_dir);
            remove_tree(glm53_v2_dir);
            remove_tree(ik2_v1_dir);
            remove_tree(ik2_v2_dir);
        };
        cleanup_all();

        int failures = 0;

        /* --------------------------------------------------------------------
         * [1/4] Synthesize `.safetensors` & Convert All 3 Regimes to `.strata`
         * -------------------------------------------------------------------- */
        std::printf("[1/4] Synthesizing `mini-deepseek`, `mini-qwen`, `mini-glm` & Converting 4-Axis `.strata`...\n");
        StrataConvertStats ds_stats{}, qw_stats{}, glm_stats{};
        bool conv_ok = true;

        if (strata_synth_safetensors_checkpoint(ds_st_dir.c_str(), "mini-deepseek", 42ULL) != 0 ||
            strata_convert_safetensors_to_bundle(ds_st_dir.c_str(), ds_bdl_dir.c_str(), &ds_stats) != 0 ||
            !ds_stats.rans_roundtrip_verified || !ds_stats.lba_alignment_verified ||
            !verify_bundle_lba_and_rans(ds_bdl_dir, /*expect_cold_extents=*/true)) {
            conv_ok = false;
        }

        if (strata_synth_safetensors_checkpoint(qw_st_dir.c_str(), "mini-qwen", 42ULL) != 0 ||
            strata_convert_safetensors_to_bundle(qw_st_dir.c_str(), qw_bdl_dir.c_str(), &qw_stats) != 0 ||
            !qw_stats.rans_roundtrip_verified || !qw_stats.lba_alignment_verified ||
            !verify_bundle_lba_and_rans(qw_bdl_dir, /*expect_cold_extents=*/false)) {
            conv_ok = false;
        }

        if (strata_synth_safetensors_checkpoint(glm_st_dir.c_str(), "mini-glm", 42ULL) != 0 ||
            strata_convert_safetensors_to_bundle(glm_st_dir.c_str(), glm_bdl_dir.c_str(), &glm_stats) != 0 ||
            !glm_stats.rans_roundtrip_verified || !glm_stats.lba_alignment_verified ||
            !verify_bundle_lba_and_rans(glm_bdl_dir, /*expect_cold_extents=*/true)) {
            conv_ok = false;
        }

        /* Remove raw .safetensors immediately once converted to keep disk usage minimal */
        remove_tree(ds_st_dir);
        remove_tree(qw_st_dir);
        remove_tree(glm_st_dir);

        std::printf("  mini-deepseek : %u tensors -> Axis A %.2f MiB | Axis B %.2f MiB | Axis C %.2f MiB | Axis D %.2f MiB (%u cold)\n",
                    ds_stats.num_safetensors_tensors,
                    static_cast<double>(ds_stats.axis_a_ngram_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(ds_stats.axis_b_spine_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(ds_stats.axis_c_uma_hot_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(ds_stats.axis_d_cold_strata_bytes) / (1024.0 * 1024.0),
                    ds_stats.num_cold_experts_total);
        std::printf("  mini-qwen     : %u tensors -> 100%% UMA + PLE (%.2f MiB) | Tile-rANS & 256 KiB Align Verified\n",
                    qw_stats.num_safetensors_tensors,
                    static_cast<double>(qw_stats.axis_a_ngram_bytes) / (1024.0 * 1024.0));
        std::printf("  mini-glm      : %u tensors -> BitNest QTIP + 4-lvl Refine (%u cold extents, %.2f MiB)\n",
                    glm_stats.num_safetensors_tensors,
                    glm_stats.num_cold_experts_total,
                    static_cast<double>(glm_stats.axis_d_cold_strata_bytes) / (1024.0 * 1024.0));
        std::printf("  4-Axis Pack & Tile-rANS Round-Trip Gate : %s\n\n", conv_ok ? "PASS" : "FAIL");
        if (!conv_ok) ++failures;

        /* --------------------------------------------------------------------
         * [2/4] Exactness Gate 1: Greedy Equivalence (`K=0` vs `K=5` Mode A, 24 tokens)
         * -------------------------------------------------------------------- */
        std::printf("[2/4] Exactness Gate 1 — Greedy Equivalence (`K=0` vs `K=5` STRATA_MODE_A_STATIONARY, 24 tokens)...\n");
        StrataInferEngine* eng_ds = strata_infer_create(ds_bdl_dir.c_str());
        if (!eng_ds) {
            std::fprintf(stderr, "FATAL: strata_infer_create failed on mini-deepseek\n");
            cleanup_all();
            return 1;
        }

        const char* test_prompt = "STRATA executes four-axis speculative decoding on Apple M5 Max";
        constexpr uint32_t kGreedyLen = 24U;
        std::vector<uint32_t> toks_k0(kGreedyLen, 0U);
        std::vector<uint32_t> toks_k5(kGreedyLen, 0U);
        uint32_t gen_k0 = 0U, gen_k5 = 0U;
        StrataInferTelemetry tel_k0{}, tel_k5{};

        StrataInferOptions opt_k0 = strata_infer_default_options();
        opt_k0.mode        = STRATA_MODE_A_STATIONARY;
        opt_k0.max_tokens  = kGreedyLen;
        opt_k0.draft_k     = 0U;
        opt_k0.greedy      = true;
        opt_k0.temperature = 0.0f;
        opt_k0.seed        = 42ULL;
        opt_k0.stop_on_eos = false;

        StrataInferOptions opt_k5 = opt_k0;
        opt_k5.draft_k     = 5U;

        const int rc_k0 = strata_infer_generate(
            eng_ds, test_prompt, &opt_k0, toks_k0.data(), kGreedyLen, &gen_k0, nullptr, nullptr, &tel_k0);
        const int rc_k5 = strata_infer_generate(
            eng_ds, test_prompt, &opt_k5, toks_k5.data(), kGreedyLen, &gen_k5, nullptr, nullptr, &tel_k5);

        bool gate1_identical = (rc_k0 == 0 && rc_k5 == 0 && gen_k0 == kGreedyLen && gen_k5 == kGreedyLen);
        uint32_t first_mismatch = kGreedyLen;
        if (gate1_identical) {
            for (uint32_t i = 0U; i < kGreedyLen; ++i) {
                if (toks_k0[i] != toks_k5[i]) {
                    gate1_identical = false;
                    first_mismatch  = i;
                    break;
                }
            }
        }

        const StrataCheckpointBundle* bdl_ds = strata_infer_get_bundle(eng_ds);
        const std::string text_k5 = decode_tokens_to_text(bdl_ds, toks_k5.data(), gen_k5);
        std::printf("  K=0 Baseline   : %u toks in %2u steps (%6.2f tok/s | MajFlt=%ld)\n",
                    gen_k0, tel_k0.decode_steps, tel_k0.tokens_per_sec, tel_k0.steady_major_faults);
        std::printf("  K=5 Mode A Spec: %u toks in %2u steps (%6.2f tok/s | %.2f accepted/step | draft hit %.1f%% | MajFlt=%ld)\n",
                    gen_k5, tel_k5.decode_steps, tel_k5.tokens_per_sec,
                    tel_k5.avg_accepted_per_step, 100.0 * tel_k5.draft_accept_rate,
                    tel_k5.steady_major_faults);
        std::printf("  Generated Text : \"%s\"\n", text_k5.c_str());

        if (!gate1_identical) {
            std::printf("  Exactness Gate 1 (24-tok Bit-Identical) : FAIL (mismatch at idx %u: k0=%u vs k5=%u)\n\n",
                        first_mismatch,
                        (first_mismatch < kGreedyLen) ? toks_k0[first_mismatch] : 0U,
                        (first_mismatch < kGreedyLen) ? toks_k5[first_mismatch] : 0U);
            ++failures;
        } else {
            std::printf("  Exactness Gate 1 (24-tok Bit-Identical) : PASS (24/24 exact token ID match!)\n\n");
        }

        /* --------------------------------------------------------------------
         * [3/4] Exactness Gate 2: Tri-Engine Co-Execution & Mode B Zero-Stall
         * -------------------------------------------------------------------- */
        std::printf("[3/4] Exactness Gate 2 — Tri-Engine Co-Execution (`40c NAX` + `16c ANE` + `3x SME2` + `NVMe`) & Mode B Zero-Stall...\n");

        StrataInferOptions opt_b = opt_k5;
        opt_b.mode = STRATA_MODE_B_ZERO_STALL;
        std::vector<uint32_t> toks_b(kGreedyLen, 0U);
        uint32_t gen_b = 0U;
        StrataInferTelemetry tel_b{};
        const int rc_b = strata_infer_generate(
            eng_ds, test_prompt, &opt_b, toks_b.data(), kGreedyLen, &gen_b, nullptr, nullptr, &tel_b);

        /* Also exercise stochastic Leviathan sampling (T=0.7, K=5) on mini-deepseek */
        StrataInferOptions opt_stoch = strata_infer_default_options();
        opt_stoch.mode        = STRATA_MODE_A_STATIONARY;
        opt_stoch.max_tokens  = 24U;
        opt_stoch.draft_k     = 5U;
        opt_stoch.greedy      = false;
        opt_stoch.temperature = 0.7f;
        opt_stoch.seed        = 20261118ULL;
        opt_stoch.stop_on_eos = false;
        std::vector<uint32_t> toks_stoch(24U, 0U);
        uint32_t gen_stoch = 0U;
        StrataInferTelemetry tel_stoch{};
        const int rc_stoch = strata_infer_generate(
            eng_ds, test_prompt, &opt_stoch, toks_stoch.data(), 24U, &gen_stoch, nullptr, nullptr, &tel_stoch);

        strata_infer_destroy(eng_ds);

        /* Also verify Regime I BitNest QTIP (`mini-glm`) & Regime I 100% UMA + PLE (`mini-qwen`) */
        StrataInferEngine* eng_glm = strata_infer_create(glm_bdl_dir.c_str());
        StrataInferTelemetry tel_glm_b{};
        uint32_t gen_glm = 0U;
        std::vector<uint32_t> toks_glm(16U, 0U);
        int rc_glm = -1;
        if (eng_glm) {
            StrataInferOptions opt_glm = opt_b;
            opt_glm.max_tokens = 16U;
            rc_glm = strata_infer_generate(
                eng_glm, test_prompt, &opt_glm, toks_glm.data(), 16U, &gen_glm, nullptr, nullptr, &tel_glm_b);
            strata_infer_destroy(eng_glm);
        }

        StrataInferEngine* eng_qw = strata_infer_create(qw_bdl_dir.c_str());
        StrataInferTelemetry tel_qw{};
        uint32_t gen_qw0 = 0U, gen_qw5 = 0U;
        std::vector<uint32_t> qw_k0(16U, 0U), qw_k5(16U, 0U);
        bool qw_exact = false;
        if (eng_qw) {
            StrataInferOptions oq0 = opt_k0;
            oq0.max_tokens = 16U;
            StrataInferOptions oq5 = opt_k5;
            oq5.max_tokens = 16U;
            if (strata_infer_generate(eng_qw, test_prompt, &oq0, qw_k0.data(), 16U, &gen_qw0, nullptr, nullptr, nullptr) == 0 &&
                strata_infer_generate(eng_qw, test_prompt, &oq5, qw_k5.data(), 16U, &gen_qw5, nullptr, nullptr, &tel_qw) == 0 &&
                gen_qw0 == 16U && gen_qw5 == 16U) {
                qw_exact = (std::memcmp(qw_k0.data(), qw_k5.data(), 16U * sizeof(uint32_t)) == 0);
            }
            strata_infer_destroy(eng_qw);
        }

        const bool ane_ok = tel_k5.native_metal_gpu
            ? (tel_k5.ran_on_ane && tel_k5.compute_plan_all_ane && tel_k5.ane_probe_calls > 0ULL &&
               tel_b.ran_on_ane && tel_b.ane_probe_calls > 0ULL)
            : (tel_k5.ane_probe_calls > 0ULL);
        const bool sme2_ok = (tel_k5.sme2_phase1_calls > 0ULL && tel_k5.hardware_sme2_active &&
                              tel_b.sme2_phase1_calls > 0ULL);
        const bool ring_ok = (tel_k5.ring_extents_streamed > 0ULL && tel_k5.ring_bytes_streamed > 0ULL &&
                              tel_b.ring_extents_streamed > 0ULL && tel_b.ring_bytes_streamed > 0ULL);
        const bool fault_ok = (tel_k0.steady_major_faults == 0 &&
                               tel_k5.steady_major_faults == 0 &&
                               tel_b.steady_major_faults == 0 &&
                               tel_stoch.steady_major_faults == 0 &&
                               tel_glm_b.steady_major_faults == 0 &&
                               tel_qw.steady_major_faults == 0);
        const bool nan_ok = (tel_k0.logits_nan_inf_count == 0ULL &&
                             tel_k5.logits_nan_inf_count == 0ULL &&
                             tel_b.logits_nan_inf_count == 0ULL &&
                             tel_stoch.logits_nan_inf_count == 0ULL &&
                             tel_glm_b.logits_nan_inf_count == 0ULL &&
                             tel_qw.logits_nan_inf_count == 0ULL);
        const bool mode_b_zero_stall_ok =
            (rc_b == 0 && gen_b == kGreedyLen && tel_b.ring_wait_calls == 0ULL && tel_b.ring_wait_us == 0.0 &&
             rc_glm == 0 && gen_glm == 16U && tel_glm_b.ring_wait_calls == 0ULL &&
             rc_stoch == 0 && gen_stoch == 24U && qw_exact);

        std::printf("  16-Core ANE Window Probe Gate  : %s (calls=%llu, ran_on_ane=%d, plan_all_ane=%d, SQNR=%.2f dB)\n",
                    ane_ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(tel_k5.ane_probe_calls),
                    tel_k5.ran_on_ane ? 1 : 0,
                    tel_k5.compute_plan_all_ane ? 1 : 0,
                    tel_k5.ane_vs_sme2_sqnr_db);
        std::printf("  3x ARMv9.2 SME2 Phase-1 Gate   : %s (calls=%llu, hw_sme2=%d, I/O=%.1f us, SME2=%.1f us)\n",
                    sme2_ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(tel_k5.sme2_phase1_calls),
                    tel_k5.hardware_sme2_active ? 1 : 0,
                    tel_k5.sme2_io_us, tel_k5.sme2_compute_us);
        std::printf("  MTLIOCommandQueue Ring DMA Gate: %s (Mode A=%llu exts [%.2f MiB], Mode B=%llu exts [%.2f MiB])\n",
                    ring_ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(tel_k5.ring_extents_streamed),
                    static_cast<double>(tel_k5.ring_bytes_streamed) / (1024.0 * 1024.0),
                    static_cast<unsigned long long>(tel_b.ring_extents_streamed),
                    static_cast<double>(tel_b.ring_bytes_streamed) / (1024.0 * 1024.0));
        std::printf("  Mode B Zero-Stall & Multi-Arch : %s (DeepSeek waits=%llu, fb=%llu | GLM QTIP waits=%llu | Qwen K0==K5=%d)\n",
                    mode_b_zero_stall_ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(tel_b.ring_wait_calls),
                    static_cast<unsigned long long>(tel_b.moe_svd_qtip_fallbacks),
                    static_cast<unsigned long long>(tel_glm_b.ring_wait_calls),
                    qw_exact ? 1 : 0);
        std::printf("  Zero Page-Fault & Zero NaN Gate: %s (Steady MajFlt=%ld, NaN/Inf=%llu)\n\n",
                    (fault_ok && nan_ok) ? "PASS" : "FAIL",
                    tel_k5.steady_major_faults + tel_b.steady_major_faults,
                    static_cast<unsigned long long>(tel_k5.logits_nan_inf_count + tel_b.logits_nan_inf_count));

        if (!ane_ok || !sme2_ok || !ring_ok || !fault_ok || !nan_ok || !mode_b_zero_stall_ok) {
            ++failures;
        }

        /* --------------------------------------------------------------------
         * [4/4] Peak RSS <= 4.0 GiB, Chat & GLM-5.3 CLI Unit Suite & `bench_data/` (`0` B)
         * -------------------------------------------------------------------- */
        std::printf("[4/4] Auditing Multi-Turn Chat REPL, GLM-5.3 Regime II CLI, Peak Process RSS & Disk Hygiene...\n");
        const int chat_unit_fails    = strata_chat_repl_unit_tests();
        const int server_unit_fails  = strata_openai_server_unit_tests();
        const int ik2_cli_unit_fails = strata_inkling2_cli_unit_tests();
        const char* help_chat[]       = {"strata_cli", "chat", "--help"};
        const char* help_flash[]      = {"strata_cli", "chat-flash", "--help"};
        const char* help_hf[]         = {"strata_cli", "chat-hf", "--help"};
        const char* help_chat_glm53[] = {"strata_cli", "chat-glm53", "--help"};
        const char* help_chat_ik2[]   = {"strata_cli", "chat-inkling2", "--help"};
        const char* help_serve[]      = {"strata_cli", "serve", "--help"};
        const char* help_run_fl[]      = {"strata_cli", "run-flash", "--help"};
        const char* help_bnch_fl[]     = {"strata_cli", "bench-flash", "--help"};
        const char* help_conv_glm53[]  = {"strata_cli", "convert-glm53", "--help"};
        const char* help_cfast_glm53[] = {"strata_cli", "convert-glm53-fast", "--help"};
        const char* help_insp_glm53[]  = {"strata_cli", "inspect-glm53", "--help"};
        const char* help_run_glm53[]   = {"strata_cli", "run-glm53", "--help"};
        const char* help_bnch_glm53[]  = {"strata_cli", "bench-glm53", "--help"};
        const char* help_conv_ik2[]    = {"strata_cli", "convert-inkling2", "--help"};
        const char* help_cvis_ik2[]    = {"strata_cli", "convert-inkling2-vision", "--help"};
        const char* help_rep_ik2[]     = {"strata_cli", "repack-inkling2", "--help"};
        const char* help_insp_ik2[]    = {"strata_cli", "inspect-inkling2", "--help"};
        const char* help_run_ik2[]     = {"strata_cli", "run-inkling2", "--help"};
        const char* help_bnch_ik2[]    = {"strata_cli", "bench-inkling2", "--help"};
        const bool chat_help_ok =
            (strata_cmd_chat(3, const_cast<char**>(help_chat)) == 0) &&
            (strata_cmd_chat_flash(3, const_cast<char**>(help_flash)) == 0) &&
            (strata_cmd_chat_hf(3, const_cast<char**>(help_hf)) == 0) &&
            (strata_cmd_chat_glm53(3, const_cast<char**>(help_chat_glm53)) == 0) &&
            (strata_cmd_chat_inkling2(3, const_cast<char**>(help_chat_ik2)) == 0) &&
            (strata_cmd_serve(3, const_cast<char**>(help_serve)) == 0) &&
            (strata_qwen_flash_cmd_run(3, const_cast<char**>(help_run_fl)) == 0) &&
            (strata_qwen_flash_cmd_bench(3, const_cast<char**>(help_bnch_fl)) == 0) &&
            (strata_glm53_cmd_convert(3, const_cast<char**>(help_conv_glm53)) == 0) &&
            (strata_glm53_cmd_convert_fast(3, const_cast<char**>(help_cfast_glm53)) == 0) &&
            (strata_glm53_cmd_inspect(3, const_cast<char**>(help_insp_glm53)) == 0) &&
            (strata_glm53_cmd_run(3, const_cast<char**>(help_run_glm53)) == 0) &&
            (strata_glm53_cmd_bench(3, const_cast<char**>(help_bnch_glm53)) == 0) &&
            (strata_inkling2_cmd_convert(3, const_cast<char**>(help_conv_ik2)) == 0) &&
            (strata_inkling2_cmd_convert_vision(3, const_cast<char**>(help_cvis_ik2)) == 0) &&
            (strata_inkling2_cmd_repack(3, const_cast<char**>(help_rep_ik2)) == 0) &&
            (strata_inkling2_cmd_inspect(3, const_cast<char**>(help_insp_ik2)) == 0) &&
            (strata_inkling2_cmd_run(3, const_cast<char**>(help_run_ik2)) == 0) &&
            (strata_inkling2_cmd_bench(3, const_cast<char**>(help_bnch_ik2)) == 0);

        StrataGlm53Config gcfg{};
        strata_glm53_default_config(&gcfg);
        auto ghdr = std::make_unique<StrataGlm53SpineHeader>();
        auto ghdr_v2 = std::make_unique<StrataGlm53SpineHeader>();
        strata_glm53_init_layout(&gcfg, ghdr.get());
        strata_glm53_init_layout_v2(&gcfg, ghdr_v2.get());
        const bool glm53_layout_ok =
            (ghdr->magic == STRATA_GLM53_SPINE_MAGIC_U64) &&
            (ghdr->version == STRATA_GLM53_FORMAT_VERSION_V1) &&
            (ghdr->total_spine_bytes == 5756420096ULL) &&
            ((ghdr->total_spine_bytes % STRATA_GLM53_PAGE_BYTES) == 0ULL) &&
            (ghdr->experts_nvme_bytes == 175305129984ULL) &&
            ((ghdr->experts_nvme_bytes % STRATA_GLM53_NVME_STRIPE_BYTES) == 0ULL) &&
            (ghdr->total_routed_experts == STRATA_GLM53_CANONICAL_TOTAL_EXPERTS) &&
            (ghdr->mtp.enabled == 1U) &&
            (ghdr_v2->magic == STRATA_GLM53_SPINE_MAGIC_U64) &&
            (ghdr_v2->version == STRATA_GLM53_FORMAT_VERSION_V2) &&
            (ghdr_v2->is_ram_resident == 1U) &&
            (ghdr_v2->main_expert_dtype == STRATA_GLM53_EXPERT_DTYPE_E8S18_V4) &&
            (ghdr_v2->part0_bytes == 45185236992ULL) &&
            (ghdr_v2->part1_bytes == 45185236992ULL) &&
            (ghdr_v2->mtp_experts_bytes == 4076863488ULL) &&
            (ghdr_v2->experts_nvme_bytes == 0ULL);

        const bool glm53_v2_e2e_ok = run_glm53_v2_ram_resident_cli_test(glm53_v2_dir);
        remove_tree(glm53_v2_dir);

        const bool inkling2_turbo_e2e_ok = run_inkling2_turbo_cli_test(ik2_v1_dir, ik2_v2_dir);
        /* Review 5 #6: Inkling-2 bundle validator must reject OOB logical_to_phys and frspec without MTP */
        bool ik2_sec_ok = true;
        if (inkling2_turbo_e2e_ok) {
            const std::string ik2_src_uma = ik2_v2_dir + "/" + STRATA_INKLING2_UMA_FILENAME;
            std::ifstream ifs(ik2_src_uma, std::ios::binary);
            std::vector<uint8_t> base_uma((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            if (base_uma.size() >= sizeof(StrataInkling2Header)) {
                const std::string bad_ik2_dir = bench_root + "/cli_ik2_sec_bad";
                (void)::mkdir(bad_ik2_dir.c_str(), 0755);
                auto test_tamper = [&](const char* label, auto mutator) {
                    std::vector<uint8_t> mod = base_uma;
                    auto* hh = reinterpret_cast<StrataInkling2Header*>(mod.data());
                    mutator(*hh);
                    std::ofstream ofs(bad_ik2_dir + "/" + STRATA_INKLING2_UMA_FILENAME, std::ios::binary);
                    ofs.write(reinterpret_cast<const char*>(mod.data()), static_cast<std::streamsize>(mod.size()));
                    ofs.close();
                    StrataInkling2Bundle bdl{};
                    if (strata_inkling2_load_bundle(bad_ik2_dir.c_str(), &bdl) == 0) {
                        strata_inkling2_free_bundle(&bdl);
                        std::fprintf(stderr, "  [FAIL] security_gate: inkling2 loader accepted %s!\n", label);
                        ik2_sec_ok = false;
                    }
                };
                test_tamper("OOB logical_to_phys", [](StrataInkling2Header& hh) {
                    hh.cfg.logical_to_phys[0] = 99U;
                });
                test_tamper("frspec.enabled with mtp.enabled=0", [](StrataInkling2Header& hh) {
                    hh.mtp.enabled = 0U;
                    hh.mtp.frspec.enabled = 1U;
                });
                test_tamper("oversized experts_w13 rows", [](StrataInkling2Header& hh) {
                    hh.layers[0].experts_w13.num_experts = hh.cfg.num_experts + 1U;
                });
                remove_tree(bad_ik2_dir);
            } else {
                ik2_sec_ok = false;
            }
        }
        remove_tree(ik2_v1_dir);
        remove_tree(ik2_v2_dir);

        /* Step 4 ABI, Phase B Spec Tree/Sync & Phase C Dense Quant Layout Invariants */
        StrataQwenFlashCompactionStats cstats{};
        strata_qwen_flash_slot_get_stats(nullptr, &cstats);
        char ane_note[128] = {};
        StrataQwenFlashConfig fcfg{};
        strata_qwen_flash_default_config(&fcfg);
        auto fh0_fp16 = std::make_unique<StrataQwenFlashPart0Header>();
        auto fh1_fp16 = std::make_unique<StrataQwenFlashPart1Header>();
        auto fh0_fast = std::make_unique<StrataQwenFlashPart0Header>();
        auto fh1_fast = std::make_unique<StrataQwenFlashPart1Header>();
        strata_qwen_flash_init_layout(&fcfg, fh0_fp16.get(), fh1_fp16.get(), nullptr);
        strata_qwen_flash_init_layout_quant(&fcfg, true, fh0_fast.get(), fh1_fast.get(), nullptr);
        const StrataQwenFlashQuantMatrix sh_qm = sqf_q4_slice_as_qmat(&fh0_fast->layers[0].shared_gate_proj);
        const bool phase_c_layout_ok =
            (fh0_fast->layers[0].attn_hc.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) &&
            (fh0_fast->layers[0].attn_hc.input_mix_weight_up.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) &&
            (fh0_fast->layers[0].attn_hc.block_inject_weight.dtype == STRATA_QWEN_FLASH_DTYPE_F16) &&
            (fh0_fast->layers[0].shared_gate_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) &&
            (fh0_fast->layers[0].shared_down_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) &&
            (fh0_fast->layers[1].ple.key_proj.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) &&
            (fh1_fast->final_mixer.input_mix_weight_down.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) &&
            (fh1_fast->mtp.fc_embedding.dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) &&
            (sh_qm.rows_logical == 640U && sh_qm.cols == 2560U && sh_qm.group_size == 64U) &&
            (fh0_fast->active_bytes_per_tok + 900000000ULL < fh0_fp16->active_bytes_per_tok) &&
            (fh0_fast->mtp_active_bytes_per_tok + 30000000ULL < fh0_fp16->mtp_active_bytes_per_tok) &&
            ((fh0_fast->total_file_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL) &&
            ((fh1_fast->total_file_bytes % STRATA_QWEN_FLASH_PAGE_BYTES) == 0ULL) &&
            (strata_qwen_flash_repack_bundle("/tmp/a", "/tmp/a", 1U, nullptr) != 0);

        /* Null-safety checks for new 5-rec engine setters */
        strata_qwen_flash_gpu_set_int8_prefill(nullptr, true);
        strata_qwen_flash_gpu_set_adaptive_k(nullptr, false);
        strata_qwen_flash_gpu_set_frspec(nullptr, false);
        strata_glm53_gpu_set_int8_prefill(nullptr, true);
        strata_glm53_gpu_set_adaptive_k(nullptr, false);
        strata_glm53_gpu_set_frspec(nullptr, false);
        strata_inkling2_gpu_set_int8_prefill(nullptr, true);
        strata_inkling2_gpu_set_bitplane_head(nullptr, true);
        strata_inkling2_gpu_set_beam_tree(nullptr, true);
        strata_inkling2_vision_destroy(nullptr);
        const bool ik2_v0219_null_ok =
            !strata_inkling2_gpu_get_bitplane_head(nullptr) &&
            !strata_inkling2_gpu_get_beam_tree(nullptr) &&
            (strata_inkling2_vision_resolve_path(nullptr, nullptr, 0U) == 0) &&
            (STRATA_INKLING2_MAX_BATCH_SIZE == 64U) &&
            (strata_inkling2_slot_seq_pos(nullptr) == 0U) &&
            !strata_inkling2_slot_mtp_synced(nullptr) &&
            (strata_inkling2_slot_memory_bytes(nullptr) == 0U) &&
            (strata_inkling2_gpu_decode_batch(nullptr, nullptr, 0U) != 0);

        const bool compact_abi_ok =
            (STRATA_QWEN_FLASH_KV_LOC_UMA_BIT == 0x80000000U) &&
            (STRATA_QWEN_FLASH_KV_LOC_IDX_MASK == 0x7FFFFFFFU) &&
            (STRATA_QWEN_FLASH_KV_BLOCK_TOKENS == 4U) &&
            (STRATA_QWEN_FLASH_KV_BLOCK_BYTES == 8192U) &&
            (STRATA_COMPACT_OFF == 0) &&
            (STRATA_COMPACT_SHADOW == 1) &&
            (STRATA_COMPACT_PAGED == 2) &&
            (STRATA_COMPACT_HYBRID == 3) &&
            (strata_qwen_flash_slot_create(nullptr, 4096U, nullptr) == nullptr) &&
            (strata_qwen_flash_bound_slot(nullptr) == nullptr) &&
            (strata_qwen_flash_slot_seq_pos(nullptr) == 0U) &&
            (!strata_qwen_flash_slot_is_paged(nullptr)) &&
            (strata_qwen_flash_slot_bytes(nullptr) == 0ULL) &&
            (strata_qwen_flash_ane_probe_ready(nullptr, ane_note, sizeof(ane_note)) == 0) &&
            (strata_qwen_flash_gpu_prefill_slice(nullptr, nullptr, 32U, false, nullptr) != 0) &&
            (!strata_qwen_flash_gpu_get_spec_tree(nullptr)) &&
            (!strata_qwen_flash_gpu_get_mtp_kv_sync(nullptr)) &&
            phase_c_layout_ok &&
            (cstats.nvme_blocks_paged_out == 0ULL) &&
            (cstats.rcu_swaps_completed == 0U);

        const bool loop_sanity_ok = run_shingle_loop_sanity_unit_test();

        /* Security parser regression gate:
         *  1. Tokenizer uint32 wraparound (`id = 4294967295`) rejected across all 4 engines
         *  2. Vision sidecar `num_channels != 3` / OOB slice rejected on load
         *  3. CPU rANS decoder OOB `dir_offset` / `tile_u8_offsets` rejected cleanly
         *  4. Path-traversal shard names in `model.safetensors.index.json` rejected */
        bool sec_parser_ok = true;
        {
            const std::string sec_dir = bench_root + "/cli_sec_tmp";
            (void)::mkdir(sec_dir.c_str(), 0755);

            /* 1. Crafted tokenizer with id=4294967295 (UINT32_MAX -> vocab_size wraps to 0 if unguarded) */
            const std::string bad_tok = sec_dir + "/bad_tokenizer.json";
            {
                std::ofstream ofs(bad_tok, std::ios::binary);
                ofs << "{\"added_tokens\":[],\"model\":{\"vocab\":{\"a\":0,\"evil\":4294967295},\"merges\":[]}}";
            }
            if (strata_glm53_tokenizer_load(bad_tok.c_str()) != nullptr ||
                strata_qwen_flash_tokenizer_load(bad_tok.c_str()) != nullptr ||
                strata_qwen38_tokenizer_load(bad_tok.c_str()) != nullptr ||
                strata_inkling2_tokenizer_load(bad_tok.c_str()) != nullptr) {
                std::fprintf(stderr, "  [FAIL] security_gate: tokenizer accepted UINT32_MAX vocab id!\n");
                sec_parser_ok = false;
            }

            /* 2. Crafted Vision sidecar with num_channels=1 (would overflow 3-channel RGB packer) */
            const std::string bad_vis = sec_dir + "/bad_vision.strata";
            {
                std::vector<uint8_t> raw(STRATA_INKLING2_VISION_HEADER_PADDED_BYTES + 16384U, 0U);
                StrataInkling2VisionHeader vh{};
                vh.magic                    = STRATA_INKLING2_VISION_MAGIC_U64;
                vh.format_version           = STRATA_INKLING2_VISION_FORMAT_VERSION;
                vh.header_bytes             = sizeof(vh);
                vh.total_file_bytes         = raw.size();
                vh.cfg.hidden_size          = 64U;
                vh.cfg.num_hidden_layers    = 1U;
                vh.cfg.num_attention_heads  = 1U;
                vh.cfg.head_dim             = 64U;
                vh.cfg.intermediate_size    = 64U;
                vh.cfg.native_patch_size    = 20U;
                vh.cfg.adapter_merge_size   = 2U;
                vh.cfg.posemb_grid_size     = 4U;
                vh.cfg.projection_hidden_dim = 64U;
                vh.cfg.text_hidden_size     = 64U;
                vh.cfg.num_channels         = 1U; /* Malicious: != 3 */
                std::memcpy(raw.data(), &vh, sizeof(vh));
                std::ofstream ofs(bad_vis, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            }
            if (StrataInkling2VisionEngine* ve = strata_inkling2_vision_create(nil, bad_vis.c_str())) {
                strata_inkling2_vision_destroy(ve);
                std::fprintf(stderr, "  [FAIL] security_gate: vision loader accepted num_channels=1 sidecar!\n");
                sec_parser_ok = false;
            }

            /* 3. Crafted CPU rANS payload with out-of-bounds dir_offset */
            {
                std::vector<uint8_t> bad_rans(sizeof(StrataTileRansHeader) + 2048U + 16U, 0U);
                StrataTileRansHeader rh{};
                rh.magic              = 0x534E4152U;
                rh.freq_bits          = 10U;
                rh.uncompressed_count = 64U;
                rh.compressed_bytes   = static_cast<uint32_t>(bad_rans.size() - sizeof(rh));
                rh.reserved[0]        = 64U;    /* tile_syms */
                rh.reserved[1]        = 1U;     /* num_tiles */
                rh.reserved[2]        = static_cast<uint32_t>(sizeof(rh) + 2048U);
                rh.reserved[3]        = 0x7FFFFFF0U; /* OOB dir_offset */
                std::memcpy(bad_rans.data(), &rh, sizeof(rh));
                uint8_t out_syms[64] = {};
                if (strata_codec_rans_decode_32way_cpu_ref(bad_rans.data(), bad_rans.size(), out_syms, 64U) == 0) {
                    std::fprintf(stderr, "  [FAIL] security_gate: CPU rANS decoder accepted OOB dir_offset!\n");
                    sec_parser_ok = false;
                }
            }

            /* 4. Crafted model.safetensors.index.json with path traversal shard name */
            const std::string trav_dir = sec_dir + "/trav_hf";
            const std::string trav_out = sec_dir + "/trav_out";
            (void)::mkdir(trav_dir.c_str(), 0755);
            {
                std::ofstream cf(trav_dir + "/config.json");
                cf << "{\"hidden_size\":256,\"num_hidden_layers\":2,\"intermediate_size\":512,"
                      "\"vocab_size\":512,\"num_attention_heads\":4,\"num_key_value_heads\":2,"
                      "\"head_dim\":64,\"quantization\":{\"group_size\":64,\"bits\":4}}";
                std::ofstream ix(trav_dir + "/model.safetensors.index.json");
                ix << "{\"weight_map\":{\"model.embed_tokens.weight\":\"../../../../etc/passwd\"}}";
            }
            StrataQwen38ConvertStats q38_st{};
            if (strata_qwen38_convert_hf_directory(trav_dir.c_str(), trav_out.c_str(), &q38_st) == 0) {
                std::fprintf(stderr, "  [FAIL] security_gate: qwen38 converter accepted traversal shard!\n");
                sec_parser_ok = false;
            }
            /* Also reject curl globbing chars (`[`, `]`, `{`, `}`) in shard names */
            {
                std::ofstream ix(trav_dir + "/model.safetensors.index.json");
                ix << "{\"weight_map\":{\"model.embed_tokens.weight\":\"model-[1-9].safetensors\"}}";
            }
            if (strata_qwen38_convert_hf_directory(trav_dir.c_str(), trav_out.c_str(), &q38_st) == 0) {
                std::fprintf(stderr, "  [FAIL] security_gate: qwen38 converter accepted globbing shard name!\n");
                sec_parser_ok = false;
            }

            /* 5. Crafted GLM-5.3 spine with valid offset inside file but undersized lm_head.scales_bytes */
            const std::string bad_glm_dir = sec_dir + "/bad_glm53";
            (void)::mkdir(bad_glm_dir.c_str(), 0755);
            {
                StrataGlm53Config gcfg{};
                strata_glm53_default_config(&gcfg);
                gcfg.num_hidden_layers       = 2U;
                gcfg.num_nextn_predict_layers = 0U;
                gcfg.first_k_dense_replace   = 2U;
                gcfg.hidden_size             = 256U;
                gcfg.vocab_size              = 512U;
                gcfg.intermediate_size       = 256U;
                gcfg.linear_num_heads        = 4U;
                gcfg.linear_head_dim         = 64U;
                gcfg.num_attention_heads     = 4U;
                gcfg.num_key_value_heads     = 4U;
                gcfg.q_lora_rank             = 128U;
                gcfg.kv_lora_rank            = 64U;
                gcfg.qk_nope_head_dim        = 64U;
                gcfg.v_head_dim              = 64U;
                gcfg.index_n_heads           = 4U;
                gcfg.index_head_dim          = 64U;
                gcfg.n_routed_experts        = 8U;
                gcfg.num_experts_per_tok     = 2U;
                gcfg.moe_intermediate_size   = 128U;
                gcfg.layer_attn_types[0]     = STRATA_GLM53_ATTN_KDA_LINEAR;
                gcfg.layer_attn_types[1]     = STRATA_GLM53_ATTN_DSA_MLA;
                gcfg.layer_mlp_types[0]      = STRATA_GLM53_MLP_DENSE;
                gcfg.layer_mlp_types[1]      = STRATA_GLM53_MLP_DENSE;
                auto sh = std::make_unique<StrataGlm53SpineHeader>();
                strata_glm53_init_layout(&gcfg, sh.get());
                /* Tamper: shrink lm_head.scales_bytes below rows_padded * (cols / 64) * 2 */
                sh->lm_head.scales_bytes = 64ULL;
                std::vector<uint8_t> raw(static_cast<size_t>(sh->total_spine_bytes), 0U);
                std::memcpy(raw.data(), sh.get(), sizeof(*sh));
                std::ofstream sf(bad_glm_dir + "/" + STRATA_GLM53_SPINE_FILENAME, std::ios::binary);
                sf.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            }
            StrataGlm53Bundle bad_gbdl{};
            if (strata_glm53_load_bundle(bad_glm_dir.c_str(), &bad_gbdl) == 0) {
                strata_glm53_free_bundle(&bad_gbdl);
                std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted undersized lm_head.scales_bytes!\n");
                sec_parser_ok = false;
            }
            /* Also reject quant_group_size=128 (engine hard-codes 64) */
            const std::string bad_gs_dir = sec_dir + "/bad_glm53_gs128";
            (void)::mkdir(bad_gs_dir.c_str(), 0755);
            {
                StrataGlm53Config gcfg{};
                strata_glm53_default_config(&gcfg);
                gcfg.num_hidden_layers        = 2U;
                gcfg.num_nextn_predict_layers = 0U;
                gcfg.first_k_dense_replace    = 2U;
                gcfg.hidden_size              = 256U;
                gcfg.vocab_size               = 512U;
                gcfg.intermediate_size        = 256U;
                gcfg.quant_group_size         = 128U; /* Malicious: != 64 */
                gcfg.linear_num_heads         = 4U;
                gcfg.linear_head_dim          = 64U;
                gcfg.num_attention_heads      = 4U;
                gcfg.num_key_value_heads      = 4U;
                gcfg.q_lora_rank              = 128U;
                gcfg.kv_lora_rank             = 64U;
                gcfg.qk_nope_head_dim         = 64U;
                gcfg.v_head_dim               = 64U;
                gcfg.index_n_heads            = 4U;
                gcfg.index_head_dim           = 64U;
                gcfg.n_routed_experts         = 8U;
                gcfg.num_experts_per_tok      = 2U;
                gcfg.moe_intermediate_size    = 128U;
                gcfg.layer_attn_types[0]      = STRATA_GLM53_ATTN_KDA_LINEAR;
                gcfg.layer_attn_types[1]      = STRATA_GLM53_ATTN_DSA_MLA;
                gcfg.layer_mlp_types[0]       = STRATA_GLM53_MLP_DENSE;
                gcfg.layer_mlp_types[1]       = STRATA_GLM53_MLP_DENSE;
                auto sh = std::make_unique<StrataGlm53SpineHeader>();
                strata_glm53_init_layout(&gcfg, sh.get());
                std::vector<uint8_t> raw(static_cast<size_t>(sh->total_spine_bytes), 0U);
                std::memcpy(raw.data(), sh.get(), sizeof(*sh));
                std::ofstream sf(bad_gs_dir + "/" + STRATA_GLM53_SPINE_FILENAME, std::ios::binary);
                sf.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            }
            if (strata_glm53_load_bundle(bad_gs_dir.c_str(), &bad_gbdl) == 0) {
                strata_glm53_free_bundle(&bad_gbdl);
                std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted quant_group_size=128!\n");
                sec_parser_ok = false;
            }

            /* 6. Review 5 #8: Vision sidecar head_dim=128 (kernels hard-code 64) */
            const std::string bad_vis_hd = sec_dir + "/bad_vision_hd128.strata";
            {
                std::vector<uint8_t> raw(STRATA_INKLING2_VISION_HEADER_PADDED_BYTES + 16384U, 0U);
                StrataInkling2VisionHeader vh{};
                vh.magic                     = STRATA_INKLING2_VISION_MAGIC_U64;
                vh.format_version            = STRATA_INKLING2_VISION_FORMAT_VERSION;
                vh.header_bytes              = sizeof(vh);
                vh.total_file_bytes          = raw.size();
                vh.cfg.hidden_size           = 128U;
                vh.cfg.num_hidden_layers     = 1U;
                vh.cfg.num_attention_heads   = 1U;
                vh.cfg.head_dim              = 128U; /* Malicious: != 64 */
                vh.cfg.intermediate_size     = 64U;
                vh.cfg.native_patch_size     = 20U;
                vh.cfg.adapter_merge_size    = 2U;
                vh.cfg.posemb_grid_size      = 4U;
                vh.cfg.projection_hidden_dim = 64U;
                vh.cfg.text_hidden_size      = 64U;
                vh.cfg.num_channels          = 3U;
                std::memcpy(raw.data(), &vh, sizeof(vh));
                std::ofstream ofs(bad_vis_hd, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            }
            if (StrataInkling2VisionEngine* ve = strata_inkling2_vision_create(nil, bad_vis_hd.c_str())) {
                strata_inkling2_vision_destroy(ve);
                std::fprintf(stderr, "  [FAIL] security_gate: vision loader accepted head_dim=128 sidecar!\n");
                sec_parser_ok = false;
            }

            /* 7. Review 5 #2, #4, #8: GLM-5.3 n_routed_experts>512, mtp.layer.moe_layer_idx mismatch, oversized lm_head */
            {
                auto make_glm_cfg = []() {
                    StrataGlm53Config gcfg{};
                    strata_glm53_default_config(&gcfg);
                    gcfg.num_hidden_layers        = 2U;
                    gcfg.num_nextn_predict_layers = 1U;
                    gcfg.first_k_dense_replace    = 1U;
                    gcfg.hidden_size              = 256U;
                    gcfg.vocab_size               = 512U;
                    gcfg.intermediate_size        = 256U;
                    gcfg.linear_num_heads         = 4U;
                    gcfg.linear_head_dim          = 64U;
                    gcfg.num_attention_heads      = 4U;
                    gcfg.num_key_value_heads      = 4U;
                    gcfg.q_lora_rank              = 128U;
                    gcfg.kv_lora_rank             = 64U;
                    gcfg.qk_nope_head_dim         = 64U;
                    gcfg.v_head_dim               = 64U;
                    gcfg.index_n_heads            = 4U;
                    gcfg.index_head_dim           = 64U;
                    gcfg.n_routed_experts         = 8U;
                    gcfg.num_experts_per_tok      = 2U;
                    gcfg.moe_intermediate_size    = 128U;
                    gcfg.layer_attn_types[0]      = STRATA_GLM53_ATTN_KDA_LINEAR;
                    gcfg.layer_attn_types[1]      = STRATA_GLM53_ATTN_DSA_MLA;
                    gcfg.layer_mlp_types[0]       = STRATA_GLM53_MLP_DENSE;
                    gcfg.layer_mlp_types[1]       = STRATA_GLM53_MLP_SPARSE_MOE;
                    return gcfg;
                };
                auto check_bad_glm = [&](const char* subname, auto mutator) {
                    const std::string dpath = sec_dir + "/" + subname;
                    (void)::mkdir(dpath.c_str(), 0755);
                    StrataGlm53Config gcfg = make_glm_cfg();
                    auto sh = std::make_unique<StrataGlm53SpineHeader>();
                    strata_glm53_init_layout(&gcfg, sh.get());
                    mutator(*sh);
                    std::vector<uint8_t> raw(static_cast<size_t>(sh->total_spine_bytes), 0U);
                    std::memcpy(raw.data(), sh.get(), sizeof(*sh));
                    std::ofstream sf(dpath + "/" + STRATA_GLM53_SPINE_FILENAME, std::ios::binary);
                    sf.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    sf.close();
                    StrataGlm53Bundle bdl{};
                    if (strata_glm53_load_bundle(dpath.c_str(), &bdl) == 0) {
                        strata_glm53_free_bundle(&bdl);
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted %s!\n", subname);
                        sec_parser_ok = false;
                    }
                };
                check_bad_glm("bad_glm53_mtp_lidx", [](StrataGlm53SpineHeader& sh) {
                    sh.mtp.layer.moe_layer_idx = 99; /* Wrong inner layer index (#4) */
                });
                check_bad_glm("bad_glm53_lm_rows", [](StrataGlm53SpineHeader& sh) {
                    sh.lm_head.rows_logical = sh.cfg.vocab_size * 2U; /* Oversized rows (#8) */
                    sh.lm_head.rows_padded  = sh.cfg.vocab_size * 2U;
                });
                check_bad_glm("bad_glm53_exp600", [](StrataGlm53SpineHeader& sh) {
                    sh.cfg.n_routed_experts = 600U; /* Exceeds 512 array bound (#2) */
                });
            }

            /* 8. Review 5 #1 & #7: Qwen-Flash bad part_id and PLE heads_per_ngram mismatch */
            {
                StrataQwenFlashConfig fcfg{};
                strata_qwen_flash_default_config(&fcfg);
                fcfg.num_hidden_layers        = 2U;
                fcfg.split_layer_idx          = 1U;
                fcfg.num_linear_layers        = 1U;
                fcfg.num_full_attn_layers     = 1U;
                fcfg.full_attention_interval  = 2U;
                fcfg.hidden_size              = 128U;
                fcfg.vocab_size               = 256U;
                fcfg.hc_count                 = 2U;
                fcfg.hc_lowrank               = 64U;
                fcfg.num_attention_heads      = 2U;
                fcfg.num_key_value_heads      = 2U;
                fcfg.head_dim                 = 64U;
                fcfg.linear_num_key_heads     = 2U;
                fcfg.linear_key_head_dim      = 64U;
                fcfg.linear_num_value_heads   = 2U;
                fcfg.linear_value_head_dim    = 64U;
                fcfg.indexer_n_heads          = 2U;
                fcfg.indexer_kv_heads         = 1U;
                fcfg.indexer_head_dim         = 64U;
                fcfg.num_experts              = 4U;
                fcfg.num_experts_per_tok      = 2U;
                fcfg.moe_intermediate_size    = 64U;
                fcfg.shared_intermediate_size = 64U;
                fcfg.has_ple                  = 1U;
                fcfg.ple_layer_idx            = 0U;
                fcfg.ple_embed_dim            = 256U;
                fcfg.ngram_size               = 3U;
                fcfg.heads_per_ngram          = 2U;
                fcfg.ngram_vocab_size_base    = 17U;
                fcfg.has_mtp                  = 0U;

                auto check_bad_flash = [&](const char* subname, auto mutator) {
                    const std::string dpath = sec_dir + "/" + subname;
                    (void)::mkdir(dpath.c_str(), 0755);
                    auto h0  = std::make_unique<StrataQwenFlashPart0Header>();
                    auto h1  = std::make_unique<StrataQwenFlashPart1Header>();
                    auto ple = std::make_unique<StrataQwenFlashPleHeader>();
                    strata_qwen_flash_init_layout(&fcfg, h0.get(), h1.get(), ple.get());
                    mutator(*h0, *h1, *ple);
                    auto write_file_pad = [&](const std::string& p, const void* hdr_p, size_t hdr_sz, uint64_t tot) {
                        std::vector<uint8_t> raw(static_cast<size_t>(tot), 0U);
                        std::memcpy(raw.data(), hdr_p, hdr_sz);
                        std::ofstream ofs(p, std::ios::binary);
                        ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    };
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PART0_FILENAME, h0.get(), sizeof(*h0), h0->total_file_bytes);
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PART1_FILENAME, h1.get(), sizeof(*h1), h1->total_file_bytes);
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PLE_FILENAME, ple.get(), sizeof(*ple), ple->total_file_bytes);
                    StrataQwenFlashBundle fbdl{};
                    if (strata_qwen_flash_load_bundle(dpath.c_str(), 1ULL << 20, &fbdl) == 0) {
                        strata_qwen_flash_free_bundle(&fbdl);
                        std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash loader accepted %s!\n", subname);
                        sec_parser_ok = false;
                    }
                };
                check_bad_flash("bad_flash_partid", [](StrataQwenFlashPart0Header& h0, StrataQwenFlashPart1Header&, StrataQwenFlashPleHeader&) {
                    h0.layers[0].part_id = 1U; /* Wrong part_id (#7) */
                });
                check_bad_flash("bad_flash_ple_hpn", [](StrataQwenFlashPart0Header&, StrataQwenFlashPart1Header&, StrataQwenFlashPleHeader& ple) {
                    ple.heads_per_ngram = 32U; /* Overflows 32-entry stack array (#1) */
                });
                /* Positive control: unmodified synthetic Qwen-Flash bundle MUST load cleanly */
                {
                    const std::string dpath = sec_dir + "/good_flash_ctrl";
                    (void)::mkdir(dpath.c_str(), 0755);
                    auto h0  = std::make_unique<StrataQwenFlashPart0Header>();
                    auto h1  = std::make_unique<StrataQwenFlashPart1Header>();
                    auto ple = std::make_unique<StrataQwenFlashPleHeader>();
                    strata_qwen_flash_init_layout(&fcfg, h0.get(), h1.get(), ple.get());
                    auto write_file_pad = [&](const std::string& p, const void* hdr_p, size_t hdr_sz, uint64_t tot) {
                        std::vector<uint8_t> raw(static_cast<size_t>(tot), 0U);
                        std::memcpy(raw.data(), hdr_p, hdr_sz);
                        std::ofstream ofs(p, std::ios::binary);
                        ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    };
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PART0_FILENAME, h0.get(), sizeof(*h0), h0->total_file_bytes);
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PART1_FILENAME, h1.get(), sizeof(*h1), h1->total_file_bytes);
                    write_file_pad(dpath + "/" + STRATA_QWEN_FLASH_PLE_FILENAME, ple.get(), sizeof(*ple), ple->total_file_bytes);
                    StrataQwenFlashBundle fbdl{};
                    if (strata_qwen_flash_load_bundle(dpath.c_str(), 1ULL << 20, &fbdl) != 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash loader rejected valid positive control!\n");
                        sec_parser_ok = false;
                    } else {
                        strata_qwen_flash_free_bundle(&fbdl);
                    }
                }
            }

            /* 9. Round 6 (Q1, Q2, Q3): Qwen3.8 convert-then-load round trip (non-64-multiple nv -> padded 64) + exact 27B pin */
            {
                const std::string q38_hf  = sec_dir + "/q38_mini_hf";
                const std::string q38_out = sec_dir + "/q38_mini_bdl";
                (void)::mkdir(q38_hf.c_str(), 0755);
                {
                    std::ofstream cf(q38_hf + "/config.json");
                    cf << "{"
                       << "\"num_hidden_layers\":2,\"hidden_size\":64,\"intermediate_size\":64,"
                       << "\"vocab_size\":256,\"full_attention_interval\":2,"
                       << "\"num_attention_heads\":2,\"num_key_value_heads\":2,\"head_dim\":32,"
                       << "\"partial_rotary_factor\":0.25,\"attn_output_gate\":true,"
                       << "\"linear_num_key_heads\":2,\"linear_key_head_dim\":32,"
                       << "\"linear_num_value_heads\":6,\"linear_value_head_dim\":32,"
                       << "\"linear_conv_kernel_dim\":4,\"rms_norm_eps\":1e-6,"
                       << "\"tie_word_embeddings\":false,\"max_position_embeddings\":1024,"
                       << "\"bos_token_id\":1,\"eos_token_id\":[2],"
                       << "\"quantization\":{\"group_size\":64,\"bits\":4}"
                       << "}";
                    /* Build valid 256-entry GPT-2 byte-level BPE tokenizer.json */
                    std::ofstream tf(q38_hf + "/tokenizer.json", std::ios::binary);
                    tf << "{\"added_tokens\":[],\"model\":{\"vocab\":{";
                    int extra_cp = 256;
                    for (int b = 0; b < 256; ++b) {
                        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
                        const int cp = direct ? b : (extra_cp++);
                        if (b > 0) tf << ",";
                        tf << "\"";
                        if (cp == '"') tf << "\\\"";
                        else if (cp == '\\') tf << "\\\\";
                        else if (cp < 0x80) tf << static_cast<char>(cp);
                        else {
                            tf << static_cast<char>(0xC0 | ((cp >> 6) & 0x1F))
                               << static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        tf << "\":" << b;
                    }
                    tf << "},\"merges\":[]}}";
                }
                {
                    struct StSpec {
                        std::string           name;
                        std::string           dtype;
                        std::vector<uint64_t> shape;
                        uint64_t              bytes;
                    };
                    std::vector<StSpec> specs;
                    auto add_f32 = [&](const std::string& n, std::vector<uint64_t> sh) {
                        uint64_t elems = 1ULL;
                        for (uint64_t d : sh) elems *= d;
                        specs.push_back(StSpec{n, "F32", std::move(sh), elems * 4ULL});
                    };
                    auto add_q4 = [&](const std::string& pfx, uint64_t rows, uint64_t cols) {
                        specs.push_back(StSpec{pfx + ".weight", "U32", {rows, cols / 8ULL}, rows * (cols / 8ULL) * 4ULL});
                        specs.push_back(StSpec{pfx + ".scales", "F16", {rows, cols / 64ULL}, rows * (cols / 64ULL) * 2ULL});
                        specs.push_back(StSpec{pfx + ".biases", "F16", {rows, cols / 64ULL}, rows * (cols / 64ULL) * 2ULL});
                    };
                    const uint64_t D = 64ULL, DFF = 64ULL, V = 256ULL;
                    const uint64_t KDIM = 64ULL, VDIM = 192ULL, QKVDIM = 2ULL * KDIM + VDIM;
                    const uint64_t NV = 6ULL, QDIM = 64ULL, QGDIM = 128ULL, KVDIM = 64ULL;
                    add_q4("model.embed_tokens", V, D);
                    /* Layer 0: linear_attn */
                    add_f32("model.layers.0.input_layernorm.weight", {D});
                    add_f32("model.layers.0.post_attention_layernorm.weight", {D});
                    add_q4("model.layers.0.linear_attn.in_proj_qkv", QKVDIM, D);
                    add_q4("model.layers.0.linear_attn.in_proj_z",   VDIM,   D);
                    add_q4("model.layers.0.linear_attn.in_proj_a",   NV,     D);
                    add_q4("model.layers.0.linear_attn.in_proj_b",   NV,     D);
                    add_q4("model.layers.0.linear_attn.out_proj",    D,      VDIM);
                    add_f32("model.layers.0.linear_attn.conv1d.weight", {QKVDIM, 4ULL});
                    add_f32("model.layers.0.linear_attn.A_log",         {NV});
                    add_f32("model.layers.0.linear_attn.dt_bias",       {NV});
                    add_f32("model.layers.0.linear_attn.norm.weight",   {32ULL});
                    add_q4("model.layers.0.mlp.gate_proj", DFF, D);
                    add_q4("model.layers.0.mlp.up_proj",   DFF, D);
                    add_q4("model.layers.0.mlp.down_proj", D, DFF);
                    /* Layer 1: full_attn */
                    add_f32("model.layers.1.input_layernorm.weight", {D});
                    add_f32("model.layers.1.post_attention_layernorm.weight", {D});
                    add_q4("model.layers.1.self_attn.q_proj", QGDIM, D);
                    add_q4("model.layers.1.self_attn.k_proj", KVDIM, D);
                    add_q4("model.layers.1.self_attn.v_proj", KVDIM, D);
                    add_q4("model.layers.1.self_attn.o_proj", D, QDIM);
                    add_f32("model.layers.1.self_attn.q_norm.weight", {32ULL});
                    add_f32("model.layers.1.self_attn.k_norm.weight", {32ULL});
                    add_q4("model.layers.1.mlp.gate_proj", DFF, D);
                    add_q4("model.layers.1.mlp.up_proj",   DFF, D);
                    add_q4("model.layers.1.mlp.down_proj", D, DFF);
                    /* Final norm + lm_head */
                    add_f32("model.norm.weight", {D});
                    add_q4("lm_head", V, D);

                    std::string jhdr = "{";
                    uint64_t off = 0ULL;
                    for (size_t i = 0U; i < specs.size(); ++i) {
                        if (i > 0U) jhdr += ",";
                        jhdr += "\"" + specs[i].name + "\":{\"dtype\":\"" + specs[i].dtype + "\",\"shape\":[";
                        for (size_t k = 0U; k < specs[i].shape.size(); ++k) {
                            if (k > 0U) jhdr += ",";
                            jhdr += std::to_string(specs[i].shape[k]);
                        }
                        jhdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + specs[i].bytes) + "]}";
                        off += specs[i].bytes;
                    }
                    jhdr += "}";
                    while ((jhdr.size() % 8U) != 0U) jhdr.push_back(' ');
                    const uint64_t jlen = jhdr.size();
                    std::ofstream stf(q38_hf + "/model.safetensors", std::ios::binary);
                    stf.write(reinterpret_cast<const char*>(&jlen), 8);
                    stf.write(jhdr.data(), static_cast<std::streamsize>(jhdr.size()));
                    std::vector<char> zeros(static_cast<size_t>(off), 0);
                    stf.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
                }
                StrataQwen38ConvertStats q38_rt_stats{};
                StrataQwen38Bundle q38_bdl{};
                if (strata_qwen38_convert_hf_directory(q38_hf.c_str(), q38_out.c_str(), &q38_rt_stats) != 0) {
                    std::fprintf(stderr, "  [FAIL] security_gate: Qwen3.8 round-trip convert failed!\n");
                    sec_parser_ok = false;
                } else if (strata_qwen38_load_bundle(q38_out.c_str(), &q38_bdl) != 0) {
                    std::fprintf(stderr, "  [FAIL] security_gate: Qwen3.8 round-trip load failed (Q3 regression)!\n");
                    sec_parser_ok = false;
                } else {
                    if (!q38_bdl.header ||
                        q38_bdl.header->layers[0].lin_A_log.shape0 != 64U ||
                        q38_bdl.header->layers[0].lin_dt_bias.shape0 != 64U) {
                        std::fprintf(stderr, "  [FAIL] security_gate: Qwen3.8 lin_A_log/lin_dt_bias not padded to 64!\n");
                        sec_parser_ok = false;
                    }
                    /* Q1/Q2: Non-27B synthetic geometry MUST be rejected by strata_qwen38_gpu_create */
                    if (StrataQwen38GpuEngine* q38_eng = strata_qwen38_gpu_create(&q38_bdl, 64U)) {
                        strata_qwen38_gpu_destroy(q38_eng);
                        std::fprintf(stderr, "  [FAIL] security_gate: Qwen3.8 gpu_create accepted non-27B geometry (Q1)!\n");
                        sec_parser_ok = false;
                    }
                    strata_qwen38_free_bundle(&q38_bdl);

                    /* Q3 negative: unpadded shape0=6 for lin_A_log must be rejected by loader */
                    const std::string q38_bad = sec_dir + "/q38_bad_alog";
                    (void)::mkdir(q38_bad.c_str(), 0755);
                    std::ifstream ifs(q38_out + "/weights.strata", std::ios::binary);
                    std::vector<uint8_t> wraw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    if (wraw.size() >= sizeof(StrataQwen38WeightsHeader)) {
                        StrataQwen38WeightsHeader wh{};
                        std::memcpy(&wh, wraw.data(), sizeof(wh));
                        wh.layers[0].lin_A_log.shape0     = 6U;
                        wh.layers[0].lin_A_log.elem_count = 6U;
                        wh.layers[0].lin_A_log.bytes      = 24U;
                        std::memcpy(wraw.data(), &wh, sizeof(wh));
                        std::ofstream ofs(q38_bad + "/weights.strata", std::ios::binary);
                        ofs.write(reinterpret_cast<const char*>(wraw.data()), static_cast<std::streamsize>(wraw.size()));
                    }
                    StrataQwen38Bundle bad_q38{};
                    if (strata_qwen38_load_bundle(q38_bad.c_str(), &bad_q38) == 0) {
                        strata_qwen38_free_bundle(&bad_q38);
                        std::fprintf(stderr, "  [FAIL] security_gate: Qwen3.8 loader accepted unpadded lin_A_log!\n");
                        sec_parser_ok = false;
                    }
                }
            }

            /* 10. Round 6 (I1-I10): Inkling-2 loader & gpu_create negative tests + vision sidecar mismatch */
            {
                const std::string ik2_base = sec_dir + "/ik2_r6_base";
                if (!synthesize_mini_inkling2_v1_bundle(ik2_base)) {
                    std::fprintf(stderr, "  [FAIL] security_gate: failed to synthesize Inkling-2 base bundle!\n");
                    sec_parser_ok = false;
                } else {
                    std::ifstream ifs(ik2_base + "/" + STRATA_INKLING2_UMA_FILENAME, std::ios::binary);
                    const std::vector<uint8_t> base_uma((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

                    auto check_bad_ik2_loader = [&](const char* subname, auto mutator) {
                        const std::string dpath = sec_dir + "/" + subname;
                        (void)::mkdir(dpath.c_str(), 0755);
                        std::vector<uint8_t> raw = base_uma;
                        auto hdr = std::make_unique<StrataInkling2Header>();
                        std::memcpy(hdr.get(), raw.data(), sizeof(*hdr));
                        mutator(*hdr);
                        std::memcpy(raw.data(), hdr.get(), sizeof(*hdr));
                        std::ofstream ofs(dpath + "/" + STRATA_INKLING2_UMA_FILENAME, std::ios::binary);
                        ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                        ofs.close();
                        StrataInkling2Bundle bdl{};
                        bdl.uma_fd = -1;
                        if (strata_inkling2_load_bundle(dpath.c_str(), &bdl) == 0) {
                            strata_inkling2_free_bundle(&bdl);
                            std::fprintf(stderr, "  [FAIL] security_gate: inkling2 loader accepted %s!\n", subname);
                            sec_parser_ok = false;
                        }
                    };
                    auto to_valid_q8 = [](StrataInkling2TensorSlice& sl) {
                        uint64_t wb = 0ULL, sb = 0ULL;
                        strata_inkling2_q8_layout(sl.shape0, sl.shape1, &wb, &sb);
                        sl.dtype = STRATA_INKLING2_DTYPE_Q8_0;
                        sl.bytes = wb + sb;
                    };

                    /* I1: 48 logical visits all on physical layer 0 -> num_blocks reaches 8 */
                    check_bad_ik2_loader("bad_ik2_i1_48visits", [](StrataInkling2Header& h) {
                        h.cfg.num_logical_layers  = 48U;
                        h.cfg.attn_res_block_size = 4U;
                        h.cfg.loop_num_passes     = 1U;
                        h.cfg.loop_core_start     = 0U;
                        h.cfg.loop_core_end       = 0U;
                        h.cross_pass_queries      = StrataInkling2TensorSlice{};
                        h.cross_pass_kv_queries   = StrataInkling2TensorSlice{};
                        for (uint32_t i = 0U; i < h.cfg.num_phys_layers; ++i) {
                            h.layers[i].in_loop_core = 0U;
                            for (uint32_t m = 0U; m < 4U; ++m) {
                                h.layers[i].controllers[m] = StrataInkling2LoopCtrlDesc{};
                            }
                        }
                        for (uint32_t il = 0U; il < 48U; ++il) {
                            h.cfg.logical_to_phys[il]  = 0U;
                            h.cfg.logical_pass_idx[il] = 0U;
                            h.cfg.logical_is_local[il] = 1U;
                            h.cfg.logical_kv_slot[il]  = il;
                        }
                        h.cfg.num_local_visits  = 48U;
                        h.cfg.num_global_visits = 0U;
                    });

                    /* I9: embed_tokens as Q8_0 */
                    check_bad_ik2_loader("bad_ik2_i9_embed_q8", [&](StrataInkling2Header& h) {
                        to_valid_q8(h.embed_tokens);
                    });
                    /* I9: mismatched controller dtype (ctrl[1].down_weight Q8_0 while ctrl[0] F16) */
                    check_bad_ik2_loader("bad_ik2_i9_ctrl_mismatch", [&](StrataInkling2Header& h) {
                        to_valid_q8(h.layers[2].controllers[1].down_weight);
                    });
                    /* I9: mtp.markov_w1 as Q8_0 */
                    check_bad_ik2_loader("bad_ik2_i9_markov_w1_q8", [&](StrataInkling2Header& h) {
                        to_valid_q8(h.mtp.markov_w1);
                    });
                    /* I9: mtp.markov_hidden_out as Q8_0 */
                    check_bad_ik2_loader("bad_ik2_i9_markov_hout_q8", [&](StrataInkling2Header& h) {
                        to_valid_q8(h.mtp.markov_hidden_out);
                    });
                    /* I9: MTP gate_proj.dtype != up_proj.dtype */
                    check_bad_ik2_loader("bad_ik2_i9_mtp_gate_up", [&](StrataInkling2Header& h) {
                        to_valid_q8(h.mtp.layers[0].gate_proj);
                    });

                    /* I2-I8: geometry rules enforced in strata_inkling2_gpu_create */
                    StrataInkling2Bundle ik2_bdl{};
                    ik2_bdl.uma_fd = -1;
                    if (strata_inkling2_load_bundle(ik2_base.c_str(), &ik2_bdl) != 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: inkling2 loader rejected positive control!\n");
                        sec_parser_ok = false;
                    } else {
                        auto check_bad_ik2_gpu = [&](const char* label, auto mutator) {
                            auto saved = std::make_unique<StrataInkling2Header>(ik2_bdl.hdr);
                            mutator(ik2_bdl.hdr);
                            StrataInkling2GpuEngine* eng = strata_inkling2_gpu_create(&ik2_bdl, 128U);
                            ik2_bdl.hdr = *saved;
                            if (eng) {
                                strata_inkling2_gpu_destroy(eng);
                                std::fprintf(stderr, "  [FAIL] security_gate: inkling2 gpu_create accepted %s!\n", label);
                                sec_parser_ok = false;
                            }
                        };
                        check_bad_ik2_gpu("I2_swa_4096",       [](StrataInkling2Header& h) { h.cfg.sliding_window_size = 4096U; });
                        check_bad_ik2_gpu("I3_ratio_4_2",      [](StrataInkling2Header& h) { h.cfg.num_attention_heads = 4U; });
                        check_bad_ik2_gpu("I4_mtp_blk4",       [](StrataInkling2Header& h) { h.mtp.block_size = 4U; });
                        check_bad_ik2_gpu("I4_mtp_blk16",      [](StrataInkling2Header& h) { h.mtp.block_size = 16U; });
                        check_bad_ik2_gpu("I5_hidden_192",     [](StrataInkling2Header& h) { h.cfg.hidden_size = 192U; });
                        check_bad_ik2_gpu("I5_inter_192",      [](StrataInkling2Header& h) { h.cfg.intermediate_size = 192U; });
                        check_bad_ik2_gpu("I6_loop_rank_32",   [](StrataInkling2Header& h) { h.cfg.loop_gate_rank = 32U; });
                        check_bad_ik2_gpu("I7_experts_160",    [](StrataInkling2Header& h) { h.cfg.num_experts = 160U; });
                        check_bad_ik2_gpu("I8_topk_12",        [](StrataInkling2Header& h) { h.cfg.num_experts_per_tok = 12U; });
                        /* Round 7 (I11): a window larger than the 1024-entry checkpoint ring; (I12): an expert count
                         * that is not a whole number of the prefill router's 32-wide threadgroups */
                        check_bad_ik2_gpu("I11_swa_2048",      [](StrataInkling2Header& h) { h.cfg.sliding_window_size = 2048U; });
                        check_bad_ik2_gpu("I11_swa_1028",      [](StrataInkling2Header& h) { h.cfg.sliding_window_size = 1028U; });
                        check_bad_ik2_gpu("I11_mtp_swa_2048",  [](StrataInkling2Header& h) { h.mtp.sliding_window = 2048U; });
                        check_bad_ik2_gpu("I12_experts_48",    [](StrataInkling2Header& h) { h.cfg.num_experts = 48U; });
                        strata_inkling2_free_bundle(&ik2_bdl);
                    }

                    /* I10: Valid vision sidecar with text_hidden_size=128 attached to hidden_size=256 text bundle */
                    const std::string mis_vis = sec_dir + "/mismatch_vision.strata";
                    {
                        StrataInkling2VisionHeader vh{};
                        vh.magic                     = STRATA_INKLING2_VISION_MAGIC_U64;
                        vh.format_version            = STRATA_INKLING2_VISION_FORMAT_VERSION;
                        vh.header_bytes              = sizeof(vh);
                        vh.cfg.hidden_size           = 64U;
                        vh.cfg.num_hidden_layers     = 1U;
                        vh.cfg.num_attention_heads   = 1U;
                        vh.cfg.head_dim              = 64U;
                        vh.cfg.intermediate_size     = 64U;
                        vh.cfg.native_patch_size     = 4U;
                        vh.cfg.adapter_merge_size    = 2U;
                        vh.cfg.posemb_grid_size      = 4U;
                        vh.cfg.projection_hidden_dim = 64U;
                        vh.cfg.text_hidden_size      = 128U; /* Mismatch vs text hidden_size=256 */
                        vh.cfg.num_channels          = 3U;
                        uint64_t vcur = STRATA_INKLING2_VISION_HEADER_PADDED_BYTES;
                        auto vslice = [&](uint32_t dt, uint64_t nbytes) {
                            vcur = (vcur + 255ULL) & ~255ULL;
                            StrataInkling2VisionTensorSlice s{};
                            s.offset = vcur;
                            s.bytes  = nbytes;
                            s.dtype  = dt;
                            vcur += nbytes;
                            return s;
                        };
                        const uint64_t D = 64ULL, MLP = 64ULL, PH = 64ULL, TD = 128ULL;
                        const uint64_t MD = 4ULL * D, PD = 16ULL * 3ULL, GP = 16ULL;
                        vh.patch_emb_w = vslice(STRATA_INKLING2_VISION_DTYPE_F16, D * PD * 2ULL);
                        vh.class_emb   = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        vh.cls_pos_emb = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        vh.pos_emb     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, GP * D * 4ULL);
                        vh.ln_pre_w    = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        vh.ln_pre_b    = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        vh.proj_in_w   = vslice(STRATA_INKLING2_VISION_DTYPE_F16, PH * MD * 2ULL);
                        vh.proj_out_w  = vslice(STRATA_INKLING2_VISION_DTYPE_F16, TD * PH * 2ULL);
                        auto& ly = vh.layers[0];
                        ly.ln1_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.ln1_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.qkv_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F16, 3ULL * D * D * 2ULL);
                        ly.qkv_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, 3ULL * D * 4ULL);
                        ly.out_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F16, D * D * 2ULL);
                        ly.out_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.ls1_gamma = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.ln2_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.ln2_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.fc1_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F16, MLP * D * 2ULL);
                        ly.fc1_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, MLP * 4ULL);
                        ly.fc2_w     = vslice(STRATA_INKLING2_VISION_DTYPE_F16, D * MLP * 2ULL);
                        ly.fc2_b     = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        ly.ls2_gamma = vslice(STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL);
                        vcur = (vcur + 16383ULL) & ~16383ULL;
                        vh.total_file_bytes = vcur;
                        std::vector<uint8_t> vraw(static_cast<size_t>(vcur), 0U);
                        std::memcpy(vraw.data(), &vh, sizeof(vh));
                        std::ofstream vofs(mis_vis, std::ios::binary);
                        vofs.write(reinterpret_cast<const char*>(vraw.data()), static_cast<std::streamsize>(vraw.size()));
                    }
                    const std::string dummy_img = sec_dir + "/dummy.png";
                    { std::ofstream dof(dummy_img, std::ios::binary); dof << "x"; }
                    const char* run_av[] = {
                        "strata_cli", "run-inkling2",
                        "--bundle", ik2_base.c_str(),
                        "--vision", mis_vis.c_str(),
                        "--image",  dummy_img.c_str(),
                        "--max-tokens", "1"
                    };
                    if (strata_inkling2_cmd_run(10, const_cast<char**>(run_av)) == 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: run-inkling2 accepted mismatched vision text_hidden_size (I10)!\n");
                        sec_parser_ok = false;
                    }
                }
            }

            /* 11. Round 6 (G1-G9): GLM-5.3 loader & gpu_create negative tests */
            {
                auto make_r6_glm_cfg = [](bool with_dsa_and_mtp) {
                    StrataGlm53Config gcfg{};
                    strata_glm53_default_config(&gcfg);
                    gcfg.num_hidden_layers        = with_dsa_and_mtp ? 2U : 1U;
                    gcfg.num_nextn_predict_layers = with_dsa_and_mtp ? 1U : 0U;
                    gcfg.first_k_dense_replace    = 1U;
                    gcfg.hidden_size              = 256U;
                    gcfg.vocab_size               = 512U;
                    gcfg.intermediate_size        = 256U;
                    gcfg.linear_num_heads         = 4U;
                    gcfg.linear_head_dim          = 64U;
                    gcfg.num_attention_heads      = 4U;
                    gcfg.num_key_value_heads      = 4U;
                    gcfg.q_lora_rank              = 128U;
                    gcfg.kv_lora_rank             = 64U;
                    gcfg.qk_nope_head_dim         = 64U;
                    gcfg.v_head_dim               = 64U;
                    gcfg.index_n_heads            = 4U;
                    gcfg.index_head_dim           = 64U;
                    gcfg.n_routed_experts         = 8U;
                    gcfg.num_experts_per_tok      = 2U;
                    gcfg.moe_intermediate_size    = 128U;
                    gcfg.layer_attn_types[0]      = STRATA_GLM53_ATTN_KDA_LINEAR;
                    gcfg.layer_mlp_types[0]       = STRATA_GLM53_MLP_DENSE;
                    if (with_dsa_and_mtp) {
                        gcfg.layer_attn_types[1]  = STRATA_GLM53_ATTN_DSA_MLA;
                        gcfg.layer_mlp_types[1]   = STRATA_GLM53_MLP_SPARSE_MOE;
                    }
                    return gcfg;
                };
                auto write_glm_bundle = [&](const std::string& dpath, const StrataGlm53SpineHeader& sh) {
                    (void)::mkdir(dpath.c_str(), 0755);
                    std::vector<uint8_t> raw(static_cast<size_t>(sh.total_spine_bytes), 0U);
                    std::memcpy(raw.data(), &sh, sizeof(sh));
                    std::ofstream sf(dpath + "/" + STRATA_GLM53_SPINE_FILENAME, std::ios::binary);
                    sf.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    if (sh.experts_nvme_bytes > 0ULL) {
                        std::vector<uint8_t> eraw(static_cast<size_t>(sh.experts_nvme_bytes), 0U);
                        std::ofstream ef(dpath + "/" + STRATA_GLM53_EXPERTS_NVME_FILENAME, std::ios::binary);
                        ef.write(reinterpret_cast<const char*>(eraw.data()), static_cast<std::streamsize>(eraw.size()));
                    }
                };

                /* Positive control: unmodified synthetic GLM-5.3 v1 bundle loads cleanly */
                const std::string good_glm_dir = sec_dir + "/good_glm53_r6";
                {
                    StrataGlm53Config gcfg = make_r6_glm_cfg(true);
                    auto sh = std::make_unique<StrataGlm53SpineHeader>();
                    strata_glm53_init_layout(&gcfg, sh.get());
                    write_glm_bundle(good_glm_dir, *sh);
                }
                StrataGlm53Bundle glm_bdl{};
                if (strata_glm53_load_bundle(good_glm_dir.c_str(), &glm_bdl) != 0) {
                    std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader rejected valid positive control!\n");
                    sec_parser_ok = false;
                } else {
                    /* G2-G7, G8, G9: geometry rules enforced in strata_glm53_gpu_create */
                    auto check_bad_glm_gpu = [&](const char* label, auto mutator) {
                        const StrataGlm53Config saved_cfg = glm_bdl.cfg;
                        mutator(glm_bdl.cfg);
                        StrataGlm53GpuEngine* eng = strata_glm53_gpu_create(
                            &glm_bdl, 128U, 0.05f, STRATA_GLM53_MODE_ADAPTIVE);
                        glm_bdl.cfg = saved_cfg;
                        if (eng) {
                            strata_glm53_gpu_destroy(eng);
                            std::fprintf(stderr, "  [FAIL] security_gate: glm53 gpu_create accepted %s!\n", label);
                            sec_parser_ok = false;
                        }
                    };
                    check_bad_glm_gpu("G2_hc_mult_2",      [](StrataGlm53Config& c) { c.hc_mult = 2U; });
                    check_bad_glm_gpu("G2_hc_mult_8",      [](StrataGlm53Config& c) { c.hc_mult = 8U; });
                    check_bad_glm_gpu("G3_shared_exp_4",   [](StrataGlm53Config& c) { c.n_shared_experts = 4U; });
                    check_bad_glm_gpu("G4_kvlora_544",     [](StrataGlm53Config& c) { c.kv_lora_rank = 544U; });
                    check_bad_glm_gpu("G4_kvlora_48",      [](StrataGlm53Config& c) { c.kv_lora_rank = 48U; });
                    check_bad_glm_gpu("G5_lin_hd_256",     [](StrataGlm53Config& c) { c.linear_head_dim = 256U; });
                    check_bad_glm_gpu("G6_idx_hd_256",     [](StrataGlm53Config& c) { c.index_head_dim = 256U; });
                    check_bad_glm_gpu("G7_conv_k_3",       [](StrataGlm53Config& c) { c.linear_conv_kernel_dim = 3U; });
                    check_bad_glm_gpu("G8_kpool_8_gpu",    [](StrataGlm53Config& c) { c.index_kpool = 8U; });
                    check_bad_glm_gpu("G9_topk_4096",      [](StrataGlm53Config& c) { c.index_topk = 4096U; });
                    check_bad_glm_gpu("G9_topk_3",         [](StrataGlm53Config& c) { c.index_topk = 3U; });
                    /* Round 7 (G10): paged-KV probe snapshot rows must be whole 16-byte copy units */
                    check_bad_glm_gpu("G10_idx_heads_6",   [](StrataGlm53Config& c) { c.index_n_heads = 6U; });
                    check_bad_glm_gpu("G10_qlora_100",     [](StrataGlm53Config& c) { c.q_lora_rank = 100U; });
                    strata_glm53_free_bundle(&glm_bdl);
                }

                /* Round 7 loader rules: every rejected header differs from an ACCEPTED one in a single field */
                {
                    auto glm_hdr_loads = [&](const char* subname, auto mutator) -> bool {
                        const std::string dpath = sec_dir + "/" + subname;
                        StrataGlm53Config gcfg = make_r6_glm_cfg(true);
                        auto sh = std::make_unique<StrataGlm53SpineHeader>();
                        strata_glm53_init_layout(&gcfg, sh.get());
                        mutator(*sh);
                        write_glm_bundle(dpath, *sh);
                        StrataGlm53Bundle b{};
                        if (strata_glm53_load_bundle(dpath.c_str(), &b) != 0) return false;
                        strata_glm53_free_bundle(&b);
                        return true;
                    };
                    /* `hc_sinkhorn_iters` is a per-thread GPU loop count: 64 is the largest accepted value */
                    if (!glm_hdr_loads("glm53_r7_sinkhorn_64", [](StrataGlm53SpineHeader& sh) { sh.cfg.hc_sinkhorn_iters = 64U; })) {
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader rejected hc_sinkhorn_iters=64 (boundary control)!\n");
                        sec_parser_ok = false;
                    }
                    if (glm_hdr_loads("bad_glm53_r7_sinkhorn_65", [](StrataGlm53SpineHeader& sh) { sh.cfg.hc_sinkhorn_iters = 65U; })) {
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted hc_sinkhorn_iters=65!\n");
                        sec_parser_ok = false;
                    }
                    if (glm_hdr_loads("bad_glm53_r7_sinkhorn_max", [](StrataGlm53SpineHeader& sh) { sh.cfg.hc_sinkhorn_iters = 0xFFFFFFFFU; })) {
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted hc_sinkhorn_iters=UINT32_MAX!\n");
                        sec_parser_ok = false;
                    }
                    /* Routed experts with a zero top-k (the positive control above loads with top-k 2) */
                    if (glm_hdr_loads("bad_glm53_r7_topk_zero", [](StrataGlm53SpineHeader& sh) { sh.cfg.num_experts_per_tok = 0U; })) {
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted num_experts_per_tok=0 with routed experts!\n");
                        sec_parser_ok = false;
                    }
                }

                /* G1 (loader): mtp.layer.has_mhc = 1 with well-formed mHC descriptors copied from layer 0 */
                {
                    const std::string dpath = sec_dir + "/bad_glm53_g1_mtp_mhc";
                    StrataGlm53Config gcfg = make_r6_glm_cfg(true);
                    auto sh = std::make_unique<StrataGlm53SpineHeader>();
                    strata_glm53_init_layout(&gcfg, sh.get());
                    sh->mtp.layer.has_mhc = 1U;
                    sh->mtp.layer.hc_attn = sh->layers[0].hc_attn;
                    sh->mtp.layer.hc_ffn  = sh->layers[0].hc_ffn;
                    write_glm_bundle(dpath, *sh);
                    StrataGlm53Bundle b{};
                    if (strata_glm53_load_bundle(dpath.c_str(), &b) == 0) {
                        strata_glm53_free_bundle(&b);
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted mtp.layer.has_mhc=1 (G1)!\n");
                        sec_parser_ok = false;
                    }
                }
                /* G8 (loader): index_kpool=8 on KDA-only bundle (no DSA layer, no MTP) */
                {
                    const std::string dpath = sec_dir + "/bad_glm53_g8_kpool8";
                    StrataGlm53Config gcfg = make_r6_glm_cfg(false);
                    gcfg.index_kpool = 8U;
                    auto sh = std::make_unique<StrataGlm53SpineHeader>();
                    strata_glm53_init_layout(&gcfg, sh.get());
                    write_glm_bundle(dpath, *sh);
                    StrataGlm53Bundle b{};
                    if (strata_glm53_load_bundle(dpath.c_str(), &b) == 0) {
                        strata_glm53_free_bundle(&b);
                        std::fprintf(stderr, "  [FAIL] security_gate: glm53 loader accepted index_kpool=8 on KDA-only bundle (G8)!\n");
                        sec_parser_ok = false;
                    }
                }
            }

            /* 12. Round 6 (F1-F9): Qwen-Flash loader & gpu_create negative tests */
            {
                auto make_r6_flash_cfg = [](bool with_full_attn) {
                    StrataQwenFlashConfig fcfg{};
                    strata_qwen_flash_default_config(&fcfg);
                    fcfg.num_hidden_layers        = 2U;
                    fcfg.split_layer_idx          = 1U;
                    fcfg.num_linear_layers        = with_full_attn ? 1U : 2U;
                    fcfg.num_full_attn_layers     = with_full_attn ? 1U : 0U;
                    fcfg.full_attention_interval  = with_full_attn ? 2U : 8U;
                    fcfg.hidden_size              = 128U;
                    fcfg.vocab_size               = 256U;
                    fcfg.hc_count                 = 2U;
                    fcfg.hc_lowrank               = 64U;
                    fcfg.num_attention_heads      = 2U;
                    fcfg.num_key_value_heads      = 2U;
                    fcfg.head_dim                 = 64U;
                    fcfg.rotary_dim               = 32U;
                    fcfg.linear_num_key_heads     = 2U;
                    fcfg.linear_key_head_dim      = 64U;
                    fcfg.linear_num_value_heads   = 2U;
                    fcfg.linear_value_head_dim    = 64U;
                    fcfg.indexer_n_heads          = 2U;
                    fcfg.indexer_kv_heads         = 1U;
                    fcfg.indexer_head_dim         = 64U;
                    fcfg.num_experts              = 4U;
                    fcfg.num_experts_per_tok      = 2U;
                    fcfg.moe_intermediate_size    = 64U;
                    fcfg.shared_intermediate_size = 64U;
                    fcfg.has_ple                  = 1U;
                    fcfg.ple_layer_idx            = 0U;
                    fcfg.ple_embed_dim            = 256U;
                    fcfg.ngram_size               = 3U;
                    fcfg.heads_per_ngram          = 2U;
                    fcfg.ngram_vocab_size_base    = 17U;
                    fcfg.has_mtp                  = 0U;
                    return fcfg;
                };
                auto write_flash_bundle = [&](const std::string& dpath, const StrataQwenFlashPart0Header& h0,
                                              const StrataQwenFlashPart1Header& h1, const StrataQwenFlashPleHeader* ple) {
                    (void)::mkdir(dpath.c_str(), 0755);
                    auto wp = [&](const std::string& p, const void* hp, size_t hsz, uint64_t tot) {
                        std::vector<uint8_t> raw(static_cast<size_t>(tot), 0U);
                        std::memcpy(raw.data(), hp, hsz);
                        std::ofstream ofs(p, std::ios::binary);
                        ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                    };
                    wp(dpath + "/" + STRATA_QWEN_FLASH_PART0_FILENAME, &h0, sizeof(h0), h0.total_file_bytes);
                    wp(dpath + "/" + STRATA_QWEN_FLASH_PART1_FILENAME, &h1, sizeof(h1), h1.total_file_bytes);
                    if (ple) wp(dpath + "/" + STRATA_QWEN_FLASH_PLE_FILENAME, ple, sizeof(*ple), ple->total_file_bytes);
                };
                auto check_bad_flash_r6 = [&](const char* subname, bool with_full, auto mutator) {
                    const std::string dpath = sec_dir + "/" + subname;
                    StrataQwenFlashConfig fcfg = make_r6_flash_cfg(with_full);
                    auto h0  = std::make_unique<StrataQwenFlashPart0Header>();
                    auto h1  = std::make_unique<StrataQwenFlashPart1Header>();
                    auto ple = std::make_unique<StrataQwenFlashPleHeader>();
                    strata_qwen_flash_init_layout(&fcfg, h0.get(), h1.get(), ple.get());
                    mutator(*h0, *h1, *ple);
                    write_flash_bundle(dpath, *h0, *h1, ple.get());
                    StrataQwenFlashBundle b{};
                    if (strata_qwen_flash_load_bundle(dpath.c_str(), 1ULL << 20, &b) == 0) {
                        strata_qwen_flash_free_bundle(&b);
                        std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash loader accepted %s!\n", subname);
                        sec_parser_ok = false;
                    }
                };

                /* F1: cfg.has_ple == 0 while layer 0 has_ple == 1 */
                check_bad_flash_r6("bad_flash_f1_ple_mismatch", true,
                    [](StrataQwenFlashPart0Header& h0, StrataQwenFlashPart1Header& h1, StrataQwenFlashPleHeader&) {
                        h0.cfg.has_ple = 0U;
                        h1.cfg.has_ple = 0U;
                    });
                /* F2: ngram_size == 5 (overflows hist[12]) */
                check_bad_flash_r6("bad_flash_f2_ngram5", true,
                    [](StrataQwenFlashPart0Header& h0, StrataQwenFlashPart1Header& h1, StrataQwenFlashPleHeader& ple) {
                        h0.cfg.ngram_size   = 5U;
                        h1.cfg.ngram_size   = 5U;
                        ple.ngram_size      = 5U;
                        ple.ngram_heads     = 4U;
                        ple.heads_per_ngram = 1U;
                    });
                /* F2: cfg.ngram_size != ple.ngram_size */
                check_bad_flash_r6("bad_flash_f2_ngram_diff", true,
                    [](StrataQwenFlashPart0Header& h0, StrataQwenFlashPart1Header& h1, StrataQwenFlashPleHeader&) {
                        h0.cfg.ngram_size = 2U;
                        h1.cfg.ngram_size = 2U;
                    });
                /* F9: indexer_compress_ratio == 8 on linear-only bundle (no full-attn layers) */
                check_bad_flash_r6("bad_flash_f9_ratio8", false,
                    [](StrataQwenFlashPart0Header& h0, StrataQwenFlashPart1Header& h1, StrataQwenFlashPleHeader&) {
                        h0.cfg.indexer_compress_ratio = 8U;
                        h1.cfg.indexer_compress_ratio = 8U;
                    });

                /* Round 7 (F10, F11) loader rules. The config is changed BEFORE the layout is built, so every rejected
                 * bundle is self-consistent and differs from an ACCEPTED one in a single config value. */
                {
                    auto flash_cfg_loads = [&](const char* subname, auto cfg_mutator) -> bool {
                        const std::string dpath = sec_dir + "/" + subname;
                        StrataQwenFlashConfig fcfg = make_r6_flash_cfg(true);
                        cfg_mutator(fcfg);
                        auto h0  = std::make_unique<StrataQwenFlashPart0Header>();
                        auto h1  = std::make_unique<StrataQwenFlashPart1Header>();
                        auto ple = std::make_unique<StrataQwenFlashPleHeader>();
                        strata_qwen_flash_init_layout(&fcfg, h0.get(), h1.get(), ple.get());
                        write_flash_bundle(dpath, *h0, *h1, ple.get());
                        StrataQwenFlashBundle b{};
                        if (strata_qwen_flash_load_bundle(dpath.c_str(), 1ULL << 20, &b) != 0) return false;
                        strata_qwen_flash_free_bundle(&b);
                        return true;
                    };
                    auto expect_flash = [&](bool want_ok, const char* subname, auto cfg_mutator) {
                        if (flash_cfg_loads(subname, cfg_mutator) != want_ok) {
                            std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash loader %s %s!\n",
                                         want_ok ? "rejected boundary control" : "accepted", subname);
                            sec_parser_ok = false;
                        }
                    };
                    /* F10: `indexer_budget` is whole KV blocks, 4 .. STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET */
                    expect_flash(true,  "flash_r7_budget_2048",      [](StrataQwenFlashConfig& c) { c.indexer_budget = 2048U; });
                    expect_flash(true,  "flash_r7_budget_4",         [](StrataQwenFlashConfig& c) { c.indexer_budget = 4U; });
                    expect_flash(false, "bad_flash_f10_budget_2052", [](StrataQwenFlashConfig& c) { c.indexer_budget = 2052U; });
                    expect_flash(false, "bad_flash_f10_budget_max",  [](StrataQwenFlashConfig& c) { c.indexer_budget = 0xFFFFFFFCU; });
                    expect_flash(false, "bad_flash_f10_budget_2046", [](StrataQwenFlashConfig& c) { c.indexer_budget = 2046U; });
                    expect_flash(false, "bad_flash_f10_budget_0",    [](StrataQwenFlashConfig& c) { c.indexer_budget = 0U; });
                    /* F11: the router kernels keep `sel_ids[16]` */
                    expect_flash(true,  "flash_r7_topk_16",          [](StrataQwenFlashConfig& c) { c.num_experts = 32U; c.num_experts_per_tok = 16U; });
                    expect_flash(false, "bad_flash_f11_topk_17",     [](StrataQwenFlashConfig& c) { c.num_experts = 32U; c.num_experts_per_tok = 17U; });
                }

                /* F3-F8: geometry rules enforced in strata_qwen_flash_gpu_create */
                const std::string flash_gpu_dir = sec_dir + "/flash_r6_gpu_base";
                {
                    StrataQwenFlashConfig fcfg = make_r6_flash_cfg(true);
                    auto h0  = std::make_unique<StrataQwenFlashPart0Header>();
                    auto h1  = std::make_unique<StrataQwenFlashPart1Header>();
                    auto ple = std::make_unique<StrataQwenFlashPleHeader>();
                    strata_qwen_flash_init_layout(&fcfg, h0.get(), h1.get(), ple.get());
                    write_flash_bundle(flash_gpu_dir, *h0, *h1, ple.get());
                }
                StrataQwenFlashBundle fbdl{};
                if (strata_qwen_flash_load_bundle(flash_gpu_dir.c_str(), 1ULL << 20, &fbdl) != 0) {
                    std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash loader rejected r6 base!\n");
                    sec_parser_ok = false;
                } else {
                    auto check_bad_flash_gpu = [&](const char* label, auto mutator) {
                        StrataQwenFlashConfig saved = fbdl.cfg;
                        /* Normalize baseline dims to valid 128-wide kernel geometry so only the mutated field fails */
                        fbdl.cfg.head_dim               = 128U;
                        fbdl.cfg.rotary_dim             = 64U;
                        fbdl.cfg.indexer_head_dim       = 128U;
                        fbdl.cfg.linear_key_head_dim    = 128U;
                        fbdl.cfg.linear_value_head_dim  = 128U;
                        fbdl.cfg.linear_conv_kernel_dim = 4U;
                        fbdl.cfg.ple_conv_kernel_size   = 4U;
                        mutator(fbdl.cfg);
                        StrataQwenFlashGpuEngine* eng = strata_qwen_flash_gpu_create(&fbdl, 128U);
                        fbdl.cfg = saved;
                        if (eng) {
                            strata_qwen_flash_gpu_destroy(eng);
                            std::fprintf(stderr, "  [FAIL] security_gate: qwen_flash gpu_create accepted %s!\n", label);
                            sec_parser_ok = false;
                        }
                    };
                    check_bad_flash_gpu("F3_rotary_128",    [](StrataQwenFlashConfig& c) { c.rotary_dim = 128U; });
                    check_bad_flash_gpu("F3_rotary_odd",    [](StrataQwenFlashConfig& c) { c.rotary_dim = 63U; });
                    check_bad_flash_gpu("F4_gqa_5_2",       [](StrataQwenFlashConfig& c) { c.num_attention_heads = 5U; c.num_key_value_heads = 2U; });
                    check_bad_flash_gpu("F5_head_dim_512",  [](StrataQwenFlashConfig& c) { c.head_dim = 512U; });
                    check_bad_flash_gpu("F6_idx_hd_64",     [](StrataQwenFlashConfig& c) { c.indexer_head_dim = 64U; });
                    check_bad_flash_gpu("F6_idx_hd_256",    [](StrataQwenFlashConfig& c) { c.indexer_head_dim = 256U; });
                    check_bad_flash_gpu("F7_lin_khd_64",    [](StrataQwenFlashConfig& c) { c.linear_key_head_dim = 64U; });
                    check_bad_flash_gpu("F7_lin_vhd_64",    [](StrataQwenFlashConfig& c) { c.linear_value_head_dim = 64U; });
                    check_bad_flash_gpu("F7_lin_v_mod_k",   [](StrataQwenFlashConfig& c) { c.linear_num_value_heads = 5U; c.linear_num_key_heads = 2U; });
                    check_bad_flash_gpu("F8_lin_conv_2",    [](StrataQwenFlashConfig& c) { c.linear_conv_kernel_dim = 2U; });
                    check_bad_flash_gpu("F8_ple_conv_2",    [](StrataQwenFlashConfig& c) { c.ple_conv_kernel_size = 2U; });
                    /* Round 7 (F10, F11): the engine re-checks the rules itself (a caller can hand it any config) */
                    check_bad_flash_gpu("F10_budget_4096",  [](StrataQwenFlashConfig& c) { c.indexer_budget = 4096U; });
                    check_bad_flash_gpu("F10_budget_2046",  [](StrataQwenFlashConfig& c) { c.indexer_budget = 2046U; });
                    check_bad_flash_gpu("F10_budget_0",     [](StrataQwenFlashConfig& c) { c.indexer_budget = 0U; });
                    check_bad_flash_gpu("F11_topk_17",      [](StrataQwenFlashConfig& c) { c.num_experts_per_tok = 17U; });
                    check_bad_flash_gpu("F11_topk_0",       [](StrataQwenFlashConfig& c) { c.num_experts_per_tok = 0U; });
                    strata_qwen_flash_free_bundle(&fbdl);
                }

                /* Pre-pub V1: convert-flash config bounds reject hostile PLE / full_attention_interval before creating outputs */
                {
                    auto check_v1_reject = [&](const char* name, const std::string& extra_fields) {
                        const std::string hf_d  = sec_dir + "/" + name + "_hf";
                        const std::string out_d = sec_dir + "/" + name + "_out";
                        (void)::mkdir(hf_d.c_str(), 0755);
                        {
                            std::ofstream cf(hf_d + "/config.json", std::ios::binary);
                            cf << "{\"model_type\":\"qwen3_next\",\"hidden_size\":64,\"vocab_size\":64,"
                               << "\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1,"
                               << "\"head_dim\":64,\"num_experts\":2,\"num_experts_per_tok\":1,"
                               << "\"moe_intermediate_size\":64,\"shared_expert_intermediate_size\":64,"
                               << extra_fields << "}\n";
                        }
                        {
                            std::ofstream tf(hf_d + "/tokenizer.json", std::ios::binary);
                            tf << "{}\n";
                        }
                        struct stat st{};
                        if (strata_qwen_flash_convert_hf(hf_d.c_str(), out_d.c_str(), 1U, 1U, nullptr) == 0 ||
                            ::stat(out_d.c_str(), &st) == 0) {
                            std::fprintf(stderr, "  [FAIL] security_gate: V1 convert-flash accepted or created output for %s!\n", name);
                            sec_parser_ok = false;
                        }
                    };
                    check_v1_reject("v1_ple_65536",
                        "\"ple_layer_ids\":[1],\"ple_embed_dim\":65536,\"ngram_size\":2,\"heads_per_ngram\":1");
                    check_v1_reject("v1_ngram_1",
                        "\"ple_layer_ids\":[1],\"ple_embed_dim\":128,\"ngram_size\":1,\"heads_per_ngram\":1");
                    check_v1_reject("v1_heads_0",
                        "\"ple_layer_ids\":[1],\"ple_embed_dim\":128,\"ngram_size\":2,\"heads_per_ngram\":0");
                    check_v1_reject("v1_parts_129",
                        "\"ple_layer_ids\":[1],\"ple_embed_dim\":128,\"ngram_size\":2,\"heads_per_ngram\":1,\"split_ngram_parts\":129");
                    check_v1_reject("v1_fai_0",
                        "\"ple_layer_ids\":[],\"full_attention_interval\":0");
                }

                /* Pre-pub V2: repack-flash rejects unvalidated/oversized layer headers before creating dst or planning layout */
                {
                    auto check_v2_reject = [&](const char* name, uint32_t split_idx, uint32_t n_layers) {
                        const std::string src_d = sec_dir + "/" + name + "_src";
                        const std::string dst_d = sec_dir + "/" + name + "_dst";
                        (void)::mkdir(src_d.c_str(), 0755);
                        auto h0 = std::make_unique<StrataQwenFlashPart0Header>();
                        auto h1 = std::make_unique<StrataQwenFlashPart1Header>();
                        std::memset(h0.get(), 0, sizeof(*h0));
                        std::memset(h1.get(), 0, sizeof(*h1));
                        StrataQwenFlashConfig cfg = make_r6_flash_cfg(true);
                        cfg.num_hidden_layers = n_layers;
                        cfg.split_layer_idx   = split_idx;
                        h0->magic               = STRATA_QWEN_FLASH_PART0_MAGIC_U64;
                        h0->version             = STRATA_QWEN_FLASH_FORMAT_VERSION;
                        h0->header_padded_bytes = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
                        h0->total_file_bytes    = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
                        h0->cfg                 = cfg;
                        h0->num_part0_layers    = split_idx;
                        h1->magic               = STRATA_QWEN_FLASH_PART1_MAGIC_U64;
                        h1->version             = STRATA_QWEN_FLASH_FORMAT_VERSION;
                        h1->header_padded_bytes = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
                        h1->total_file_bytes    = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
                        h1->cfg                 = cfg;
                        h1->first_global_layer_idx = split_idx;
                        h1->num_part1_layers       = n_layers - split_idx;
                        write_flash_bundle(src_d, *h0, *h1, nullptr);
                        struct stat st{};
                        if (strata_qwen_flash_repack_bundle(src_d.c_str(), dst_d.c_str(), 1U, nullptr) == 0 ||
                            ::stat(dst_d.c_str(), &st) == 0) {
                            std::fprintf(stderr, "  [FAIL] security_gate: V2 repack-flash accepted or created dst for %s!\n", name);
                            sec_parser_ok = false;
                        }
                    };
                    check_v2_reject("v2_p0_40", 40U, 40U);
                    check_v2_reject("v2_p1_40", 0U, 40U);

                    /* Positive control: valid synthetic bundle repacks and reloads cleanly */
                    const std::string repack_ok_dir = sec_dir + "/flash_v2_repack_ok";
                    if (strata_qwen_flash_repack_bundle(flash_gpu_dir.c_str(), repack_ok_dir.c_str(), 1U, nullptr) != 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: V2 repack-flash rejected valid positive control!\n");
                        sec_parser_ok = false;
                    } else {
                        StrataQwenFlashBundle rb{};
                        if (strata_qwen_flash_load_bundle(repack_ok_dir.c_str(), 1ULL << 20, &rb) != 0) {
                            std::fprintf(stderr, "  [FAIL] security_gate: V2 repacked bundle failed reload!\n");
                            sec_parser_ok = false;
                        } else {
                            strata_qwen_flash_free_bundle(&rb);
                        }
                    }
                }
            }

            /* 13. Round 6 (M1-M4): Mini engine loader negative tests + JSON nesting depth limits */
            {
                std::ifstream sifs(ds_bdl_dir + "/spine.bin", std::ios::binary);
                std::ifstream uifs(ds_bdl_dir + "/uma_hot.bin", std::ios::binary);
                std::ifstream cifs(ds_bdl_dir + "/cold_experts.strata", std::ios::binary);
                const std::vector<uint8_t> base_spine((std::istreambuf_iterator<char>(sifs)), std::istreambuf_iterator<char>());
                const std::vector<uint8_t> base_uma((std::istreambuf_iterator<char>(uifs)), std::istreambuf_iterator<char>());
                const std::vector<uint8_t> base_cold((std::istreambuf_iterator<char>(cifs)), std::istreambuf_iterator<char>());

                auto write_mini_copy = [&](const std::string& dpath, const std::vector<uint8_t>& sp) {
                    (void)::mkdir(dpath.c_str(), 0755);
                    std::ofstream so(dpath + "/spine.bin", std::ios::binary);
                    so.write(reinterpret_cast<const char*>(sp.data()), static_cast<std::streamsize>(sp.size()));
                    std::ofstream uo(dpath + "/uma_hot.bin", std::ios::binary);
                    uo.write(reinterpret_cast<const char*>(base_uma.data()), static_cast<std::streamsize>(base_uma.size()));
                    std::ofstream co(dpath + "/cold_experts.strata", std::ios::binary);
                    co.write(reinterpret_cast<const char*>(base_cold.data()), static_cast<std::streamsize>(base_cold.size()));
                };

                /* Positive control: copied Mini bundle loads cleanly */
                {
                    const std::string good_mini = sec_dir + "/mini_r6_good";
                    write_mini_copy(good_mini, base_spine);
                    StrataCheckpointBundle mb{};
                    if (strata_bundle_load(good_mini.c_str(), &mb) != 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: mini loader rejected positive control!\n");
                        sec_parser_ok = false;
                    } else {
                        strata_bundle_free(&mb);
                    }
                }

                constexpr size_t kCfgOff = 8U + sizeof(StrataCheckpointBundle::arch_name);
                auto patch_u32 = [](std::vector<uint8_t>& v, size_t off, uint32_t val) {
                    if (off + 4U <= v.size()) std::memcpy(v.data() + off, &val, 4U);
                };

                /* M1: rANS blob with reserved[0] != 1024 */
                {
                    std::vector<uint8_t> sp = base_spine;
                    bool patched = false;
                    for (size_t i = kCfgOff; i + sizeof(StrataTileRansHeader) <= sp.size(); i += 4U) {
                        StrataTileRansHeader rh{};
                        std::memcpy(&rh, sp.data() + i, sizeof(rh));
                        if (rh.magic == 0x534E4152U && rh.freq_bits == 10U && rh.reserved[0] == 1024U) {
                            rh.reserved[0] = 512U;
                            rh.reserved[1] *= 2U;
                            std::memcpy(sp.data() + i, &rh, sizeof(rh));
                            patched = true;
                            break;
                        }
                    }
                    const std::string dpath = sec_dir + "/bad_mini_m1_rans512";
                    write_mini_copy(dpath, sp);
                    StrataCheckpointBundle mb{};
                    if (!patched || strata_bundle_load(dpath.c_str(), &mb) == 0) {
                        if (patched) strata_bundle_free(&mb);
                        std::fprintf(stderr, "  [FAIL] security_gate: mini loader accepted non-1024 rANS tile (M1, patched=%d)!\n", patched ? 1 : 0);
                        sec_parser_ok = false;
                    }
                }
                /* M2: n_heads % n_kv_heads != 0 (3 heads / 2 kv heads) */
                {
                    std::vector<uint8_t> sp = base_spine;
                    patch_u32(sp, kCfgOff + 16U * 4U, 3U); /* n_heads */
                    patch_u32(sp, kCfgOff + 17U * 4U, 2U); /* n_kv_heads */
                    const std::string dpath = sec_dir + "/bad_mini_m2_heads";
                    write_mini_copy(dpath, sp);
                    StrataCheckpointBundle mb{};
                    if (strata_bundle_load(dpath.c_str(), &mb) == 0) {
                        strata_bundle_free(&mb);
                        std::fprintf(stderr, "  [FAIL] security_gate: mini loader accepted n_heads=3, n_kv_heads=2 (M2)!\n");
                        sec_parser_ok = false;
                    }
                }
                /* M3: ngram_heads_per_tok == 65 (> 64) */
                {
                    std::vector<uint8_t> sp = base_spine;
                    patch_u32(sp, kCfgOff + 12U * 4U, 65U); /* ngram_heads_per_tok */
                    const std::string dpath = sec_dir + "/bad_mini_m3_ngram65";
                    write_mini_copy(dpath, sp);
                    StrataCheckpointBundle mb{};
                    if (strata_bundle_load(dpath.c_str(), &mb) == 0) {
                        strata_bundle_free(&mb);
                        std::fprintf(stderr, "  [FAIL] security_gate: mini loader accepted ngram_heads_per_tok=65 (M3)!\n");
                        sec_parser_ok = false;
                    }
                }
                /* M4: deeply nested JSON rejected cleanly without stack overflow across all 5 parsers */
                {
                    const std::string deep_tok = sec_dir + "/deep_nested.json";
                    std::string deep = "{\"deep\":";
                    for (int d = 0; d < 200; ++d) deep.push_back('[');
                    deep += "0";
                    for (int d = 0; d < 200; ++d) deep.push_back(']');
                    deep += "}";
                    {
                        std::ofstream dof(deep_tok, std::ios::binary);
                        dof << deep;
                    }
                    if (strata_glm53_tokenizer_load(deep_tok.c_str()) != nullptr ||
                        strata_qwen_flash_tokenizer_load(deep_tok.c_str()) != nullptr ||
                        strata_qwen38_tokenizer_load(deep_tok.c_str()) != nullptr ||
                        strata_inkling2_tokenizer_load(deep_tok.c_str()) != nullptr) {
                        std::fprintf(stderr, "  [FAIL] security_gate: tokenizer accepted 200-deep nested JSON (M4)!\n");
                        sec_parser_ok = false;
                    }
                    const std::string deep_hf  = sec_dir + "/deep_hf";
                    const std::string deep_out = sec_dir + "/deep_out";
                    (void)::mkdir(deep_hf.c_str(), 0755);
                    {
                        std::ofstream cf(deep_hf + "/config.json", std::ios::binary);
                        cf << deep;
                    }
                    StrataQwen38ConvertStats d_q38_st{};
                    StrataConvertStats d_mini_st{};
                    if (strata_qwen38_convert_hf_directory(deep_hf.c_str(), deep_out.c_str(), &d_q38_st) == 0 ||
                        strata_convert_safetensors_to_bundle(deep_hf.c_str(), deep_out.c_str(), &d_mini_st) == 0) {
                        std::fprintf(stderr, "  [FAIL] security_gate: converter accepted 200-deep nested config.json (M4)!\n");
                        sec_parser_ok = false;
                    }
                }
            }

            remove_tree(sec_dir);
        }

        const bool chat_unit_ok = (chat_unit_fails == 0) && (server_unit_fails == 0) &&
                                  (ik2_cli_unit_fails == 0) && ik2_v0219_null_ok &&
                                  chat_help_ok && glm53_layout_ok && glm53_v2_e2e_ok &&
                                  inkling2_turbo_e2e_ok && ik2_sec_ok && compact_abi_ok && loop_sanity_ok && sec_parser_ok;
        std::printf("  Chat/Serve & Turbo E2E Gate    : %s (chat_fails=%d, serve_fails=%d, ik2_cli_fails=%d, help_rc=%d, glm53_layout=%d, glm53_v2=%d, ik2_turbo=%d, s4_abi=%d, sec=%d)\n",
                    chat_unit_ok ? "PASS" : "FAIL",
                    chat_unit_fails,
                    server_unit_fails,
                    ik2_cli_unit_fails,
                    chat_help_ok ? 0 : 1,
                    glm53_layout_ok ? 1 : 0,
                    glm53_v2_e2e_ok ? 1 : 0,
                    inkling2_turbo_e2e_ok ? 1 : 0,
                    compact_abi_ok ? 1 : 0,
                    sec_parser_ok ? 1 : 0);
        if (!chat_unit_ok) ++failures;

        cleanup_all();

        uint32_t leftover_files = 0U;
        const uint64_t leftover_bytes = count_dir_entries_and_bytes(bench_root, &leftover_files);

        struct rusage ru_final{};
        getrusage(RUSAGE_SELF, &ru_final);
        const double peak_rss_gib = static_cast<double>(ru_final.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
        const bool rss_ok  = (peak_rss_gib <= 4.0);
        const bool disk_ok = (leftover_files == 0U && leftover_bytes == 0ULL);

        std::printf("  Peak Process RSS               : %.3f GiB <= 4.000 GiB (%s)\n",
                    peak_rss_gib, rss_ok ? "PASS" : "FAIL");
        std::printf("  bench_data/ Disk Cleanup       : %u files, %llu bytes remaining (%s)\n",
                    leftover_files, static_cast<unsigned long long>(leftover_bytes),
                    disk_ok ? "PASS" : "FAIL");
        if (!rss_ok || !disk_ok) ++failures;

        std::printf("================================================================================\n");
        std::printf("  [test_cli Result] %s (%d failures)\n",
                    (failures == 0) ? "ALL CLI & TRI-ENGINE GATES PASSED" : "FAILED",
                    failures);
        std::printf("================================================================================\n");
        return (failures == 0) ? 0 : 1;
    }
}
