#ifndef STRATA_GLM53_H
#define STRATA_GLM53_H

/*
 * STRATA-M5 v2.0 — Step 3: Regime II Frontier MoE (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`)
 * `Glm5NextForConditionalGeneration` (`glm5_next` / `glm5_next_text` + Native Layer-45 `MTP`):
 * `320B` Total / `18B` Active (`5.6%`), `43` Shards = `181.69 GB` (`169.22 GiB` > `128 GiB` Physical UMA).
 *
 * Disaggregated Tri-Engine Regime II Binary Contract:
 *   - Zero-Copy `spine.strata` (`5.3611 GiB`, `16 KiB` page-aligned for `newBufferWithBytesNoCopy:` UMA wiring):
 *     `embed_tokens`, `final_norm`, `lm_head`, dense layers `0..2`, all `45L + 1L MTP` `mHC` + `34L KDA` +
 *     `12L DSA NoPE MLA` (including pre-dequantized `FP16` `W_UK [64, 512, 256]` & `W_UV [64, 256, 512]` for
 *     zero-expansion `1 KiB/tok/layer` latent `MLA` decode!) + `Glm5NextIndexer` + `shared_experts` +
 *     `FP16 [288, 4096]` router gates (shared by Metal 4 GPU & 16-Core `ANE` `MLComputePlan` Lookahead Probes)
 *     + `FP32 [288]` `e_score_correction_bias` + non-routed `MTP` (`enorm`, `hnorm`, `eh_proj`, `shared_head.norm`).
 *   - `256 KiB`-Stripe-Aligned `experts_nvme.strata` (`163.2656 GiB` = `12,384 x 13.5000 MiB` on `F_NOCACHE` NVMe):
 *     Every routed expert `(moe_layer_idx in 0..42, expert_idx in 0..287)` across the `42` main MoE layers (`3..44`)
 *     + `1` `MTP` MoE layer (`45`) stored as an exact `14,155,776 B` (`13.5000 MiB = 54 * 256 KiB = 864 * 16 KiB`,
 *     `0 B` waste!) contiguous record with `BF16` scales/biases converted to IEEE-754 `F16` at pack time.
 *   - Configurable `56.0 GiB` UMA LFU Hot-Expert Cache (`4,248 / 12,384` = `34.3%` wired `< 80.64 GiB` `MTLBuffer`),
 *     `2.0 GiB` `Ring_DMA` staging buffer, 8-FD parallel `F_NOCACHE` NVMe DMA, 16-Core `ANE` Router Lookahead,
 *     `LOSSLESS` & `ADAPTIVE` execution modes, Native `MTP` (`draft_k = 0..3`), and `O(1)` GPU Checkpoints.
 *
 * Target Hardware: Mac17,10 (Apple M5 Max: 40c Metal 4 NAX GPU, 16c ANE, ~508 GB/s LPDDR5X, 128 GiB UMA,
 *                  ~14.3 GB/s ANS Gen5 NVMe, [MTLDevice maxBufferLength] = 80.64 GiB).
 *
 * ====================================================================================================
 * 100% VERIFIED GROUND-TRUTH CHECKPOINT & MATHEMATICAL SPECIFICATION
 * (Verified directly against `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` Shards `00001..00043` binary headers &
 *  tensor bytes, `transformers` `modeling_glm5_next.py`, and `mlx-vlm` `models/glm5_next/{language,config}.py`
 *  + `models/{deepseek_v4/hyper_connection,gated_delta,mla}.py` + `speculative/drafters/glm5_next_mtp/`):
 * ====================================================================================================
 *
 * 0. Checkpoint Inventory (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`, `181,693,692,792 B = 169.215 GiB`, 43 shards):
 *    - Total indexed tensors: `114,160` across `model-00001-of-00043.safetensors .. model-00043-of-00043.safetensors`:
 *        * `model.visual.*`: `347` BF16 vision tensors — skipped cleanly at conversion.
 *        * Top-level language tensors (`7` keys):
 *            - `model.language_model.embed_tokens.{weight,scales,biases}`: Q4 affine `[154880, 4096]` (`gs = 64`)
 *            - `model.language_model.norm.weight`: `BF16 [4096]`
 *            - `lm_head.{weight,scales,biases}`: untied Q4 affine `[154880, 4096]` (`gs = 64`)
 *        * Main decoder layers `model.language_model.layers.0..44.*` (`45` layers, `111,165` keys):
 *            - Layers `0, 1, 2` (`3` layers): `mHC` + `KDA` linear attn + **Dense** `Glm5NextMLP` (`d_ff = 12288`).
 *            - Layers `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43` (`11` layers, `l % 4 == 3`):
 *              `mHC` + **`DSA` NoPE `MLA` + `Glm5NextIndexer` (`"full"`)** + **`288`-expert `top-8` MoE** (`d_moe = 2048`).
 *            - Remaining `31` layers (`l >= 4 && l % 4 != 3`):
 *              `mHC` + **`KDA` linear attn** + **`288`-expert `top-8` MoE** (`d_moe = 2048`).
 *            - Summary: `34` `KDA` layers + `11` `DSA MLA` layers = `45` main layers; `3` dense + `42` MoE (`moe_layer_idx 0..41`).
 *        * Native 1-Layer `MTP` `model.language_model.layers.45.*` (`2,641` keys = `49` non-expert + `2,592` expert keys):
 *            - **Single-stream** (`NO hc_attn_* / hc_ffn_*`!) `enorm`, `hnorm`, `eh_proj` + `DSA` NoPE `MLA` +
 *              `Glm5NextIndexer` (`"full"`) + `288`-expert `top-8` MoE (`moe_layer_idx = 42`) + `shared_head.norm` -> `lm_head`!
 *    - **ZERO TRIPLET SHARD STRADDLING (`0 / 37,695`):** Every `(weight, scales, biases)` Q4 triplet resides 100%
 *      inside the SAME `.safetensors` shard. Note: in Layers `1` & `9`, `q_proj`/`k_proj`/`v_proj` and `q/k/v_conv1d`
 *      reside in adjacent shards, and `layers.45` resides in Shards `00001..00002` due to lexicographic shard sorting!
 *      Using `strata_glm53_q4_row_subslice()` and `strata_glm53_slice_row_offset()`, `@glm53-stream` `pwrite`s every
 *      tensor directly to its deterministic byte offset in `spine.strata` and `experts_nvme.strata` in ANY shard order
 *      with **ZERO cross-shard RAM buffering**!
 *
 * 1. CRITICAL Raw `.safetensors` Dtype, Pack-Time Conversion & RMSNorm Rule (`@glm53-stream` & `@glm53-gpu`):
 *    - **DO NOT ADD `+1.0f` TO ANY NORM WEIGHT AT CONVERSION OR RUNTIME!**
 *      `Glm5NextTextRMSNorm` (`modeling_glm5_next.py:66`) & `mlx-vlm` (`language.py:752`) use standard `T5LayerNorm`
 *      (`y = (x * rsqrt(mean(x^2) + eps)) * weight`), NEVER `1.0 + weight`! Verified on Shard `00001` & `00043` bytes
 *      (`layers.0.input_layernorm ~ 0.099`, `q_a_layernorm ~ 1.15`, `model.norm ~ 1.45`, `shared_head.norm ~ 1.72`).
 *      Convert all `BF16` norm weights losslessly `BF16 -> F32` (`(uint32_t)bf16 << 16`).
 *    - **EXACT NORM EPSILONS (`1e-5f` vs `1e-6f`!):**
 *        * ALL `RMSNorm`s use `cfg.rms_norm_eps = 1e-5f` (`0.00001f`):
 *          `mHC` unweighted 16384-dim input norm, `input_layernorm [4096]`, `post_attention_layernorm [4096]`,
 *          `KDA` `o_norm [128]`, `MLA` `q_a_layernorm [1536]` & `kv_a_layernorm [512]`, `final_norm [4096]`,
 *          and `MTP` `enorm [4096]`, `hnorm [4096]`, `input_layernorm`, `post_attention_layernorm`, `shared_head.norm`.
 *        * **`self_attn.indexer.k_norm` (`[128]`) is a FULL `nn.LayerNorm(128, eps = 1e-6f)` WITH MEAN & BIAS:**
 *          `mu = mean(x)`, `var = mean((x - mu)^2)`, `y = (x - mu) * rsqrt(var + 1e-6f) * k_norm_weight + k_norm_bias`!
 *    - **Pack-Time Dtype Conversions:**
 *        * Q4 affine triplets: copy `weight` (`U32`) verbatim; convert `scales` & `biases` `BF16 -> IEEE-754 F16`.
 *          For `self_attn.indexer.weights_proj` (`[32, 4096]`), `rows_logical = 32` is zero-padded to `rows_padded = 64`.
 *        * `mlp.gate.weight` (`BF16 [288, 4096]`): convert `BF16 -> F16 [288, 4096]` (`moe_router_gate` in `spine.strata`
 *          — normal `BF16` has 7 mantissa bits vs `F16` 10 bits, so `BF16 -> F16` is lossless for router weights AND feeds
 *          both Metal 4 GPU and 16-Core `ANE` FP16 `MLComputePlan` probes directly!).
 *        * `mlp.gate.e_score_correction_bias` (`F32 [288]`): copy `F32` verbatim (`moe_e_score_bias`).
 *        * `hc_{attn,ffn}_fn` (`BF16 [24, 16384]`): convert losslessly `BF16 -> F32 [24, 16384]`; `base` (`F32 [24]`) &
 *          `scale` (`F32 [3]`) copy `F32` verbatim.
 *        * `KDA` `A_log` (`F32 [64]`) & `dt_bias` (`F32 [8192]`): copy `F32` verbatim; `{q,k,v}_conv1d.weight`
 *          (`BF16 [8192, 1, 4]`): convert `BF16 -> F32` into contiguous `[24576, 4]` causal taps `[t-3, t-2, t-1, t]`.
 *        * `MLA` `self_attn.kv_b_proj` (`Q4 [32768, 512]`, reshaped `[64, 512, 512]` where rows `0..255` = `W_K[h]` and
 *          rows `256..511` = `W_V[h]`): unpack via `sg53_unpack_kv_b_proj()` into BOTH lossless Q4 (`mla_kv_b_k`,
 *          `mla_kv_b_v` `[16384, 512]`) AND pre-dequantized `FP16` absorbed latent matrices:
 *            - `mla_w_uk` (`F16 [64 * 512, 256]`, transposed `mla_w_uk[h, c, r] = dequant(W_K[h][r, c])`!)
 *            - `mla_w_uv` (`F16 [64 * 256, 512]`, `mla_w_uv[h, r, c] = dequant(W_V[h][r, c])`!).
 *        * `indexer.index_kpool_compress_ape` (`BF16 [4, 128]`) & `index_kpool_compress_gate` (`BF16 [128, 4096]`):
 *          convert losslessly `BF16 -> F32`.
 *
 * 2. 4-Stream Manifold-Constrained Hyper-Connections (`mHC`, Layers `0..44` ONLY; `hc_mult = 4, D = 4096`):
 *    - Entry after `e = embed_tokens(tok)` (`[4096]`): replicate 4x -> `H[s, d] = e[d]` for `s in 0..3` (`[4, 4096]`).
 *    - Before `self_attn` (`hc_attn`) and before `mlp` (`hc_ffn`), given input streams `H_in [4, 4096]`:
 *        1. Flatten `H_in` to `y [16384]` in FP32 and apply **unweighted, ungrouped** RMSNorm (`eps = 1e-5f`):
 *             `z[i] = y[i] * rsqrt(mean_{j=0..16383}(y[j]^2) + 1e-5f)`                   // [16384] FP32
 *        2. Project `mix = hc_fn @ z` (`[24]` FP32):
 *             `pre[s]   = sigmoid(mix[s]     * scale[0] + base[s])     + 1e-6f`          // s = 0..3 (+hc_eps!)
 *             `post[s]  = 2.0f * sigmoid(mix[4 + s] * scale[1] + base[4 + s])`           // s = 0..3 (no +hc_eps)
 *             `V[r, c]  = mix[8 + 4*r + c] * scale[2] + base[8 + 4*r + c]`               // r, c in 0..3
 *        3. **20-Iteration Doubly-Stochastic Sinkhorn-Knopp on `4x4` `comb` (`hc_sinkhorn_iters = 20, hc_eps = 1e-6f`):**
 *             `E[r, c]  = exp(V[r, c] - max_{k=0..3} V[r, k])`
 *             `R[r, c]  = E[r, c] / (sum_{k=0..3} E[r, k] + 1e-6f) + 1e-6f`              // iter 0: softmax + 1e-6f
 *             `R[r, c]  = R[r, c] / (sum_{q=0..3} R[q, c] + 1e-6f)`                      // iter 0: col norm
 *             For `iter = 1 .. 19` (`19` additional row+col sweeps):
 *               `R[r, c] = R[r, c] / (sum_{k=0..3} R[r, k] + 1e-6f)`                     // row norm
 *               `R[r, c] = R[r, c] / (sum_{q=0..3} R[q, c] + 1e-6f)`                     // col norm
 *             `comb[r, c] = R[r, c]`
 *        4. **Collapse (`[4, 4096] -> [4096]`):**
 *             `x_col[d] = sum_{s=0..3} pre[s] * H_in[s, d]`
 *        5. **Sublayer (`norm` + `self_attn` or `mlp`):**
 *             `y_sub = sublayer(rms_norm_4096(x_col, layernorm_w, 1e-5f))`               // [4096]
 *        6. **Expand + Doubly-Stochastic Residual (`comb^T @ H_in` — NOTE TRANSPOSE `comb[r, s]`!):**
 *             `H_out[s, d] = post[s] * y_sub[d] + sum_{r=0..3} comb[r, s] * H_in[r, d]`  // s = 0..3, d = 0..4095
 *    - Final `Glm5NextTextHyperHead` after Layer `44`:
 *        `h_mean[d] = 0.25f * (H[0, d] + H[1, d] + H[2, d] + H[3, d])`                   // unweighted mean!
 *        `h_final   = rms_norm_4096(h_mean, final_norm_w, 1e-5f)`                        // [4096] -> `lm_head` & `MTP`!
 *
 * 3. 34-Layer `KDA` (`Key-Driven Linear Attention`, `64` heads x `128`, **Per-Channel Bounded Decay**):
 *    - Given normed input `x [4096]`:
 *        1. `qkv_raw = lin_qkv_proj(x)` (`[24576]`: `q [0..8191], k [8192..16383], v [16384..24575]`).
 *        2. Causal depthwise `conv1d(k=4)` + `SiLU` (`c in 0..24575`, zero-initialized 3-tap history `x_{t-3..t-1}`):
 *             `qkv[c] = silu(w[c,0]*x_{t-3}[c] + w[c,1]*x_{t-2}[c] + w[c,2]*x_{t-1}[c] + w[c,3]*qkv_raw[c])`
 *           Split `qkv` into `q [64, 128]`, `k [64, 128]`, `v [64, 128]` (`1:1` Q/K/V heads: `hk == hv = h`).
 *        3. Per-head FP32 L2 normalization (`eps = 1e-6f`, with `scale = 1 / sqrt(128.0f)` on `q` ONLY):
 *             `q_hat[h, d] = (q[h, d] * rsqrt(sum_{j=0..127} q[h, j]^2 + 1e-6f)) * (1.0f / sqrt(128.0f))`
 *             `k_hat[h, d] =  k[h, d] * rsqrt(sum_{j=0..127} k[h, j]^2 + 1e-6f)`
 *        4. Low-rank forget, beta & output gates (`fbg = lin_fbg_a_proj(x)` `[320]`):
 *             `f_a = fbg[0..127]`, `b = fbg[128..191]`, `g_a = fbg[192..319]`
 *             `a   = reshape(lin_f_b_proj(f_a), [64, 128])`                              // per-channel!
 *             `beta[h] = sigmoid(b[h])`                                                  // [64] FP32
 *             `decay[h, dk] = exp(-5.0f * sigmoid(exp(A_log[h]) * (a[h, dk] + dt_bias[h*128 + dk])))`
 *             *(CRITICAL: `gate_lower_bound = -5.0f` uses `-5.0f * sigmoid(exp(A_log) * (a + dt_bias))`, NOT softplus!)*
 *        5. Recurrent FP32 State `S[64, Dv=128, Dk=128]` (`4.00 MiB/layer`, `136.00 MiB` across 34 KDA layers;
 *           **`decay[h, dk]` multiplies the `dk` KEY axis**, NOT `dv`!):
 *             `S[h, dv, dk] <- S[h, dv, dk] * decay[h, dk]`
 *             `kv_mem[h, dv] = sum_{dk=0..127} S[h, dv, dk] * k_hat[h, dk]`
 *             `delta[h, dv]  = (v[h, dv] - kv_mem[h, dv]) * beta[h]`
 *             `S[h, dv, dk] <- S[h, dv, dk] + delta[h, dv] * k_hat[h, dk]`
 *             `out[h, dv]    = sum_{dk=0..127} S[h, dv, dk] * q_hat[h, dk]`
 *        6. Sigmoid-gated RMSNorm (`eps = 1e-5f`) + output projection:
 *             `gate = reshape(lin_g_b_proj(g_a), [64, 128])`
 *             `y[h] = rms_norm_128(out[h], lin_o_norm, 1e-5f) * sigmoid(gate[h])`
 *             `attn_out = lin_o_proj(flatten(y))`                                        // [4096]
 *
 * 4. 12-Layer `DSA` Sparse NoPE `MLA` + `Glm5NextIndexer` (`11` main layers `l % 4 == 3` + `1` MTP layer `45`):
 *    - **`qk_rope_head_dim == 0` (`mla_use_nope == true`): ZERO RoPE ANYWHERE in `GLM-5.3-Flash` text backbone!**
 *    - Given normed input `x [4096]` at global sequence position `q_pos` (`0, 1, ...`):
 *        1. Query & Compressed Latent KV (`qkv_a = mla_qkv_a_proj(x)` `[2048]`):
 *             `q_a = qkv_a[0..1535]`, `kv_a = qkv_a[1536..2047]`
 *             `q_resid = rms_norm_1536(q_a, mla_q_a_layernorm, 1e-5f)`                   // [1536]
 *             `q       = reshape(mla_q_b_proj(q_resid), [64, 256])`                      // [64, 256]
 *             `c_kv    = rms_norm_512(kv_a, mla_kv_a_layernorm, 1e-5f)`                  // [512] -> write to KV cache!
 *           *(KV cache stores ONLY `FP16 c_kv [512]` = **`1,024 B = 1 KiB/token/layer`** = `12 KiB/tok` total!)*
 *        2. **`Glm5NextIndexer` (`index_kpool = 4`, `index_topk = 2048` -> `512` pools, `always_select_tail = true`):**
 *             - Always update indexer key & gate for `q_pos`:
 *                 `raw_k = layer_norm_128(indexer_wk(x), k_norm_w, k_norm_b, 1e-6f)`     // LayerNorm w/ mean & bias!
 *                 `k_gate = indexer_compress_gate @ x`                                   // [128] FP32
 *               Store `(raw_k, k_gate)` in the 4-token tail window at `slot = q_pos & 3`. Whenever `(q_pos & 3) == 3`,
 *               compress completed pool `p = q_pos >> 2` (`tokens 4p..4p+3`) via **per-channel 4-way softmax**:
 *                 `L[j, d] = k_gate_win[j, d] + compress_ape[j, d]`                      // j = 0..3, d = 0..127
 *                 `P[j, d] = softmax_{j=0..3}(L[j, d])`                                  // softmax over j = 0..3!
 *                 `pool_key[p, d] = sum_{j=0..3} P[j, d] * raw_k_win[j, d]`              // write FP16 [128] to pool cache
 *             - Candidate selection (`num_complete_pools = (q_pos + 1) >> 2`, `tail_len = (q_pos + 1) & 3`):
 *                 * **Fast Dense Path (`num_complete_pools <= 512`, i.e., `q_pos < 2051`):**
 *                   ALL causal tokens `0 .. q_pos` are selected (`wq_b` & `weights_proj` need not even run!).
 *                 * **Sparse Pool Path (`num_complete_pools > 512`, i.e., `q_pos >= 2051`):**
 *                   `q_idx  = reshape(indexer_wq_b(q_resid), [32, 128])`
 *                   `w_head = (indexer_weights_proj(x)) * (1.0f / sqrt(32.0f))`          // [32] FP32 (can be < 0!)
 *                   For `p = 0 .. num_complete_pools - 1`:
 *                     `score[p] = sum_{h=0..31} w_head[h] * max(0.0f, dot_128(q_idx[h], pool_key[p]) * (1.0f / sqrt(128.0f)))`
 *                   Select top-`512` pools `p_0 < p_1 < ... < p_511` (sorted chronologically!), expand each `p_i` to
 *                   `{4*p_i .. 4*p_i+3}` (`2,048` tokens), and append tail `{num_complete_pools*4 .. q_pos}` (`<= 2,051` total).
 *        3. **Absorbed Latent `MLA` Attention (`scale = 1 / sqrt(256.0f) = 0.0625f`):**
 *             `q_latent[h] = mla_w_uk[h] @ q[h]`                                         // [512] per head h = 0..63
 *             Over selected causal token indices `S_sel` (`|S_sel| <= 2,051`, sharing ONE `[|S_sel|, 512]` `c_kv` slice!):
 *               `a_prob[h, t]  = softmax_{t in S_sel}( dot_512(q_latent[h], c_kv[t]) * 0.0625f )`
 *               `c_out[h]      = sum_{t in S_sel} a_prob[h, t] * c_kv[t]`                // [512] per head h = 0..63
 *             `v_out[h] = mla_w_uv[h] @ c_out[h]` (or Q4 `mla_kv_b_v[h] @ c_out[h]`)     // [256] per head h = 0..63
 *             `attn_out = mla_o_proj(flatten(v_out))`                                    // [4096]
 *
 * 5. `LimitedSwiGLU(10.0f)` (`Dense` Layers `0..2` + `43` `288`-Expert `noaux_tc` Sigmoid MoE Layers):
 *    - **CRITICAL CLAMPED ACTIVATION (`swiglu_limit = 10.0f` on ALL Dense, Shared & Routed MLPs!):**
 *        `limited_swiglu(gate, up) = silu(min(gate, 10.0f)) * clamp(up, -10.0f, +10.0f)`
 *        *(Upper-only clamp `min(gate, 10.0f)` on `gate`; symmetric `[-10.0f, +10.0f]` clamp on `up`!)*
 *    - **Dense Layers `0..2` (`d_ff = 12288`):**
 *        `gate, up = split_12288(dense_gate_up_proj(x))`; `mlp_out = dense_down_proj(limited_swiglu(gate, up))`.
 *    - **MoE Layers `3..44` & `45` (`288` routed experts, `top_k = 8`, `d_moe = 2048`, `routed_scaling_factor = 2.5f`):**
 *        1. `logits[e] = dot_4096(FP32(x), FP32(moe_router_gate[e]))`                    // e = 0..287
 *        2. `scores[e] = sigmoid(logits[e])`                                             // sigmoid, NOT softmax!
 *        3. `choice[e] = scores[e] + moe_e_score_bias[e]`                                // (`n_group=1, topk_group=1`)
 *        4. Pick `top_k = 8` indices `idx[0..7]` maximizing `choice[e]`.
 *        5. Gather **UNBIASED** `raw_w[i] = scores[idx[i]]` (WITHOUT `moe_e_score_bias`!), normalize & scale:
 *             `w[i] = (raw_w[i] / (sum_{j=0..7} raw_w[j] + 1e-20f)) * 2.5f`
 *        6. `y_routed = sum_{i=0..7} w[i] * down_proj[idx[i]](limited_swiglu(gate_proj[idx[i]](x), up_proj[idx[i]](x)))`
 *        7. Un-gated shared expert (`weight = 1.0`, NO scalar gate):
 *             `s_gate, s_up = split_2048(shared_gate_up_proj(x))`
 *             `mlp_out = y_routed + shared_down_proj(limited_swiglu(s_gate, s_up))`
 *
 * 6. Native 1-Layer `MTP` (`layers.45`, Single-Stream `DSA MLA` + `288E` MoE):
 *    - Given prompt `tok[0..L-1]`, Main Model post-`final_norm` `H_main[0..L-1]` (`[L, 4096]`), and sampled `bonus = tok[L]`:
 *        * Prefill `MTP` at `mtp_pos = 0 .. L - 1` with `shifted_tok = [tok[1..L-1], bonus]` paired with `H_main[0..L-1]`.
 *        * **CRITICAL `mtp_pos == 0` ZERO MASK (`glm5_next_mtp.py:201`):**
 *            `e_tok = (mtp_pos == 0) ? zeros(4096) : embed_tokens(shifted_tok)`
 *        * Fusion (`eh_proj` is `Q4 [4096, 8192]`):
 *            `x_mtp = mtp.eh_proj( concat_8192(rms_norm(e_tok, mtp.enorm, 1e-5f), rms_norm(h_in, mtp.hnorm, 1e-5f)) )`
 *        * Single-stream pre-norm residual (`NO mHC!`):
 *            `x_mtp = x_mtp + mtp.self_attn(rms_norm(x_mtp, mtp.input_layernorm, 1e-5f))`
 *            `x_mtp = x_mtp + mtp.mlp(rms_norm(x_mtp, mtp.post_attention_layernorm, 1e-5f))`
 *            `h_mtp_out = rms_norm(x_mtp, mtp.shared_head_norm, 1e-5f)` -> `lm_head(h_mtp_out)` -> `draft_tokens[0]`!
 *        * For `draft_k >= 2`, step `k = 1 .. draft_k - 1` feeds `(draft_tokens[k-1], h_mtp_out_{k-1})` at `mtp_pos + k`.
 *        * Main Model verifies `[pending_token_id, draft_tokens[0..draft_k-1]]` in 1 batched pass via exact Leviathan
 *          verification, rolls back rejected `KDA` + `MLA`/indexer states, and replays accepted `(emitted, H_verify)` into MTP!
 *
 * 7. Official `GLM-5.3-Flash` Tokenizer (`vocab_size = 154880`) & `chat_template.jinja` Formatting:
 *    - Special / Added Token IDs:
 *        `154820`: `<|endoftext|>` (pad & EOS #0)   `154826`: `<|system|>`        `154841`: `<think>`
 *        `154822`: `[gMASK]`       (BOS prefix #0)  `154827`: `<|user|>` (EOS #1) `154842`: `</think>`
 *        `154824`: `<sop>`         (BOS prefix #1)  `154828`: `<|assistant|>`     `154829`: `<|observation|>` (EOS #2)
 *      *(Note: in `tokenizer.json`, `154820..154837` have `special=true` and `154838..154855` incl. `<think>`/`</think>`
 *      are `added_tokens` with `special=false`. When `parse_special == true`, `strata_glm53_encode` MUST match ALL 36
 *      added tokens `154820..154855` atomically! When `skip_special == true`, `decode` skips `154820..154837`.)*
 *    - Turn 1 Chat String (`reasoning_effort` in `{"Low", "High", "Max"}`, default `"Low"` when NULL; no extra `\n`!):
 *        * `enable_thinking == false` (`--no-think`):
 *          `[gMASK]<sop><|system|>Reasoning Effort: {Effort}[<|system|>{sys}]<|user|>{user}<|assistant|><think></think>`
 *        * `enable_thinking == true`:
 *          `[gMASK]<sop><|system|>Reasoning Effort: {Effort}[<|system|>{sys}]<|user|>{user}<|assistant|><think>`
 *    - Turn 2+ Continuation String (appended directly after Turn `t-1`'s generated reply before EOS):
 *        * `enable_thinking == false`: `<|user|>{user}<|assistant|><think></think>`
 *        * `enable_thinking == true`:  `<|user|>{user}<|assistant|><think>`
 */

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint8_t  sg53_u8;
typedef uint16_t sg53_u16;
typedef uint32_t sg53_u32;
typedef uint64_t sg53_u64;
typedef int32_t  sg53_i32;
typedef int64_t  sg53_i64;
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t  sg53_u8;
typedef uint16_t sg53_u16;
typedef uint32_t sg53_u32;
typedef uint64_t sg53_u64;
typedef int32_t  sg53_i32;
typedef int64_t  sg53_i64;
#endif

/* ============================================================================
 * 1. Alignment, Magic, Expert Record Geometry & Constants (Host C++ & Metal 4)
 * ============================================================================ */

#define STRATA_GLM53_SPINE_MAGIC_U64            0x33354D47354D5453ULL /* "STM5GM53" */
#define STRATA_GLM53_FORMAT_VERSION_V1          1U                    /* Regime II NVMe Q4_A (`experts_nvme.strata`) */
#define STRATA_GLM53_FORMAT_VERSION_V2          2U                    /* 100% RAM-Resident `RHT-64 + E8S-18-V4` UMA */
#define STRATA_GLM53_FORMAT_VERSION             STRATA_GLM53_FORMAT_VERSION_V2

/* Expert quantization dtypes (`hdr->main_expert_dtype`, `hdr->mtp_expert_dtype`, `hdr->moe_layer_quant[l]`) */
#define STRATA_GLM53_EXPERT_DTYPE_Q4_A          0U /* 4-bit affine gs=64 (`14,155,776 B` = 13.500 MiB/expert) */
#define STRATA_GLM53_EXPERT_DTYPE_E8S18_V4      1U /* RHT-64 + Spherical Construction-A E8 18-bit V4 (`7,471,104 B` = 7.125 MiB/expert) */

/*
 * Exact Deterministic v2 RAM-Resident `RHT-64 + E8S-18-V4` Constants & Byte Layout
 * (Shared between Host C++20 Converter/Runtime and Metal 4.1 MSL Kernels):
 *
 *   1. `RHT-64` Orthogonal Block Transform (`y = (1/8) * H_64 * diag(sign) * x`):
 *        - `sign[i] = (splitmix64(0x5354524154414538ULL + i) & 1) ? -1.0f : +1.0f` (`i = 0..63`)
 *        - Bit `i` of `STRATA_GLM53_RHT64_SIGN_MASK` (`0x8E378CE1190887EDULL`) == 1 iff `sign[i] == -1.0f`.
 *        - Applied along `K` of `gate_proj/up_proj` (`K=4096`) and `down_proj` (`K=2048`) offline,
 *          and identically on activations `x` and `LimitedSwiGLU(gate, up)` online on GPU.
 *          CRITICAL ORDER: multiply `diag(sign)` FIRST, then 6-stage Walsh-Hadamard `H_64`, then `* 0.125f`!
 *
 *   2. Symmetry-Factored Spherical Construction-A E8 (`E8S-18-V4`):
 *        - `4,096` canonical orbits (`Half8Entry`, `64 KiB` L1 LUT) x `4` Klein `V_4` swizzles
 *          (`v4 in 0..3`: `sw = v4 & 1` swaps lo/hi, `zw = v4 & 2` applies `.zwxy`) x `16` Extended Hamming
 *          `H_8` (`[8, 4, 4]` RM(1,3)) sign masks = `258,208` distinct exact spherical E8 codewords.
 *        - Group-64 unsigned `FP8 E4M4` (`bias = 14`) scale: `0 -> 0.0h`, else `as_type<half>(ushort((uint(b) + 0x10u) << 6u))`.
 *
 *   3. Planar Per-Expert Byte Layout (`7,471,104 B = 7.125000 MiB = 456 * 16 KiB`, `0 B` waste!):
 *        - `gate_proj` (`[2048, 4096]`, `+0`,         `2,490,368 B = 2.375 MiB`)
 *        - `up_proj`   (`[2048, 4096]`, `+2,490,368`, `2,490,368 B = 2.375 MiB`)
 *        - `down_proj` (`[4096, 2048]`, `+4,980,736`, `2,490,368 B = 2.375 MiB`)
 *        - Within each `2,490,368 B` projection (`8,388,608` weights = `1,048,576` 8D vecs = `131,072` groups of 64):
 *            * `+0`         (`2,097,152 B` = `2.000 MiB`): `uint16_t lo16[R, K/8]`  = `(orbit_idx[11:0] << 4) | h_idx[3:0]`
 *            * `+2,097,152` (`  262,144 B` = `0.250 MiB`): `uint16_t hi2[R, K/64]`  = `8 x 2-bit v4_k` at bits `[2*k+1:2*k]`
 *            * `+2,359,296` (`  131,072 B` = `0.125 MiB`): `uint8_t  scale[R, K/64]` = unsigned `FP8 E4M4` (`bias=14`)
 *        - Split across TWO `16 KiB`-aligned UMA files (`<= 80.64 GiB` `MTLBuffer` ceiling):
 *            * `experts_uma_part0.strata`: `moe_l 0..20`  (`L3..L23`,  `6,048 * 7,471,104 B` = `45,185,236,992 B` = `42.082 GiB`)
 *            * `experts_uma_part1.strata`: `moe_l 21..41` (`L24..L44`, `6,048 * 7,471,104 B` = `45,185,236,992 B` = `42.082 GiB`)
 *            * `mtp_experts.strata`:       `moe_l 42`     (`L45 MTP`,  `  288 * 14,155,776 B` = `4,076,863,488 B` = ` 3.797 GiB` Q4_A)
 */
#define STRATA_GLM53_RHT64_SEED                 0x5354524154414538ULL /* "STRATAE8" */
#define STRATA_GLM53_RHT64_BLOCK                64U
#define STRATA_GLM53_RHT64_SIGN_MASK            0x8E378CE1190887EDULL /* bit i == 1 iff sign[i] == -1.0f */

#define STRATA_GLM53_E8S18_NUM_ORBITS           4096U
#define STRATA_GLM53_E8S18_LUT_BYTES            65536ULL              /* 4096 * 16 B (`Half8Entry`) */
#define STRATA_GLM53_E8S18_GROUP_SIZE           64U
#define STRATA_GLM53_E8S18_LO16_BYTES           2097152ULL            /* 2.000 MiB = [R, K/8] * 2 B */
#define STRATA_GLM53_E8S18_HI2_BYTES            262144ULL             /* 0.250 MiB = [R, K/64] * 2 B */
#define STRATA_GLM53_E8S18_SCALE_BYTES          131072ULL             /* 0.125 MiB = [R, K/64] * 1 B */
#define STRATA_GLM53_E8S18_PROJ_BYTES           2490368ULL            /* 2.375 MiB per gate/up/down */
#define STRATA_GLM53_E8S18_RECORD_BYTES         7471104ULL            /* 7.125 MiB = 456 * 16 KiB */

#define STRATA_GLM53_E8S18_PROJ_LO16_OFF        0ULL
#define STRATA_GLM53_E8S18_PROJ_HI2_OFF         2097152ULL
#define STRATA_GLM53_E8S18_PROJ_SCALE_OFF       2359296ULL

#define STRATA_GLM53_E8S18_GATE_OFF             0ULL
#define STRATA_GLM53_E8S18_UP_OFF               2490368ULL
#define STRATA_GLM53_E8S18_DOWN_OFF             4980736ULL

#define STRATA_GLM53_E8S18_PART0_LAYERS         21U                   /* Main MoE layers 0..20 (L3..L23) */
#define STRATA_GLM53_E8S18_PART1_LAYERS         21U                   /* Main MoE layers 21..41 (L24..L44) */

#ifdef __METAL_VERSION__
typedef struct StrataGlm53E8Half8Entry {
    metal::half4 lo;
    metal::half4 hi;
} StrataGlm53E8Half8Entry;

static constant uchar kSg53H8Mask[16] = {
    0x00u, 0xFFu, 0xAAu, 0x55u, 0xCCu, 0x33u, 0x66u, 0x99u,
    0xF0u, 0x0Fu, 0x5Au, 0xA5u, 0x3Cu, 0xC3u, 0x96u, 0x69u
};

static constant metal::ushort4 kSg53H8SgnLo[16] = {
    metal::ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), /* 0x00 */
    metal::ushort4(0x8000u, 0x8000u, 0x8000u, 0x8000u), /* 0xFF */
    metal::ushort4(0x0000u, 0x8000u, 0x0000u, 0x8000u), /* 0xAA */
    metal::ushort4(0x8000u, 0x0000u, 0x8000u, 0x0000u), /* 0x55 */
    metal::ushort4(0x0000u, 0x0000u, 0x8000u, 0x8000u), /* 0xCC */
    metal::ushort4(0x8000u, 0x8000u, 0x0000u, 0x0000u), /* 0x33 */
    metal::ushort4(0x0000u, 0x8000u, 0x8000u, 0x0000u), /* 0x66 */
    metal::ushort4(0x8000u, 0x0000u, 0x0000u, 0x8000u), /* 0x99 */
    metal::ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), /* 0xF0 */
    metal::ushort4(0x8000u, 0x8000u, 0x8000u, 0x8000u), /* 0x0F */
    metal::ushort4(0x0000u, 0x8000u, 0x0000u, 0x8000u), /* 0x5A */
    metal::ushort4(0x8000u, 0x0000u, 0x8000u, 0x0000u), /* 0xA5 */
    metal::ushort4(0x0000u, 0x0000u, 0x8000u, 0x8000u), /* 0x3C */
    metal::ushort4(0x8000u, 0x8000u, 0x0000u, 0x0000u), /* 0xC3 */
    metal::ushort4(0x0000u, 0x8000u, 0x8000u, 0x0000u), /* 0x96 */
    metal::ushort4(0x8000u, 0x0000u, 0x0000u, 0x8000u)  /* 0x69 */
};

static constant metal::ushort4 kSg53H8SgnHi[16] = {
    metal::ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), /* 0x00 */
    metal::ushort4(0x8000u, 0x8000u, 0x8000u, 0x8000u), /* 0xFF */
    metal::ushort4(0x0000u, 0x8000u, 0x0000u, 0x8000u), /* 0xAA */
    metal::ushort4(0x8000u, 0x0000u, 0x8000u, 0x0000u), /* 0x55 */
    metal::ushort4(0x0000u, 0x0000u, 0x8000u, 0x8000u), /* 0xCC */
    metal::ushort4(0x8000u, 0x8000u, 0x0000u, 0x0000u), /* 0x33 */
    metal::ushort4(0x0000u, 0x8000u, 0x8000u, 0x0000u), /* 0x66 */
    metal::ushort4(0x8000u, 0x0000u, 0x0000u, 0x8000u), /* 0x99 */
    metal::ushort4(0x8000u, 0x8000u, 0x8000u, 0x8000u), /* 0xF0 */
    metal::ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), /* 0x0F */
    metal::ushort4(0x8000u, 0x0000u, 0x8000u, 0x0000u), /* 0x5A */
    metal::ushort4(0x0000u, 0x8000u, 0x0000u, 0x8000u), /* 0xA5 */
    metal::ushort4(0x8000u, 0x8000u, 0x0000u, 0x0000u), /* 0x3C */
    metal::ushort4(0x0000u, 0x0000u, 0x8000u, 0x8000u), /* 0xC3 */
    metal::ushort4(0x8000u, 0x0000u, 0x0000u, 0x8000u), /* 0x96 */
    metal::ushort4(0x0000u, 0x8000u, 0x8000u, 0x0000u)  /* 0x69 */
};

inline half sg53_e4m4_decode_h(uchar b) {
    return (b == 0u) ? half(0.0h) : as_type<half>(ushort((uint(b) + 0x10u) << 6u));
}

inline void sg53_e8s18_decode_h4(
    device const StrataGlm53E8Half8Entry* lut,
    ushort lo16,
    uint   v4,
    thread metal::half4& out_lo,
    thread metal::half4& out_hi
) {
    uint oid = uint(lo16) >> 4u;
    uint h   = uint(lo16) & 15u;
    StrataGlm53E8Half8Entry e = lut[oid];
    bool sw = (v4 & 1u) != 0u;
    bool zw = (v4 & 2u) != 0u;
    metal::half4 a = sw ? e.hi : e.lo;
    metal::half4 b = sw ? e.lo : e.hi;
    metal::half4 l = zw ? a.zwxy : a;
    metal::half4 r = zw ? b.zwxy : b;
    out_lo = as_type<metal::half4>(as_type<metal::ushort4>(l) ^ kSg53H8SgnLo[h]);
    out_hi = as_type<metal::half4>(as_type<metal::ushort4>(r) ^ kSg53H8SgnHi[h]);
}
#else
typedef struct StrataGlm53E8Half8Entry {
    sg53_u16 v[8]; /* IEEE-754 FP16 (lo = v[0..3], hi = v[4..7]), sizeof = 16 B */
} StrataGlm53E8Half8Entry;
#endif

#define STRATA_GLM53_PAGE_BYTES                 16384ULL              /* 16 KiB Apple Silicon VM/UAT page */
#define STRATA_GLM53_NVME_STRIPE_BYTES          262144ULL             /* 256 KiB ANS Gen5 NVMe LBA stripe */
#define STRATA_GLM53_HEADER_PADDED_BYTES        131072U               /* 8 * 16 KiB (128 KiB >= sizeof(SpineHeader)) */
#define STRATA_GLM53_TENSOR_ALIGN               256ULL                /* 256 B Metal buffer offset alignment */

#define STRATA_GLM53_MAX_LAYERS                 48U                   /* 45 main layers (0..44) + headroom */
#define STRATA_GLM53_CANONICAL_LAYERS           45U                   /* 34 KDA + 11 DSA MLA */
#define STRATA_GLM53_CANONICAL_DENSE_LAYERS     3U                    /* Layers 0, 1, 2 */
#define STRATA_GLM53_CANONICAL_MAIN_MOE_LAYERS  42U                   /* Layers 3..44 */
#define STRATA_GLM53_CANONICAL_TOTAL_MOE_LAYERS 43U                   /* 42 main MoE + 1 MTP MoE (layer 45) */
#define STRATA_GLM53_CANONICAL_HIDDEN           4096U
#define STRATA_GLM53_CANONICAL_VOCAB            154880U
#define STRATA_GLM53_CANONICAL_DENSE_INTER      12288U
#define STRATA_GLM53_CANONICAL_MOE_INTER        2048U
#define STRATA_GLM53_CANONICAL_EXPERTS          288U
#define STRATA_GLM53_CANONICAL_TOP_K            8U
#define STRATA_GLM53_CANONICAL_TOTAL_EXPERTS    12384U                /* 43 * 288 */
#define STRATA_GLM53_MAX_SPEC_DRAFT_K           4U                    /* Supports draft_k = 0..4 (verify M <= 5) */

#define STRATA_GLM53_QUANT_GROUP_SIZE           64U
#define STRATA_GLM53_QUANT_BITS                 4U

/*
 * Exact Deterministic Routed-Expert Record Layout inside BOTH `experts_nvme.strata` (`163.265625 GiB`)
 * AND UMA `b_hot_experts` / `b_ring_dma` (`14,155,776 B = 13.5000 MiB = 54 * 256 KiB = 864 * 16 KiB`, `0 B` waste!):
 *   - `moe_layer_idx`: `0..41` for main layers `3..44` (`l - 3`), `42` for `mtp` layer `45`.
 *   - Expert `(moe_layer_idx, e)` file offset: `((uint64_t)moe_layer_idx * 288ULL + (uint64_t)e) * 14155776ULL`
 *   - `gate_proj` (`[2048, 4096]` Q4, `4,718,592 B = 4.5000 MiB = 18 * 256 KiB`):
 *       `+0`          (`4,194,304 B`): `uint32_t weight[2048, 512]`
 *       `+4,194,304`  (`  262,144 B`): `half     scales[2048, 64]`
 *       `+4,456,448`  (`  262,144 B`): `half     biases[2048, 64]`
 *   - `up_proj`   (`[2048, 4096]` Q4, `4,718,592 B = 4.5000 MiB = 18 * 256 KiB`):
 *       `+4,718,592`  (`4,194,304 B`): `uint32_t weight[2048, 512]`
 *       `+8,912,896`  (`  262,144 B`): `half     scales[2048, 64]`
 *       `+9,175,040`  (`  262,144 B`): `half     biases[2048, 64]`
 *   - `down_proj` (`[4096, 2048]` Q4, `4,718,592 B = 4.5000 MiB = 18 * 256 KiB`):
 *       `+9,437,184`  (`4,194,304 B`): `uint32_t weight[4096, 256]`
 *       `+13,631,488` (`  262,144 B`): `half     scales[4096, 32]`
 *       `+13,893,632` (`  262,144 B`): `half     biases[4096, 32]`
 */
#define STRATA_GLM53_EXPERT_PROJ_WEIGHT_BYTES   4194304ULL            /* 4.0000 MiB (16 * 256 KiB) */
#define STRATA_GLM53_EXPERT_PROJ_SCALES_BYTES   262144ULL             /* 256 KiB    ( 1 * 256 KiB) */
#define STRATA_GLM53_EXPERT_PROJ_BIASES_BYTES   262144ULL             /* 256 KiB    ( 1 * 256 KiB) */
#define STRATA_GLM53_EXPERT_PROJ_TOTAL_BYTES    4718592ULL            /* 4.5000 MiB (18 * 256 KiB) */
#define STRATA_GLM53_EXPERT_RECORD_BYTES        14155776ULL           /* 13.5000 MiB (54 * 256 KiB, 864 * 16 KiB) */

#define STRATA_GLM53_EXPERT_GATE_WEIGHT_OFF     0ULL
#define STRATA_GLM53_EXPERT_GATE_SCALES_OFF     4194304ULL
#define STRATA_GLM53_EXPERT_GATE_BIASES_OFF     4456448ULL
#define STRATA_GLM53_EXPERT_UP_WEIGHT_OFF       4718592ULL
#define STRATA_GLM53_EXPERT_UP_SCALES_OFF       8912896ULL
#define STRATA_GLM53_EXPERT_UP_BIASES_OFF       9175040ULL
#define STRATA_GLM53_EXPERT_DOWN_WEIGHT_OFF     9437184ULL
#define STRATA_GLM53_EXPERT_DOWN_SCALES_OFF     13631488ULL
#define STRATA_GLM53_EXPERT_DOWN_BIASES_OFF     13893632ULL

/* Expert buffer source tags for indirect Metal 4 MoE dispatch (`b_hot_experts` vs `b_ring_dma`) */
#define STRATA_GLM53_EXPERT_SRC_HOT_UMA         0U
#define STRATA_GLM53_EXPERT_SRC_RING_DMA        1U

/* Attention block types (`cfg.layer_types[l]`) */
#define STRATA_GLM53_ATTN_KDA_LINEAR            0U /* Key-Driven Linear Attention (`l % 4 != 3`, 34 layers) */
#define STRATA_GLM53_ATTN_DSA_MLA               1U /* DeepSeek Sparse NoPE MLA + Indexer (`l % 4 == 3`, 11L + MTP) */

/* FFN block types (`cfg.mlp_types[l]`) */
#define STRATA_GLM53_MLP_DENSE                  0U /* Dense LimitedSwiGLU (`l < 3`, d_ff = 12288) */
#define STRATA_GLM53_MLP_SPARSE_MOE             1U /* 288-Expert Top-8 + Shared Expert (`l >= 3` + MTP, d_moe = 2048) */

/* Dense tensor dtypes in `spine.strata` */
#define STRATA_GLM53_DTYPE_NONE                 0U
#define STRATA_GLM53_DTYPE_F32                  1U /* IEEE-754 float32 (`float`) */
#define STRATA_GLM53_DTYPE_F16                  2U /* IEEE-754 float16 (`uint16_t` / `half`) */
#define STRATA_GLM53_DTYPE_U32                  3U /* Packed 8x 4-bit unsigned nibbles per `uint32_t` */

/* Canonical GLM-5.3-Flash special & added token IDs (`vocab_size = 154880`) */
#define STRATA_GLM53_TOKEN_ENDOFTEXT            154820U /* `<|endoftext|>` (pad_token_id & EOS #0) */
#define STRATA_GLM53_TOKEN_MASK                 154821U /* `[MASK]` */
#define STRATA_GLM53_TOKEN_GMASK                154822U /* `[gMASK]` (BOS prefix #0) */
#define STRATA_GLM53_TOKEN_SMASK                154823U /* `[sMASK]` */
#define STRATA_GLM53_TOKEN_SOP                  154824U /* `<sop>`   (BOS prefix #1) */
#define STRATA_GLM53_TOKEN_EOP                  154825U /* `<eop>` */
#define STRATA_GLM53_TOKEN_SYSTEM               154826U /* `<|system|>` */
#define STRATA_GLM53_TOKEN_USER                 154827U /* `<|user|>` (EOS #1) */
#define STRATA_GLM53_TOKEN_ASSISTANT            154828U /* `<|assistant|>` */
#define STRATA_GLM53_TOKEN_OBSERVATION          154829U /* `<|observation|>` (EOS #2) */
#define STRATA_GLM53_SPECIAL_ID_MIN             154820U /* `special=true` in tokenizer.json: 154820..154837 */
#define STRATA_GLM53_SPECIAL_ID_MAX             154837U
#define STRATA_GLM53_TOKEN_THINK_START          154841U /* `<think>`  (added_token, special=false) */
#define STRATA_GLM53_TOKEN_THINK_END            154842U /* `</think>` (added_token, special=false) */
#define STRATA_GLM53_ADDED_ID_MAX               154855U /* `<|video|>`: highest added_token ID */

/* Default Regime II (v1) & RAM-Resident E8S-18-V4 (v2) bundle paths, F_NOCACHE FD pool size & runtime defaults */
#define STRATA_GLM53_SPINE_FILENAME             "spine.strata"
#define STRATA_GLM53_EXPERTS_NVME_FILENAME      "experts_nvme.strata"
#define STRATA_GLM53_EXPERTS_PART0_FILENAME     "experts_uma_part0.strata"
#define STRATA_GLM53_EXPERTS_PART1_FILENAME     "experts_uma_part1.strata"
#define STRATA_GLM53_MTP_EXPERTS_FILENAME       "mtp_experts.strata"
#define STRATA_GLM53_DEFAULT_HF_REPO            "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"
#define STRATA_GLM53_DEFAULT_FAST_HF_REPO       "zai-org/GLM-5.3-Flash"
#ifndef __METAL_VERSION__
#include "strata_paths.h"
#define STRATA_GLM53_DEFAULT_BUNDLE_DIR         strata_apex_bundle_path("glm-5.3-flash")
#define STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR    strata_apex_bundle_path("glm-5.3-flash-fast")
#endif

#define STRATA_GLM53_NUM_NVME_FDS               8U       /* 8 parallel `F_NOCACHE` FDs to `experts_nvme.strata` */
#define STRATA_GLM53_DEFAULT_MAX_SEQ_LEN        33792U   /* 32K (32,768) + 1,024 decode headroom */
#define STRATA_GLM53_MAX_SEQ_LEN_CAP            1048576U /* 1M native NoPE context ceiling */
#define STRATA_GLM53_DEFAULT_HOT_CACHE_GIB      76.0f    /* 76.0 GiB = 5,765 / 12,384 experts (46.6% in UMA, auto-clamped to GPU working set) */
#define STRATA_GLM53_DEFAULT_RING_DMA_GIB       2.0f     /* 2.0 GiB Ring_DMA staging (~151 cold expert slots) */
#define STRATA_GLM53_DEFAULT_LOOKAHEAD_HORIZON  4U       /* 16c ANE multi-horizon router lookahead (l+1 .. l+4) */
#define STRATA_GLM53_DEFAULT_ADAPTIVE_TAU       0.20f    /* Min scaled gate weight (out of sum=2.5) for cold fetch */
#define STRATA_GLM53_DEFAULT_ADAPTIVE_COLD_CAP  2U       /* Max cold NVMe experts/layer in STRATA_GLM53_MODE_ADAPTIVE */

/* Step 4 Paged DSA MLA KV block constants (shared Host C++ & Metal 4 MSL) */
#define STRATA_GLM53_KV_LOC_UMA_BIT             0x80000000U
#define STRATA_GLM53_KV_LOC_IDX_MASK            0x7FFFFFFFU
#define STRATA_GLM53_KV_BLOCK_TOKENS            4U       /* == index_kpool */
#define STRATA_GLM53_KV_BLOCK_BYTES             4096U    /* 4 tok x 512 FP16 latent c_kv = exact 4 KiB block! */
#define STRATA_GLM53_COMPACT_SINK_BLOCKS        32U      /* 128 pinned attention-sink tokens */
#define STRATA_GLM53_COMPACT_LOCAL_BLOCKS       256U     /* 1,024 pinned recent tokens */
#define STRATA_GLM53_COMPACT_MIN_POOL_TOKENS    2560U    /* >= top-512 pools + sinks + local + chunk */

/* ============================================================================
 * 2. Page-Aligned On-Disk `spine.strata` Descriptors (POD, 8-Byte Aligned)
 * ============================================================================ */

/*
 * 4-Bit Affine Matrix Descriptor inside `spine.strata` (`group_size = 64`, `sizeof = 64` B):
 *   - `weight_off`: 256B-aligned offset in `spine.strata` to `uint32_t [rows_padded, cols / 8]`
 *   - `scales_off`: 256B-aligned offset in `spine.strata` to IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *   - `biases_off`: 256B-aligned offset in `spine.strata` to IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *   - Dequantization: `w[r, c] = float((u32 >> (4 * (c & 7))) & 0xF) * float(scale[r, c >> 6]) + float(bias[r, c >> 6])`
 */
typedef struct StrataGlm53QuantMatrix {
    sg53_u64 weight_off;
    sg53_u64 weight_bytes;
    sg53_u64 scales_off;
    sg53_u64 scales_bytes;
    sg53_u64 biases_off;
    sg53_u64 biases_bytes;
    sg53_u32 rows_logical;
    sg53_u32 rows_padded;
    sg53_u32 cols;
    sg53_u32 group_size;
} StrataGlm53QuantMatrix;

/*
 * Dense 1D/2D Tensor Slice Descriptor inside `spine.strata` (`sizeof = 32` B, 256B-aligned):
 *   - `moe_router_gate` (`[288, 4096]`), `mla_w_uk` (`[32768, 256]`), and `mla_w_uv` (`[16384, 512]`)
 *     are stored as `STRATA_GLM53_DTYPE_F16` (`uint16_t` / `half`).
 *   - All RMSNorms, LayerNorm `k_norm.{weight,bias}`, `hc_{attn,ffn}_{fn,base,scale}`,
 *     `index_kpool_compress_{ape,gate}`, `moe_e_score_bias`, `conv1d`, `A_log`, and `dt_bias`
 *     are stored as lossless `STRATA_GLM53_DTYPE_F32` (`float`).
 */
typedef struct StrataGlm53TensorSlice {
    sg53_u64 offset;
    sg53_u64 bytes;
    sg53_u32 elem_count;
    sg53_u32 dtype;
    sg53_u32 shape0;
    sg53_u32 shape1;
} StrataGlm53TensorSlice;

/* 4-Stream Manifold-Constrained Hyper-Connection Site (`hc_attn` or `hc_ffn`, `sizeof = 96` B) */
typedef struct StrataGlm53MhcDesc {
    StrataGlm53TensorSlice fn;    /* F32 [24, 16384] (`(2 + hc_mult)*hc_mult, hc_mult*D`) */
    StrataGlm53TensorSlice base;  /* F32 [24]        (`pre_b[4], post_b[4], comb_b[16]`)  */
    StrataGlm53TensorSlice scale; /* F32 [3]         (`{pre_scale, post_scale, comb_scale}`) */
} StrataGlm53MhcDesc;

/*
 * Indirect Routed-Expert Dispatch Entry (16 B, shared Host C++ <-> Metal 4 MSL):
 * Allows a single fused indirect-indexed MoE Metal kernel to evaluate any mix of `UMA Hot-Expert Cache`
 * (`source_buf == STRATA_GLM53_EXPERT_SRC_HOT_UMA`, reading `b_hot_experts` at `slot_idx * expert_record_bytes`)
 * and `Ring_DMA` cold NVMe experts (`source_buf == STRATA_GLM53_EXPERT_SRC_RING_DMA`, reading `b_ring_dma`
 * at `slot_idx * expert_record_bytes`) in ONE pass!
 */
typedef struct StrataGlm53ExpertDispatch {
    sg53_u32 expert_id;   /* Logical expert ID `0 .. 287` */
    sg53_u32 source_buf;  /* STRATA_GLM53_EXPERT_SRC_HOT_UMA (0) or STRATA_GLM53_EXPERT_SRC_RING_DMA (1) */
    sg53_u32 slot_idx;    /* Physical slot index inside `b_hot_experts` or `b_ring_dma` */
    float    gate_weight; /* Normalized & `routed_scaling_factor` (2.5x) scaled router weight `w[i]` */
} StrataGlm53ExpertDispatch;

/* Per-Layer Spine Descriptor (`layers[0..44]` in `SpineHeader` + `mtp.layer` for Layer `45`) */
typedef struct StrataGlm53LayerDesc {
    sg53_u32 global_layer_idx; /* 0..44 for main backbone, 45 for `mtp.layer` */
    sg53_u32 attn_type;        /* STRATA_GLM53_ATTN_KDA_LINEAR (0) or ATTN_DSA_MLA (1) */
    sg53_u32 mlp_type;         /* STRATA_GLM53_MLP_DENSE (0) or MLP_SPARSE_MOE (1) */
    sg53_u32 has_mhc;          /* 1 for main layers 0..44; 0 for single-stream MTP layer 45! */
    sg53_i32 moe_layer_idx;    /* -1 for dense layers 0..2; 0..41 for main MoE 3..44; 42 for MTP 45 */
    sg53_u32 reserved0;
    sg53_u64 layer_page_off;   /* 16 KiB-aligned start offset inside `spine.strata` */
    sg53_u64 layer_page_bytes; /* 16 KiB-aligned byte size of this layer's spine payload */

    /* Populated iff `has_mhc == 1` (layers 0..44) */
    StrataGlm53MhcDesc        hc_attn;
    StrataGlm53MhcDesc        hc_ffn;

    /* Pre-attn & post-attn RMSNorms (`F32 [4096]`, `eps = 1e-5f`, NO `+1.0f`) */
    StrataGlm53TensorSlice    input_layernorm;
    StrataGlm53TensorSlice    post_attention_layernorm;

    /* Populated iff `attn_type == STRATA_GLM53_ATTN_KDA_LINEAR` (34 layers) */
    StrataGlm53QuantMatrix    lin_qkv_proj;         /* Q4  [24576, 4096]: q[0..8191], k[8192..16383], v[16384..24575] */
    StrataGlm53QuantMatrix    lin_fbg_a_proj;       /* Q4  [320,   4096]: f_a[0..127], b[128..191], g_a[192..319]     */
    StrataGlm53QuantMatrix    lin_f_b_proj;         /* Q4  [8192,  128]                                               */
    StrataGlm53QuantMatrix    lin_g_b_proj;         /* Q4  [8192,  128]                                               */
    StrataGlm53QuantMatrix    lin_o_proj;           /* Q4  [4096,  8192]                                              */
    StrataGlm53TensorSlice    lin_qkv_conv1d;       /* F32 [24576, 4] causal taps [t-3, t-2, t-1, t]                  */
    StrataGlm53TensorSlice    lin_A_log;            /* F32 [64]                                                       */
    StrataGlm53TensorSlice    lin_dt_bias;          /* F32 [8192] (`[64, 128]` per-channel bias!)                     */
    StrataGlm53TensorSlice    lin_o_norm;           /* F32 [128]  (`eps = 1e-5f`, sigmoid gate, NO +1.0f)             */

    /* Populated iff `attn_type == STRATA_GLM53_ATTN_DSA_MLA` (11 main layers + MTP layer 45) */
    StrataGlm53QuantMatrix    mla_qkv_a_proj;       /* Q4  [2048,  4096]: q_a[0..1535], kv_a_with_mqa[1536..2047]     */
    StrataGlm53TensorSlice    mla_q_a_layernorm;    /* F32 [1536] (`eps = 1e-5f`)                                     */
    StrataGlm53QuantMatrix    mla_q_b_proj;         /* Q4  [16384, 1536] -> q[64, 256]                                */
    StrataGlm53TensorSlice    mla_kv_a_layernorm;   /* F32 [512]  (`eps = 1e-5f`) -> cached latent c_kv [512]         */
    StrataGlm53QuantMatrix    mla_kv_b_k;           /* Q4  [16384, 512]  (`[64, 256, 512]` lossless W_K slice)        */
    StrataGlm53QuantMatrix    mla_kv_b_v;           /* Q4  [16384, 512]  (`[64, 256, 512]` lossless W_V slice)        */
    StrataGlm53TensorSlice    mla_w_uk;             /* F16 [32768, 256]  (`[64, 512, 256]` transposed W_K^T for Q!)   */
    StrataGlm53TensorSlice    mla_w_uv;             /* F16 [16384, 512]  (`[64, 256, 512]` dequantized W_V for V!)    */
    StrataGlm53QuantMatrix    mla_o_proj;           /* Q4  [4096,  16384]                                             */

    /* Glm5NextIndexer (`indexer_types == "full"` on all 12 DSA layers) */
    StrataGlm53QuantMatrix    indexer_wq_b;         /* Q4  [4096,  1536] -> q_idx[32, 128]                            */
    StrataGlm53QuantMatrix    indexer_wk;           /* Q4  [128,   4096] -> raw_k[128]                                */
    StrataGlm53QuantMatrix    indexer_weights_proj; /* Q4  [32,    4096] (`rows_logical=32, rows_padded=64`)          */
    StrataGlm53TensorSlice    indexer_k_norm_weight;/* F32 [128]  (`nn.LayerNorm(128, eps=1e-6f)` weight)             */
    StrataGlm53TensorSlice    indexer_k_norm_bias;  /* F32 [128]  (`nn.LayerNorm(128, eps=1e-6f)` bias!)              */
    StrataGlm53TensorSlice    indexer_compress_ape; /* F32 [4,     128]  (`index_kpool_compress_ape`)                 */
    StrataGlm53TensorSlice    indexer_compress_gate;/* F32 [128,   4096] (`index_kpool_compress_gate`)                */

    /* Populated iff `mlp_type == STRATA_GLM53_MLP_DENSE` (layers 0, 1, 2; `d_ff = 12288`) */
    StrataGlm53QuantMatrix    dense_gate_up_proj;   /* Q4  [24576, 4096]: gate[0..12287], up[12288..24575]            */
    StrataGlm53QuantMatrix    dense_down_proj;      /* Q4  [4096,  12288]                                             */

    /* Populated iff `mlp_type == STRATA_GLM53_MLP_SPARSE_MOE` (`moe_layer_idx >= 0`: layers 3..44 + MTP 45)
     * Routed `288` experts live in `experts_nvme.strata` at `(moe_layer_idx * 288 + e) * expert_record_bytes`! */
    StrataGlm53TensorSlice    moe_router_gate;      /* F16 [288,   4096] (for Metal 4 GPU + 16-Core ANE probes!)      */
    StrataGlm53TensorSlice    moe_e_score_bias;     /* F32 [288]         (`e_score_correction_bias`)                  */
    StrataGlm53QuantMatrix    shared_gate_up_proj;  /* Q4  [4096,  4096]: gate[0..2047], up[2048..4095]               */
    StrataGlm53QuantMatrix    shared_down_proj;     /* Q4  [4096,  2048]                                              */
} StrataGlm53LayerDesc;

/* Native 1-Layer Multi-Token Prediction (`model.language_model.layers.45.*`) Descriptor */
typedef struct StrataGlm53MtpDesc {
    sg53_u32               enabled;          /* 1 (`num_nextn_predict_layers == 1`) */
    sg53_u32               global_layer_idx; /* 45 */
    sg53_i32               moe_layer_idx;    /* 42 (`experts 42*288 .. 42*288 + 287` in `experts_nvme.strata`) */
    sg53_u32               reserved0;
    sg53_u64               mtp_page_off;     /* 16 KiB-aligned start offset in `spine.strata` */
    sg53_u64               mtp_page_bytes;   /* 16 KiB-aligned total spine byte size of MTP module */
    StrataGlm53TensorSlice enorm;            /* F32 [4096] (`layers.45.enorm.weight`, `eps = 1e-5f`) */
    StrataGlm53TensorSlice hnorm;            /* F32 [4096] (`layers.45.hnorm.weight`, `eps = 1e-5f`) */
    StrataGlm53QuantMatrix eh_proj;          /* Q4  [4096, 8192] (`concat(enorm(e), hnorm(h))`) */
    StrataGlm53LayerDesc   layer;            /* Single-stream (`has_mhc = 0`) DSA MLA + 288E MoE spine */
    StrataGlm53TensorSlice shared_head_norm; /* F32 [4096] (`layers.45.shared_head.norm.weight`) -> `lm_head` */
} StrataGlm53MtpDesc;

/* Complete Model Hyperparameters stored in `StrataGlm53SpineHeader` */
typedef struct StrataGlm53Config {
    sg53_u32 num_hidden_layers;            /* 45 */
    sg53_u32 num_nextn_predict_layers;     /* 1 (Layer 45 MTP) */
    sg53_u32 first_k_dense_replace;        /* 3 (Layers 0..2 dense, 3..44 sparse MoE) */
    sg53_u32 hidden_size;                  /* 4096 */
    sg53_u32 vocab_size;                   /* 154880 */
    sg53_u32 intermediate_size;            /* 12288 (dense layers 0..2) */
    sg53_u32 num_kda_layers;               /* 34 */
    sg53_u32 num_dsa_layers;               /* 11 (main) */

    /* Manifold-Constrained Hyper-Connections (`mHC`) */
    sg53_u32 mhc_enabled;                  /* 1 */
    sg53_u32 hc_mult;                      /* 4 (`hc_dim = 16384`, `mix_dim = 24`) */
    sg53_u32 hc_sinkhorn_iters;            /* 20 */
    float    hc_eps;                       /* 1e-6f */

    /* `KDA` Key-Driven Linear Attention (`34` layers) */
    sg53_u32 linear_num_heads;             /* 64 */
    sg53_u32 linear_head_dim;              /* 128 (`qkv_dim = 8192`, `conv_dim = 24576`) */
    sg53_u32 linear_conv_kernel_dim;       /* 4 */
    float    linear_gate_lower_bound;      /* -5.0f (`decay = exp(-5.0 * sigmoid(exp(A_log)*(a + dt_bias)))`) */

    /* `DSA` NoPE Multi-Head Latent Attention + `Glm5NextIndexer` (`11` main + `1` MTP) */
    sg53_u32 num_attention_heads;          /* 64 */
    sg53_u32 num_key_value_heads;          /* 64 */
    sg53_u32 q_lora_rank;                  /* 1536 */
    sg53_u32 kv_lora_rank;                 /* 512 */
    sg53_u32 qk_nope_head_dim;             /* 256 */
    sg53_u32 qk_rope_head_dim;             /* 0 (`mla_use_nope = 1` -> ZERO RoPE!) */
    sg53_u32 v_head_dim;                   /* 256 */
    sg53_u32 mla_use_nope;                 /* 1 */
    sg53_u32 index_n_heads;                /* 32 */
    sg53_u32 index_head_dim;               /* 128 */
    sg53_u32 index_kpool;                  /* 4 */
    sg53_u32 index_topk;                   /* 2048 (`select_k = 2048 / 4 = 512` pools) */
    sg53_u32 index_always_select_tail;     /* 1 */
    float    indexer_layernorm_eps;        /* 1e-6f (`nn.LayerNorm(128, eps=1e-6)`) */

    /* `288`-Expert `noaux_tc` Sigmoid MoE + Shared Expert + `LimitedSwiGLU` */
    sg53_u32 n_routed_experts;             /* 288 */
    sg53_u32 num_experts_per_tok;          /* 8 */
    sg53_u32 n_shared_experts;             /* 1 */
    sg53_u32 moe_intermediate_size;        /* 2048 */
    sg53_u32 n_group;                      /* 1 */
    sg53_u32 topk_group;                   /* 1 */
    sg53_u32 norm_topk_prob;               /* 1 */
    float    routed_scaling_factor;        /* 2.5f */
    float    swiglu_limit;                 /* 10.0f (`min(gate, 10) & clamp(up, -10, 10)`) */

    /* Normalization, Quantization, Per-Layer Schedule & Special Tokens */
    float    rms_norm_eps;                 /* 1e-5f */
    sg53_u32 quant_group_size;             /* 64 */
    sg53_u32 quant_bits;                   /* 4 */
    sg53_u32 tie_word_embeddings;          /* 0 */
    sg53_u32 max_position_embeddings;      /* 1048576 */
    sg53_u32 pad_token_id;                 /* 154820 */
    sg53_u32 bos_gmask_token_id;           /* 154822 (`[gMASK]`) */
    sg53_u32 bos_sop_token_id;             /* 154824 (`<sop>`) */
    sg53_u32 eos_token_ids[4];             /* {154820, 154827, 154829, 0} */
    sg53_u32 num_eos_token_ids;            /* 3 */

    sg53_u8  layer_attn_types[STRATA_GLM53_MAX_LAYERS]; /* 0 = KDA_LINEAR, 1 = DSA_MLA */
    sg53_u8  layer_mlp_types[STRATA_GLM53_MAX_LAYERS];  /* 0 = DENSE, 1 = SPARSE_MOE   */
} StrataGlm53Config;

/*
 * Master Header at byte offset `0` of `<bundle_dir>/spine.strata` (`~5.3611 GiB`, exact multiple of `16,384 B`).
 * Also records the deterministic geometry of `<bundle_dir>/experts_nvme.strata` (`163.2656 GiB` =
 * `12,384 * 14,155,776 B`, headerless raw `256 KiB`-stripe-aligned expert store starting at offset `0`).
 */
typedef struct StrataGlm53SpineHeader {
    sg53_u64               magic;                      /* STRATA_GLM53_SPINE_MAGIC_U64 */
    sg53_u32               version;                    /* STRATA_GLM53_FORMAT_VERSION */
    sg53_u32               header_padded_bytes;        /* 131072 (8 * 16 KiB) */
    sg53_u64               total_spine_bytes;          /* Exact multiple of 16384 (~5.3611 GiB) */
    sg53_u64               experts_nvme_bytes;         /* 12384 * 14155776 = 175,305,129,984 B (163.27 GiB) */
    sg53_u64               expert_record_bytes;        /* 14155776 (13.5000 MiB = 54 * 256 KiB = 864 * 16 KiB) */
    sg53_u64               expert_proj_bytes;          /* 4718592  (4.5000 MiB per gate/up/down sub-proj) */
    sg53_u64               expert_weight_bytes;        /* 4194304  (4.0000 MiB per sub-proj U32 quants) */
    sg53_u64               expert_scales_bytes;        /* 262144   (256 KiB per sub-proj F16 scales/biases) */
    sg53_u32               num_main_moe_layers;        /* 42 (layers 3..44) */
    sg53_u32               num_total_moe_layers;       /* 43 (42 main + 1 MTP layer 45) */
    sg53_u32               total_routed_experts;       /* 12384 (43 * 288) */
    sg53_u32               reserved0;
    sg53_u64               active_spine_bytes_per_tok; /* Main 45L + head spine bytes read per K=0 token */
    sg53_u64               active_routed_bytes_per_tok;/* 42 * 8 * 14,155,776 = 4,756,340,736 B (4.430 GiB) */
    sg53_u64               active_bytes_per_tok;       /* `active_spine + active_routed` (~8.8 GiB) */
    sg53_u64               mtp_active_bytes_per_draft; /* 1L MTP spine + top-8 experts + shared lm_head */

    StrataGlm53Config      cfg;

    /* Global Token Embedding (`[154880, 4096]` Q4 affine) — starts on 16 KiB page boundary */
    sg53_u64               embed_page_off;
    sg53_u64               embed_page_bytes;
    StrataGlm53QuantMatrix embed_tokens;

    /* 45 Main Hybrid Decoder Layer Spines (`0..44`) — each starts on a 16 KiB page boundary */
    StrataGlm53LayerDesc   layers[STRATA_GLM53_MAX_LAYERS];

    /* Final RMSNorm (`F32 [4096]`, `eps = 1e-5f`) + Untied LM Head (`[154880, 4096]` Q4) — 16 KiB aligned */
    sg53_u64               head_page_off;
    sg53_u64               head_page_bytes;
    StrataGlm53TensorSlice final_norm;
    StrataGlm53QuantMatrix lm_head;

    /* Native 1-Layer Multi-Token Prediction (`layers.45.*`) */
    StrataGlm53MtpDesc     mtp;

    /* ---- v2 RAM-Resident `RHT-64 + E8S-18-V4` Extension (`version == 2`; guaranteed zero in v1) ---- */
    sg53_u32               is_ram_resident;      /* 1 for v2 RAM-resident UMA, 0 for v1 Regime II NVMe */
    sg53_u32               main_expert_dtype;    /* STRATA_GLM53_EXPERT_DTYPE_E8S18_V4 (1) or Q4_A (0) */
    sg53_u32               mtp_expert_dtype;     /* STRATA_GLM53_EXPERT_DTYPE_Q4_A (0) */
    sg53_u32               part0_moe_layers;     /* 21 (main MoE layers `0..20` = L3..L23) */
    sg53_u32               part1_moe_layers;     /* 21 (main MoE layers `21..41` = L24..L44) */
    sg53_u32               v2_reserved0;
    sg53_u64               part0_bytes;          /* 45,185,236,992 B (42.082 GiB, % 16 KiB == 0) */
    sg53_u64               part1_bytes;          /* 45,185,236,992 B (42.082 GiB, % 16 KiB == 0) */
    sg53_u64               mtp_experts_bytes;    /*  4,076,863,488 B ( 3.797 GiB, % 16 KiB == 0) */
    sg53_u64               e8_record_bytes;      /*      7,471,104 B ( 7.125 MiB = 456 * 16 KiB) */
    sg53_u64               e8_proj_bytes;        /*      2,490,368 B ( 2.375 MiB per gate/up/down) */
    sg53_u64               e8_lo16_bytes;        /*      2,097,152 B ( 2.000 MiB = uint16[R, K/8]) */
    sg53_u64               e8_hi2_bytes;         /*        262,144 B ( 0.250 MiB = uint16[R, K/64]) */
    sg53_u64               e8_scale_bytes;       /*        131,072 B ( 0.125 MiB = uint8 [R, K/64]) */
    sg53_u64               rht_seed;             /* STRATA_GLM53_RHT64_SEED (`0x5354524154414538ULL`) */
    sg53_u8                moe_layer_quant[STRATA_GLM53_MAX_LAYERS]; /* Per-MoE-layer quant tag (`0..42`) */
} StrataGlm53SpineHeader;

/* ============================================================================
 * 3. Host Deterministic Layout Planner, Unpack Helpers, Bundle & Engine C ABI
 * ============================================================================ */

#ifndef __METAL_VERSION__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
static_assert(sizeof(StrataGlm53QuantMatrix) == 64U, "QuantMatrix must be 64 bytes");
static_assert(sizeof(StrataGlm53TensorSlice) == 32U, "TensorSlice must be 32 bytes");
static_assert(sizeof(StrataGlm53MhcDesc) == 96U, "MhcDesc must be 96 bytes");
static_assert(sizeof(StrataGlm53ExpertDispatch) == 16U, "ExpertDispatch must be 16 bytes");
static_assert(sizeof(StrataGlm53E8Half8Entry) == 16U, "E8Half8Entry must be 16 bytes");
static_assert(sizeof(StrataGlm53SpineHeader) <= STRATA_GLM53_HEADER_PADDED_BYTES,
              "SpineHeader exceeds 128 KiB padded header region");
static_assert((STRATA_GLM53_EXPERT_RECORD_BYTES % STRATA_GLM53_NVME_STRIPE_BYTES) == 0ULL,
              "Expert record must be an exact multiple of 256 KiB NVMe stripe");
static_assert((STRATA_GLM53_EXPERT_RECORD_BYTES % STRATA_GLM53_PAGE_BYTES) == 0ULL,
              "Expert record must be an exact multiple of 16 KiB VM page");
static_assert((STRATA_GLM53_E8S18_RECORD_BYTES % STRATA_GLM53_PAGE_BYTES) == 0ULL,
              "E8S-18-V4 expert record must be an exact multiple of 16 KiB VM page");
extern "C" {
#endif

/* ---- 3.1 Bit-Exact IEEE-754 BF16 / FP16 / FP32 & MLA `kv_b_proj` Unpack Helpers ---- */

#if defined(__GNUC__) || defined(__clang__)
#define SG53_INLINE static inline __attribute__((unused))
#else
#define SG53_INLINE static inline
#endif

SG53_INLINE uint64_t sg53_align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

SG53_INLINE float sg53_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

SG53_INLINE float sg53_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000U)) << 16;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t bits;
    if (exp == 0U) {
        if (mant == 0U) {
            bits = sign;
        } else {
            exp = 1U;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + 112U) << 23) | (mant << 13);
        }
    } else if (exp == 31U) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 112U) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

SG53_INLINE uint16_t sg53_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFU;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000U;
        uint32_t shift    = (uint32_t)(14 - exp);
        uint32_t half_bit = 1U << (shift - 1U);
        uint32_t sticky   = (mant & (half_bit - 1U)) != 0U ? 1U : 0U;
        uint32_t lsb      = (mant >> shift) & 1U;
        mant = (mant + half_bit - 1U + (lsb | sticky)) >> shift;
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        if (((x >> 23) & 0xFFU) == 0xFFU && mant != 0U) {
            return (uint16_t)(sign | 0x7E00U); /* NaN */
        }
        return (uint16_t)(sign | 0x7C00U); /* Clamp / Inf */
    }
    uint32_t round_bit = (mant >> 12) & 1U;
    uint32_t sticky    = (mant & 0xFFFU) != 0U ? 1U : 0U;
    uint32_t lsb       = (mant >> 13) & 1U;
    mant = (mant >> 13) + (round_bit & (lsb | sticky));
    if (mant >= 0x0400U) {
        mant = 0U;
        ++exp;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

SG53_INLINE uint16_t sg53_bf16_to_f16(uint16_t bf16) {
    return sg53_f32_to_f16(sg53_bf16_to_f32(bf16));
}

/*
 * Pack-time helper for `@glm53-stream`: unpacks ONE head `h` (`512` rows = `qk_nope_dim + v_dim` by
 * `kv_lora_rank` cols) of raw `.safetensors` `self_attn.kv_b_proj` (`U32` weight + `BF16` scales/biases) into:
 *   1. `out_k_w_u32 [qk_nope_dim * (kv_lora_rank / 8)]`, `out_k_s_f16`, `out_k_b_f16` (lossless Q4 `W_K[h]`)
 *   2. `out_v_w_u32 [v_dim       * (kv_lora_rank / 8)]`, `out_v_s_f16`, `out_v_b_f16` (lossless Q4 `W_V[h]`)
 *   3. `out_w_uk_f16 [kv_lora_rank * qk_nope_dim]`: TRANSPOSED `out_w_uk_f16[c * qk_nope_dim + r] = W_K[h][r, c]`
 *      so absorbed query projection `q_latent[h] = mla_w_uk[h] @ q[h]` is a contiguous row-major FP16 GEMV/GEMM!
 *   4. `out_w_uv_f16 [v_dim * kv_lora_rank]`: `out_w_uv_f16[r * kv_lora_rank + c] = W_V[h][r, c]`
 *      so absorbed value unembedding `v_out[h] = mla_w_uv[h] @ c_out[h]` is a contiguous row-major FP16 GEMV/GEMM!
 */
SG53_INLINE void sg53_unpack_kv_b_proj_head(
    const uint32_t* head_weight_u32,  /* `[ (qk_nope_dim + v_dim) * (kv_lora_rank / 8) ]` */
    const uint16_t* head_scales_bf16, /* `[ (qk_nope_dim + v_dim) * (kv_lora_rank / group_size) ]` */
    const uint16_t* head_biases_bf16, /* `[ (qk_nope_dim + v_dim) * (kv_lora_rank / group_size) ]` */
    uint32_t        qk_nope_dim,      /* 256 */
    uint32_t        v_dim,            /* 256 */
    uint32_t        kv_lora_rank,     /* 512 */
    uint32_t        group_size,       /* 64  */
    uint32_t*       out_k_w_u32,
    uint16_t*       out_k_s_f16,
    uint16_t*       out_k_b_f16,
    uint32_t*       out_v_w_u32,
    uint16_t*       out_v_s_f16,
    uint16_t*       out_v_b_f16,
    uint16_t*       out_w_uk_f16,     /* `[kv_lora_rank, qk_nope_dim]` transposed! */
    uint16_t*       out_w_uv_f16      /* `[v_dim, kv_lora_rank]` */
) {
    const uint32_t u32_per_row = kv_lora_rank >> 3U;
    const uint32_t grp_per_row = kv_lora_rank / group_size;

    /* 1. W_K[h]: rows `0 .. qk_nope_dim - 1` -> lossless Q4 + transposed FP16 `[kv_lora_rank, qk_nope_dim]` */
    memcpy(out_k_w_u32, head_weight_u32, (size_t)qk_nope_dim * (size_t)u32_per_row * sizeof(uint32_t));
    for (uint32_t r = 0U; r < qk_nope_dim; ++r) {
        const uint32_t* w_row = head_weight_u32  + (size_t)r * (size_t)u32_per_row;
        const uint16_t* s_row = head_scales_bf16 + (size_t)r * (size_t)grp_per_row;
        const uint16_t* b_row = head_biases_bf16 + (size_t)r * (size_t)grp_per_row;
        for (uint32_t g = 0U; g < grp_per_row; ++g) {
            out_k_s_f16[(size_t)r * grp_per_row + g] = sg53_bf16_to_f16(s_row[g]);
            out_k_b_f16[(size_t)r * grp_per_row + g] = sg53_bf16_to_f16(b_row[g]);
        }
        for (uint32_t u = 0U; u < u32_per_row; ++u) {
            const uint32_t packed = w_row[u];
            const uint32_t c0     = u << 3U;
            const uint32_t g      = c0 / group_size;
            const float    sc     = sg53_bf16_to_f32(s_row[g]);
            const float    bi     = sg53_bf16_to_f32(b_row[g]);
            for (uint32_t j = 0U; j < 8U; ++j) {
                const uint32_t c = c0 + j;
                const float    v = (float)((packed >> (4U * j)) & 0xFU) * sc + bi;
                out_w_uk_f16[(size_t)c * (size_t)qk_nope_dim + (size_t)r] = sg53_f32_to_f16(v);
            }
        }
    }

    /* 2. W_V[h]: rows `qk_nope_dim .. qk_nope_dim + v_dim - 1` -> lossless Q4 + row-major FP16 `[v_dim, kv_lora_rank]` */
    const uint32_t* v_w_base = head_weight_u32  + (size_t)qk_nope_dim * (size_t)u32_per_row;
    const uint16_t* v_s_base = head_scales_bf16 + (size_t)qk_nope_dim * (size_t)grp_per_row;
    const uint16_t* v_b_base = head_biases_bf16 + (size_t)qk_nope_dim * (size_t)grp_per_row;
    memcpy(out_v_w_u32, v_w_base, (size_t)v_dim * (size_t)u32_per_row * sizeof(uint32_t));
    for (uint32_t r = 0U; r < v_dim; ++r) {
        const uint32_t* w_row = v_w_base + (size_t)r * (size_t)u32_per_row;
        const uint16_t* s_row = v_s_base + (size_t)r * (size_t)grp_per_row;
        const uint16_t* b_row = v_b_base + (size_t)r * (size_t)grp_per_row;
        uint16_t*       d_row = out_w_uv_f16 + (size_t)r * (size_t)kv_lora_rank;
        for (uint32_t g = 0U; g < grp_per_row; ++g) {
            out_v_s_f16[(size_t)r * grp_per_row + g] = sg53_bf16_to_f16(s_row[g]);
            out_v_b_f16[(size_t)r * grp_per_row + g] = sg53_bf16_to_f16(b_row[g]);
        }
        for (uint32_t u = 0U; u < u32_per_row; ++u) {
            const uint32_t packed = w_row[u];
            const uint32_t c0     = u << 3U;
            const uint32_t g      = c0 / group_size;
            const float    sc     = sg53_bf16_to_f32(s_row[g]);
            const float    bi     = sg53_bf16_to_f32(b_row[g]);
            for (uint32_t j = 0U; j < 8U; ++j) {
                const float v = (float)((packed >> (4U * j)) & 0xFU) * sc + bi;
                d_row[c0 + j] = sg53_f32_to_f16(v);
            }
        }
    }
}

/*
 * Lossless `BF16 -> FP16` unpack of ONE head `h` (`qk_nope_dim + v_dim` rows x `kv_lora_rank` cols)
 * from official `zai-org/GLM-5.3-Flash` `BF16` `self_attn.kv_b_proj.weight` directly into:
 *   1. `out_w_uk_f16 [kv_lora_rank, qk_nope_dim]` (transposed: `out_w_uk_f16[c * qk_nope_dim + r] = W_K[h][r, c]`)
 *   2. `out_w_uv_f16 [v_dim, kv_lora_rank]`       (row-major:  `out_w_uv_f16[r * kv_lora_rank + c] = W_V[h][r, c]`)
 */
SG53_INLINE void sg53_unpack_kv_b_proj_head_bf16(
    const uint16_t* head_weight_bf16, /* `[ (qk_nope_dim + v_dim) * kv_lora_rank ]` */
    uint32_t        qk_nope_dim,      /* 256 */
    uint32_t        v_dim,            /* 256 */
    uint32_t        kv_lora_rank,     /* 512 */
    uint16_t*       out_w_uk_f16,     /* `[kv_lora_rank, qk_nope_dim]` transposed! */
    uint16_t*       out_w_uv_f16      /* `[v_dim, kv_lora_rank]` */
) {
    for (uint32_t r = 0U; r < qk_nope_dim; ++r) {
        const uint16_t* row = head_weight_bf16 + (size_t)r * (size_t)kv_lora_rank;
        for (uint32_t c = 0U; c < kv_lora_rank; ++c) {
            out_w_uk_f16[(size_t)c * (size_t)qk_nope_dim + (size_t)r] = sg53_bf16_to_f16(row[c]);
        }
    }
    const uint16_t* v_base = head_weight_bf16 + (size_t)qk_nope_dim * (size_t)kv_lora_rank;
    const size_t v_elems = (size_t)v_dim * (size_t)kv_lora_rank;
    for (size_t i = 0U; i < v_elems; ++i) {
        out_w_uv_f16[i] = sg53_bf16_to_f16(v_base[i]);
    }
}

/* ---- 3.1b Deterministic `RHT-64`, `FP8 E4M3FN`, Unsigned `FP8 E4M4` & `E8S-18-V4` Host Helpers ---- */

SG53_INLINE uint64_t sg53_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

SG53_INLINE float sg53_rht64_sign_at(uint32_t idx) {
    return ((STRATA_GLM53_RHT64_SIGN_MASK >> (idx & 63U)) & 1ULL) ? -1.0f : 1.0f;
}

/* Exact orthogonal block-64 Randomized Hadamard Transform: `y = (1/8) * H_64 * diag(sign) * x` */
SG53_INLINE void sg53_rht64_f32(const float* in64, float* out64) {
    float v[64];
    for (uint32_t i = 0U; i < 64U; ++i) {
        v[i] = in64[i] * sg53_rht64_sign_at(i);
    }
    for (uint32_t len = 1U; len < 64U; len <<= 1U) {
        const uint32_t step = len << 1U;
        for (uint32_t base = 0U; base < 64U; base += step) {
            for (uint32_t j = 0U; j < len; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + len];
                v[base + j]       = a + b;
                v[base + j + len] = a - b;
            }
        }
    }
    for (uint32_t i = 0U; i < 64U; ++i) {
        out64[i] = v[i] * 0.125f;
    }
}

/* Exact IEEE/OCP `F8_E4M3FN` (no Inf; NaN at `0x7F` & `0xFF`, bias = 7, max = 448.0f) -> F32 */
SG53_INLINE bool sg53_fp8_e4m3fn_is_nan(uint8_t u) {
    return (u & 0x7FU) == 0x7FU;
}

SG53_INLINE float sg53_fp8_e4m3fn_to_f32(uint8_t u) {
    const uint32_t sign = (uint32_t)(u >> 7U);
    const uint32_t exp  = (uint32_t)((u >> 3U) & 0x0FU);
    const uint32_t mant = (uint32_t)(u & 0x07U);
    if (exp == 15U && mant == 7U) {
        const uint32_t nan_bits = (sign << 31U) | 0x7FC00000U;
        float out;
        memcpy(&out, &nan_bits, sizeof(out));
        return out;
    }
    float val;
    if (exp == 0U) {
        val = (float)mant * (1.0f / 512.0f); /* 2^-6 * (m / 8) */
    } else {
        const uint32_t bits = ((exp + 120U) << 23U) | (mant << 20U);
        memcpy(&val, &bits, sizeof(val));
    }
    return sign ? -val : val;
}

/* Unsigned `FP8 E4M4` (`bias = 14`, `0 -> 0.0h`, else `as_type<half>(ushort((uint(b) + 0x10u) << 6u))`) */
SG53_INLINE uint16_t sg53_e4m4_decode_f16(uint8_t b) {
    return (b == 0U) ? (uint16_t)0U : (uint16_t)(((uint32_t)b + 0x10U) << 6U);
}

SG53_INLINE float sg53_e4m4_decode_f32(uint8_t b) {
    return sg53_f16_to_f32(sg53_e4m4_decode_f16(b));
}

SG53_INLINE uint8_t sg53_e4m4_encode_f32(float s) {
    /* Min positive (`b=1`): 2^-14 * (17/16) = 6.4849853515625e-5f; midpoint to 0 is 3.24249267578125e-5f */
    if (!(s > 3.24249267578125e-5f)) return 0U;
    if (s <= 6.4849853515625e-5f)    return 1U;
    if (s >= 3.875f)                 return 255U;
    uint32_t bits;
    memcpy(&bits, &s, sizeof(bits));
    /* Nearest rounding to 4 mantissa bits (FP32 has 23 mantissa bits -> half-ULP at bit 18) */
    const uint32_t rounded = bits + (1U << 18U);
    const int32_t  e = (int32_t)((rounded >> 23U) & 0xFFU) - 127 + 14;
    const uint32_t m = (rounded >> 19U) & 0x0FU;
    if (e < 0) return 1U;
    const int32_t b = (e << 4) | (int32_t)m;
    if (b < 1)   return 1U;
    if (b > 255) return 255U;
    return (uint8_t)b;
}

/* Exact 16-codeword Extended Hamming `[8, 4, 4]` RM(1,3) sign mask table (`h in 0..15`) */
SG53_INLINE uint8_t sg53_h8_mask(uint32_t h) {
    static const uint8_t kH8[16] = {
        0x00U, 0xFFU, 0xAAU, 0x55U, 0xCCU, 0x33U, 0x66U, 0x99U,
        0xF0U, 0x0FU, 0x5AU, 0xA5U, 0x3CU, 0xC3U, 0x96U, 0x69U
    };
    return kH8[h & 15U];
}

/* Klein `V_4` XOR permutation index (`v4 in 0..3`: `sw = v4 & 1`, `zw = v4 & 2`) -> `m in {0, 4, 2, 6}` */
SG53_INLINE uint32_t sg53_v4_xor_mask(uint32_t v4) {
    return ((v4 & 1U) << 2U) | (v4 & 2U);
}

SG53_INLINE int sg53_cmp_u64_asc(const void* a, const void* b) {
    const uint64_t va = *(const uint64_t*)a;
    const uint64_t vb = *(const uint64_t*)b;
    return (va < vb) ? -1 : ((va > vb) ? 1 : 0);
}

/*
 * Fast `216 KiB` host quantizer lookup tables for `E8S-18-V4` (`O(16)` nearest-codeword search):
 *   - `lut[4096]` (`64 KiB`): exact canonical `Half8Entry` LUT (upload directly to Metal `b_e8s18_lut`!)
 *   - `key16_to_code[65536]` (`128 KiB`): direct table for all valid `n in {0..3}^8` -> `(orbit << 2) | v4`
 *   - `n4_keys[4096]` + `n4_codes[4096]` (`24 KiB`): open-addressed hash table for the `<= 1,184` `max_n == 4` tuples
 */
typedef struct StrataGlm53E8QuantTables {
    StrataGlm53E8Half8Entry lut[STRATA_GLM53_E8S18_NUM_ORBITS];
    uint16_t                key16_to_code[65536];
    uint32_t                n4_keys[4096];
    uint16_t                n4_codes[4096];
} StrataGlm53E8QuantTables;

SG53_INLINE void sg53_e8s18_enumerate_orbits_u64(uint64_t* out_keys, uint32_t* out_count) {
    uint32_t count = 0U;
    uint8_t n[8];
    const uint32_t odd_sq[5] = {1U, 9U, 25U, 49U, 81U};
    for (uint32_t n0 = 0U; n0 < 5U; ++n0) {
        const uint32_t s0 = odd_sq[n0];
        if (s0 + 7U > 128U) break;
        n[0] = (uint8_t)n0;
        for (uint32_t n1 = 0U; n1 < 5U; ++n1) {
            const uint32_t s1 = s0 + odd_sq[n1];
            if (s1 + 6U > 128U) break;
            n[1] = (uint8_t)n1;
            for (uint32_t n2 = 0U; n2 < 5U; ++n2) {
                const uint32_t s2 = s1 + odd_sq[n2];
                if (s2 + 5U > 128U) break;
                n[2] = (uint8_t)n2;
                for (uint32_t n3 = 0U; n3 < 5U; ++n3) {
                    const uint32_t s3 = s2 + odd_sq[n3];
                    if (s3 + 4U > 128U) break;
                    n[3] = (uint8_t)n3;
                    for (uint32_t n4 = 0U; n4 < 5U; ++n4) {
                        const uint32_t s4 = s3 + odd_sq[n4];
                        if (s4 + 3U > 128U) break;
                        n[4] = (uint8_t)n4;
                        for (uint32_t n5 = 0U; n5 < 5U; ++n5) {
                            const uint32_t s5 = s4 + odd_sq[n5];
                            if (s5 + 2U > 128U) break;
                            n[5] = (uint8_t)n5;
                            for (uint32_t n6 = 0U; n6 < 5U; ++n6) {
                                const uint32_t s6 = s5 + odd_sq[n6];
                                if (s6 + 1U > 128U) break;
                                n[6] = (uint8_t)n6;
                                for (uint32_t n7 = 0U; n7 < 5U; ++n7) {
                                    const uint32_t sq = s6 + odd_sq[n7];
                                    if (sq > 128U) break;
                                    n[7] = (uint8_t)n7;
                                    /* Keep only lexicographically minimum representative under V_4 = {0, 2, 4, 6} */
                                    int is_min = 1;
                                    const uint32_t ms[3] = {2U, 4U, 6U};
                                    for (uint32_t mi = 0U; mi < 3U; ++mi) {
                                        const uint32_t m = ms[mi];
                                        for (uint32_t i = 0U; i < 8U; ++i) {
                                            const uint8_t pi = n[i ^ m];
                                            if (pi < n[i]) { is_min = 0; break; }
                                            if (pi > n[i]) { break; }
                                        }
                                        if (!is_min) break;
                                    }
                                    if (!is_min) continue;
                                    uint32_t mx = n0;
                                    if (n1 > mx) mx = n1;
                                    if (n2 > mx) mx = n2;
                                    if (n3 > mx) mx = n3;
                                    if (n4 > mx) mx = n4;
                                    if (n5 > mx) mx = n5;
                                    if (n6 > mx) mx = n6;
                                    if (n7 > mx) mx = n7;
                                    const uint64_t key =
                                        ((uint64_t)sq << 40U) | ((uint64_t)mx << 36U) |
                                        ((uint64_t)n0 << 32U) | ((uint64_t)n1 << 28U) |
                                        ((uint64_t)n2 << 24U) | ((uint64_t)n3 << 20U) |
                                        ((uint64_t)n4 << 16U) | ((uint64_t)n5 << 12U) |
                                        ((uint64_t)n6 << 8U)  | ((uint64_t)n7 << 4U);
                                    out_keys[count++] = key;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    qsort(out_keys, count, sizeof(uint64_t), sg53_cmp_u64_asc);
    *out_count = count;
}

/* Deterministic `64 KiB` `Half8Entry` LUT builder (`4,096` canonical orbits, identical on Host & Metal GPU!) */
SG53_INLINE void strata_glm53_e8s18_build_lut(StrataGlm53E8Half8Entry* out_lut) {
    uint64_t keys[5120];
    uint32_t total = 0U;
    sg53_e8s18_enumerate_orbits_u64(keys, &total);
    for (uint32_t oid = 0U; oid < STRATA_GLM53_E8S18_NUM_ORBITS; ++oid) {
        const uint64_t k = keys[oid];
        for (uint32_t i = 0U; i < 8U; ++i) {
            const uint32_t ni = (uint32_t)((k >> (32U - 4U * i)) & 0xFU);
            const float mag   = (float)ni + 0.5f;
            const float val   = (ni & 1U) ? -mag : mag;
            out_lut[oid].v[i] = sg53_f32_to_f16(val);
        }
    }
}

/* Host reference decoder matching Metal `sg53_e8s18_decode_h4()` bit-for-bit! */
SG53_INLINE void sg53_e8s18_decode_vec8_f32(
    const StrataGlm53E8Half8Entry* lut,
    uint16_t lo16,
    uint32_t v4,
    float    out8[8]
) {
    const uint32_t oid = (uint32_t)lo16 >> 4U;
    const uint32_t h   = (uint32_t)lo16 & 15U;
    const uint32_t m   = sg53_v4_xor_mask(v4);
    const uint8_t  hm  = sg53_h8_mask(h);
    const StrataGlm53E8Half8Entry* e = &lut[oid];
    for (uint32_t i = 0U; i < 8U; ++i) {
        uint16_t bits = e->v[i ^ m];
        if ((hm >> i) & 1U) bits ^= 0x8000U;
        out8[i] = sg53_f16_to_f32(bits);
    }
}

SG53_INLINE void sg53_e8s18_build_quant_tables(StrataGlm53E8QuantTables* tbl) {
    memset(tbl->key16_to_code, 0xFF, sizeof(tbl->key16_to_code));
    memset(tbl->n4_keys, 0xFF, sizeof(tbl->n4_keys));
    memset(tbl->n4_codes, 0xFF, sizeof(tbl->n4_codes));

    uint64_t keys[5120];
    uint32_t total = 0U;
    sg53_e8s18_enumerate_orbits_u64(keys, &total);

    for (uint32_t oid = 0U; oid < STRATA_GLM53_E8S18_NUM_ORBITS; ++oid) {
        const uint64_t k = keys[oid];
        uint32_t n[8];
        uint32_t mx = 0U;
        for (uint32_t i = 0U; i < 8U; ++i) {
            n[i] = (uint32_t)((k >> (32U - 4U * i)) & 0xFU);
            if (n[i] > mx) mx = n[i];
            const float mag = (float)n[i] + 0.5f;
            tbl->lut[oid].v[i] = sg53_f32_to_f16((n[i] & 1U) ? -mag : mag);
        }
        for (uint32_t v4 = 0U; v4 < 4U; ++v4) {
            const uint32_t m    = sg53_v4_xor_mask(v4);
            const uint16_t code = (uint16_t)((oid << 2U) | v4);
            if (mx <= 3U) {
                uint32_t key16 = 0U;
                for (uint32_t i = 0U; i < 8U; ++i) {
                    key16 |= (n[i ^ m] & 3U) << (2U * i);
                }
                if (tbl->key16_to_code[key16] == 0xFFFFU) {
                    tbl->key16_to_code[key16] = code;
                }
            } else {
                uint32_t key24 = 0U;
                for (uint32_t i = 0U; i < 8U; ++i) {
                    key24 |= (n[i ^ m] & 7U) << (3U * i);
                }
                uint32_t slot = (key24 * 2654435761U) >> 20U;
                while (tbl->n4_keys[slot] != 0xFFFFFFFFU && tbl->n4_keys[slot] != key24) {
                    slot = (slot + 1U) & 4095U;
                }
                if (tbl->n4_keys[slot] == 0xFFFFFFFFU) {
                    tbl->n4_keys[slot]  = key24;
                    tbl->n4_codes[slot] = code;
                }
            }
        }
    }
}

SG53_INLINE uint16_t sg53_e8s18_lookup_n4(const StrataGlm53E8QuantTables* tbl, uint32_t key24) {
    uint32_t slot = (key24 * 2654435761U) >> 20U;
    while (tbl->n4_keys[slot] != 0xFFFFFFFFU) {
        if (tbl->n4_keys[slot] == key24) return tbl->n4_codes[slot];
        slot = (slot + 1U) & 4095U;
    }
    return 0xFFFFU;
}

/* Fast `O(16)` nearest-neighbor search in `E8S-18-V4` for an 8D scaled vector `y[0..7]` */
SG53_INLINE void sg53_e8s18_quantize_vec8(
    const StrataGlm53E8QuantTables* tbl,
    const float                     y[8],
    uint16_t*                       out_lo16,
    uint32_t*                       out_v4,
    float                           out_c[8]
) {
    static const uint8_t kH8[16] = {
        0x00U, 0xFFU, 0xAAU, 0x55U, 0xCCU, 0x33U, 0x66U, 0x99U,
        0xF0U, 0x0FU, 0x5AU, 0xA5U, 0x3CU, 0xC3U, 0x96U, 0x69U
    };
    float    c0[8], c1[8], e0[8], d[8];
    uint32_t n0[8], n1[8];
    uint32_t n4_m0 = 0U, n4_m1 = 0U;
    float    base_err = 0.0f;

    for (uint32_t i = 0U; i < 8U; ++i) {
        const float yi = y[i];
        float k0 = floorf(yi * 0.5f + 0.25f);
        if (k0 < -2.0f) k0 = -2.0f; else if (k0 > 2.0f) k0 = 2.0f;
        const float v0 = 2.0f * k0 + 0.5f;
        const uint32_t u0 = (uint32_t)fabsf(v0);
        const float r0 = yi - v0;

        float k1 = floorf(yi * 0.5f + 0.75f);
        if (k1 < -2.0f) k1 = -2.0f; else if (k1 > 2.0f) k1 = 2.0f;
        const float v1 = 2.0f * k1 - 0.5f;
        const uint32_t u1 = (uint32_t)fabsf(v1);
        const float r1 = yi - v1;

        c0[i] = v0; n0[i] = u0; e0[i] = r0 * r0;
        c1[i] = v1; n1[i] = u1; d[i]  = (r1 * r1) - e0[i];
        base_err += e0[i];
        if (u0 == 4U) n4_m0 |= (1U << i);
        if (u1 == 4U) n4_m1 |= (1U << i);
    }

    float    best_err  = 1e30f;
    uint16_t best_code = 0xFFFFU;
    uint32_t best_h    = 0U;

    for (uint32_t h = 0U; h < 16U; ++h) {
        const uint32_t mask = kH8[h];
        const float err = base_err
            + ((mask &   1U) ? d[0] : 0.0f) + ((mask &   2U) ? d[1] : 0.0f)
            + ((mask &   4U) ? d[2] : 0.0f) + ((mask &   8U) ? d[3] : 0.0f)
            + ((mask &  16U) ? d[4] : 0.0f) + ((mask &  32U) ? d[5] : 0.0f)
            + ((mask &  64U) ? d[6] : 0.0f) + ((mask & 128U) ? d[7] : 0.0f);
        if (err >= best_err) continue;

        const uint32_t has4 = ((~mask & n4_m0) | (mask & n4_m1)) & 0xFFU;
        uint16_t code;
        if (has4 == 0U) {
            const uint32_t key16 =
                (((mask &   1U) ? n1[0] : n0[0]))        |
                (((mask &   2U) ? n1[1] : n0[1]) <<  2U) |
                (((mask &   4U) ? n1[2] : n0[2]) <<  4U) |
                (((mask &   8U) ? n1[3] : n0[3]) <<  6U) |
                (((mask &  16U) ? n1[4] : n0[4]) <<  8U) |
                (((mask &  32U) ? n1[5] : n0[5]) << 10U) |
                (((mask &  64U) ? n1[6] : n0[6]) << 12U) |
                (((mask & 128U) ? n1[7] : n0[7]) << 14U);
            code = tbl->key16_to_code[key16];
        } else {
            const uint32_t key24 =
                (((mask &   1U) ? n1[0] : n0[0]))        |
                (((mask &   2U) ? n1[1] : n0[1]) <<  3U) |
                (((mask &   4U) ? n1[2] : n0[2]) <<  6U) |
                (((mask &   8U) ? n1[3] : n0[3]) <<  9U) |
                (((mask &  16U) ? n1[4] : n0[4]) << 12U) |
                (((mask &  32U) ? n1[5] : n0[5]) << 15U) |
                (((mask &  64U) ? n1[6] : n0[6]) << 18U) |
                (((mask & 128U) ? n1[7] : n0[7]) << 21U);
            code = sg53_e8s18_lookup_n4(tbl, key24);
        }
        if (code != 0xFFFFU) {
            best_err  = err;
            best_code = code;
            best_h    = h;
        }
    }

    if (best_code == 0xFFFFU) {
        /* 1-shot spherical projection fallback for outer-shell outlier vectors (`sq > 128`) */
        float n2 = 0.0f;
        for (uint32_t i = 0U; i < 8U; ++i) n2 += y[i] * y[i];
        const float inv_norm = 1.0f / (sqrtf(n2) + 1e-12f);
        static const float kRads[6] = {5.3f, 4.9f, 4.4f, 3.8f, 3.0f, 1.5f};
        for (uint32_t ri = 0U; ri < 6U && best_code == 0xFFFFU; ++ri) {
            const float sc = kRads[ri] * inv_norm;
            n4_m0 = 0U; n4_m1 = 0U; base_err = 0.0f;
            for (uint32_t i = 0U; i < 8U; ++i) {
                const float ys = y[i] * sc;
                float k0 = floorf(ys * 0.5f + 0.25f);
                if (k0 < -2.0f) k0 = -2.0f; else if (k0 > 2.0f) k0 = 2.0f;
                const float v0 = 2.0f * k0 + 0.5f;
                const uint32_t u0 = (uint32_t)fabsf(v0);
                const float r0 = y[i] - v0;

                float k1 = floorf(ys * 0.5f + 0.75f);
                if (k1 < -2.0f) k1 = -2.0f; else if (k1 > 2.0f) k1 = 2.0f;
                const float v1 = 2.0f * k1 - 0.5f;
                const uint32_t u1 = (uint32_t)fabsf(v1);
                const float r1 = y[i] - v1;

                c0[i] = v0; n0[i] = u0; e0[i] = r0 * r0;
                c1[i] = v1; n1[i] = u1; d[i]  = (r1 * r1) - e0[i];
                base_err += e0[i];
                if (u0 == 4U) n4_m0 |= (1U << i);
                if (u1 == 4U) n4_m1 |= (1U << i);
            }
            for (uint32_t h = 0U; h < 16U; ++h) {
                const uint32_t mask = kH8[h];
                const float err = base_err
                    + ((mask &   1U) ? d[0] : 0.0f) + ((mask &   2U) ? d[1] : 0.0f)
                    + ((mask &   4U) ? d[2] : 0.0f) + ((mask &   8U) ? d[3] : 0.0f)
                    + ((mask &  16U) ? d[4] : 0.0f) + ((mask &  32U) ? d[5] : 0.0f)
                    + ((mask &  64U) ? d[6] : 0.0f) + ((mask & 128U) ? d[7] : 0.0f);
                if (err >= best_err) continue;
                const uint32_t has4 = ((~mask & n4_m0) | (mask & n4_m1)) & 0xFFU;
                uint16_t code;
                if (has4 == 0U) {
                    const uint32_t key16 =
                        (((mask &   1U) ? n1[0] : n0[0]))        |
                        (((mask &   2U) ? n1[1] : n0[1]) <<  2U) |
                        (((mask &   4U) ? n1[2] : n0[2]) <<  4U) |
                        (((mask &   8U) ? n1[3] : n0[3]) <<  6U) |
                        (((mask &  16U) ? n1[4] : n0[4]) <<  8U) |
                        (((mask &  32U) ? n1[5] : n0[5]) << 10U) |
                        (((mask &  64U) ? n1[6] : n0[6]) << 12U) |
                        (((mask & 128U) ? n1[7] : n0[7]) << 14U);
                    code = tbl->key16_to_code[key16];
                } else {
                    const uint32_t key24 =
                        (((mask &   1U) ? n1[0] : n0[0]))        |
                        (((mask &   2U) ? n1[1] : n0[1]) <<  3U) |
                        (((mask &   4U) ? n1[2] : n0[2]) <<  6U) |
                        (((mask &   8U) ? n1[3] : n0[3]) <<  9U) |
                        (((mask &  16U) ? n1[4] : n0[4]) << 12U) |
                        (((mask &  32U) ? n1[5] : n0[5]) << 15U) |
                        (((mask &  64U) ? n1[6] : n0[6]) << 18U) |
                        (((mask & 128U) ? n1[7] : n0[7]) << 21U);
                    code = sg53_e8s18_lookup_n4(tbl, key24);
                }
                if (code != 0xFFFFU) {
                    best_err  = err;
                    best_code = code;
                    best_h    = h;
                }
            }
        }
    }

    const uint32_t bmask = kH8[best_h];
    for (uint32_t i = 0U; i < 8U; ++i) {
        out_c[i] = ((bmask >> i) & 1U) ? c1[i] : c0[i];
    }
    *out_lo16 = (uint16_t)(((uint32_t)(best_code >> 2U) << 4U) | best_h);
    *out_v4   = (uint32_t)(best_code & 3U);
}

/*
 * Quantizes ONE RHT-64-rotated 64-weight group `x64[0..63]` using `ols_trials` (1..3, default 3)
 * OLS scale search into `out_lo16[8]`, `*out_hi2`, and unsigned `FP8 E4M4` `*out_scale_u8`.
 * Optionally returns exact group signal energy (`sum x^2`) & quantization squared error (`sum (x - s*c)^2`).
 */
SG53_INLINE void sg53_e8s18_quantize_group64(
    const StrataGlm53E8QuantTables* tbl,
    const float                     x64[64],
    uint32_t                        ols_trials,
    uint16_t                        out_lo16[8],
    uint16_t*                       out_hi2,
    uint8_t*                        out_scale_u8,
    double*                         sig_energy_out,
    double*                         err_energy_out
) {
    float x2 = 0.0f;
    for (uint32_t i = 0U; i < 64U; ++i) x2 += x64[i] * x64[i];
    if (sig_energy_out) *sig_energy_out += (double)x2;

    const float rms = sqrtf(x2 * (1.0f / 64.0f));
    if (!(rms > 1e-8f)) {
        for (uint32_t k = 0U; k < 8U; ++k) out_lo16[k] = 0U;
        *out_hi2      = 0U;
        *out_scale_u8 = 0U;
        if (err_energy_out) *err_energy_out += (double)x2;
        return;
    }

    float targets[3];
    uint32_t n_trials;
    if (ols_trials <= 1U) {
        targets[0] = 1.67f;
        n_trials   = 1U;
    } else if (ols_trials == 2U) {
        targets[0] = 1.52f;
        targets[1] = 1.78f;
        n_trials   = 2U;
    } else {
        targets[0] = 1.45f;
        targets[1] = 1.65f;
        targets[2] = 1.85f;
        n_trials   = 3U;
    }

    float    best_mse   = 1e30f;
    uint16_t best_lo[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t best_hi    = 0U;
    uint8_t  best_sc    = 0U;

    for (uint32_t t = 0U; t < n_trials; ++t) {
        const float inv_s = targets[t] / rms;
        uint16_t cur_lo[8];
        uint16_t cur_hi = 0U;
        float dot_xc = 0.0f;
        float dot_cc = 0.0f;

        for (uint32_t k = 0U; k < 8U; ++k) {
            const float* xk = x64 + (k << 3U);
            float yk[8], ck[8];
            for (uint32_t i = 0U; i < 8U; ++i) yk[i] = xk[i] * inv_s;
            uint16_t lo16 = 0U;
            uint32_t v4   = 0U;
            sg53_e8s18_quantize_vec8(tbl, yk, &lo16, &v4, ck);
            cur_lo[k] = lo16;
            cur_hi   |= (uint16_t)((v4 & 3U) << (2U * k));
            for (uint32_t i = 0U; i < 8U; ++i) {
                dot_xc += xk[i] * ck[i];
                dot_cc += ck[i] * ck[i];
            }
        }

        const float   s_ols = (dot_cc > 0.0f) ? (dot_xc / dot_cc) : 0.0f;
        const int32_t b0    = (int32_t)sg53_e4m4_encode_f32(s_ols);
        for (int32_t db = -1; db <= 1; ++db) {
            int32_t bb = b0 + db;
            if (bb < 0) bb = 0; else if (bb > 255) bb = 255;
            const float sq  = sg53_e4m4_decode_f32((uint8_t)bb);
            const float mse = x2 - 2.0f * sq * dot_xc + sq * sq * dot_cc;
            if (mse < best_mse) {
                best_mse = mse;
                best_hi  = cur_hi;
                best_sc  = (uint8_t)bb;
                for (uint32_t k = 0U; k < 8U; ++k) best_lo[k] = cur_lo[k];
            }
        }
    }

    for (uint32_t k = 0U; k < 8U; ++k) out_lo16[k] = best_lo[k];
    *out_hi2      = best_hi;
    *out_scale_u8 = best_sc;
    if (err_energy_out) *err_energy_out += (best_mse > 0.0f ? (double)best_mse : 0.0);
}

/*
 * Row-subslice helpers for `@glm53-stream` so concatenated Q4 / F32 spine tensors (`lin_qkv_proj`,
 * `lin_qkv_conv1d`, `lin_fbg_a_proj`, `mla_qkv_a_proj`, `dense_gate_up_proj`, `shared_gate_up_proj`)
 * can be `pwrite`n independently in ANY `.safetensors` shard order with zero cross-shard buffering!
 */
SG53_INLINE StrataGlm53QuantMatrix strata_glm53_q4_row_subslice(
    const StrataGlm53QuantMatrix* parent,
    uint32_t                      row_offset,
    uint32_t                      sub_rows
) {
    StrataGlm53QuantMatrix sub;
    memset(&sub, 0, sizeof(sub));
    const uint64_t w_row_bytes = (uint64_t)parent->cols / 2ULL;
    const uint64_t s_row_bytes = (uint64_t)(parent->cols / parent->group_size) * 2ULL;
    sub.weight_off   = parent->weight_off + (uint64_t)row_offset * w_row_bytes;
    sub.weight_bytes = (uint64_t)sub_rows * w_row_bytes;
    sub.scales_off   = parent->scales_off + (uint64_t)row_offset * s_row_bytes;
    sub.scales_bytes = (uint64_t)sub_rows * s_row_bytes;
    sub.biases_off   = parent->biases_off + (uint64_t)row_offset * s_row_bytes;
    sub.biases_bytes = (uint64_t)sub_rows * s_row_bytes;
    sub.rows_logical = sub_rows;
    sub.rows_padded  = sub_rows;
    sub.cols         = parent->cols;
    sub.group_size   = parent->group_size;
    return sub;
}

SG53_INLINE uint64_t strata_glm53_slice_row_offset(
    const StrataGlm53TensorSlice* parent,
    uint32_t                      row_offset
) {
    const uint64_t bpe  = (parent->dtype == STRATA_GLM53_DTYPE_F32 || parent->dtype == STRATA_GLM53_DTYPE_U32)
                              ? 4ULL : 2ULL;
    const uint64_t cols = parent->shape1 > 0U ? (uint64_t)parent->shape1 : 1ULL;
    return parent->offset + (uint64_t)row_offset * cols * bpe;
}

/* ---- 3.2 Deterministic Offset Planner for `spine.strata` & `experts_nvme.strata` ---- */

SG53_INLINE void strata_glm53_default_config(StrataGlm53Config* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->num_hidden_layers        = STRATA_GLM53_CANONICAL_LAYERS;          /* 45 */
    cfg->num_nextn_predict_layers = 1U;                                     /* Layer 45 MTP */
    cfg->first_k_dense_replace    = STRATA_GLM53_CANONICAL_DENSE_LAYERS;    /* 3 */
    cfg->hidden_size              = STRATA_GLM53_CANONICAL_HIDDEN;          /* 4096 */
    cfg->vocab_size               = STRATA_GLM53_CANONICAL_VOCAB;           /* 154880 */
    cfg->intermediate_size        = STRATA_GLM53_CANONICAL_DENSE_INTER;     /* 12288 */
    cfg->num_kda_layers           = 34U;
    cfg->num_dsa_layers           = 11U;

    cfg->mhc_enabled              = 1U;
    cfg->hc_mult                  = 4U;
    cfg->hc_sinkhorn_iters        = 20U;
    cfg->hc_eps                   = 1e-6f;

    cfg->linear_num_heads         = 64U;
    cfg->linear_head_dim          = 128U;
    cfg->linear_conv_kernel_dim   = 4U;
    cfg->linear_gate_lower_bound  = -5.0f;

    cfg->num_attention_heads      = 64U;
    cfg->num_key_value_heads      = 64U;
    cfg->q_lora_rank              = 1536U;
    cfg->kv_lora_rank             = 512U;
    cfg->qk_nope_head_dim         = 256U;
    cfg->qk_rope_head_dim         = 0U;
    cfg->v_head_dim               = 256U;
    cfg->mla_use_nope             = 1U;
    cfg->index_n_heads            = 32U;
    cfg->index_head_dim           = 128U;
    cfg->index_kpool              = 4U;
    cfg->index_topk               = 2048U;
    cfg->index_always_select_tail = 1U;
    cfg->indexer_layernorm_eps    = 1e-6f;

    cfg->n_routed_experts         = STRATA_GLM53_CANONICAL_EXPERTS;         /* 288 */
    cfg->num_experts_per_tok      = STRATA_GLM53_CANONICAL_TOP_K;           /* 8 */
    cfg->n_shared_experts         = 1U;
    cfg->moe_intermediate_size    = STRATA_GLM53_CANONICAL_MOE_INTER;       /* 2048 */
    cfg->n_group                  = 1U;
    cfg->topk_group               = 1U;
    cfg->norm_topk_prob           = 1U;
    cfg->routed_scaling_factor    = 2.5f;
    cfg->swiglu_limit             = 10.0f;

    cfg->rms_norm_eps             = 1e-5f;
    cfg->quant_group_size         = STRATA_GLM53_QUANT_GROUP_SIZE;          /* 64 */
    cfg->quant_bits               = STRATA_GLM53_QUANT_BITS;                /* 4 */
    cfg->tie_word_embeddings      = 0U;
    cfg->max_position_embeddings  = 1048576U;
    cfg->pad_token_id             = STRATA_GLM53_TOKEN_ENDOFTEXT;
    cfg->bos_gmask_token_id       = STRATA_GLM53_TOKEN_GMASK;
    cfg->bos_sop_token_id         = STRATA_GLM53_TOKEN_SOP;
    cfg->eos_token_ids[0]         = STRATA_GLM53_TOKEN_ENDOFTEXT;
    cfg->eos_token_ids[1]         = STRATA_GLM53_TOKEN_USER;
    cfg->eos_token_ids[2]         = STRATA_GLM53_TOKEN_OBSERVATION;
    cfg->num_eos_token_ids        = 3U;

    for (uint32_t l = 0U; l < cfg->num_hidden_layers; ++l) {
        cfg->layer_attn_types[l] = ((l % 4U) == 3U) ? STRATA_GLM53_ATTN_DSA_MLA
                                                    : STRATA_GLM53_ATTN_KDA_LINEAR;
        cfg->layer_mlp_types[l]  = (l < cfg->first_k_dense_replace) ? STRATA_GLM53_MLP_DENSE
                                                                    : STRATA_GLM53_MLP_SPARSE_MOE;
    }
}

/* Deterministic byte offset of routed expert `(moe_layer_idx, expert_idx)` inside `experts_nvme.strata`. */
SG53_INLINE uint64_t strata_glm53_expert_file_offset(
    const StrataGlm53SpineHeader* hdr,
    uint32_t moe_layer_idx,
    uint32_t expert_idx
) {
    return ((uint64_t)moe_layer_idx * (uint64_t)hdr->cfg.n_routed_experts + (uint64_t)expert_idx) *
           hdr->expert_record_bytes;
}

/* Canonical 43-MoE-Layer x 288-Expert fast-path offset (`(moe_l * 288 + e) * 14,155,776`). */
SG53_INLINE uint64_t strata_glm53_canonical_expert_offset(uint32_t moe_layer_idx, uint32_t expert_idx) {
    return ((uint64_t)moe_layer_idx * 288ULL + (uint64_t)expert_idx) * STRATA_GLM53_EXPERT_RECORD_BYTES;
}

SG53_INLINE StrataGlm53TensorSlice sg53_alloc_slice(
    uint64_t* cursor,
    uint32_t  dtype,
    uint32_t  shape0,
    uint32_t  shape1
) {
    StrataGlm53TensorSlice s;
    memset(&s, 0, sizeof(s));
    const uint64_t elems = (uint64_t)shape0 * (uint64_t)(shape1 > 0U ? shape1 : 1U);
    const uint64_t bpe   = (dtype == STRATA_GLM53_DTYPE_F32 || dtype == STRATA_GLM53_DTYPE_U32) ? 4ULL : 2ULL;
    const uint64_t off   = sg53_align_up_u64(*cursor, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t bytes = sg53_align_up_u64(elems * bpe, STRATA_GLM53_TENSOR_ALIGN);
    s.offset     = off;
    s.bytes      = bytes;
    s.elem_count = (uint32_t)elems;
    s.dtype      = dtype;
    s.shape0     = shape0;
    s.shape1     = shape1;
    *cursor      = off + bytes;
    return s;
}

SG53_INLINE StrataGlm53QuantMatrix sg53_alloc_q4(
    uint64_t* cursor,
    uint32_t  rows_logical,
    uint32_t  cols,
    uint32_t  group_size
) {
    StrataGlm53QuantMatrix m;
    memset(&m, 0, sizeof(m));
    const uint32_t rows_padded = (rows_logical + 63U) & ~63U;
    const uint64_t w_bytes = sg53_align_up_u64(
        ((uint64_t)rows_padded * (uint64_t)cols) / 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t s_bytes = sg53_align_up_u64(
        (uint64_t)rows_padded * (uint64_t)(cols / group_size) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    uint64_t off = sg53_align_up_u64(*cursor, STRATA_GLM53_TENSOR_ALIGN);

    m.weight_off   = off;
    m.weight_bytes = w_bytes;
    off += w_bytes;

    m.scales_off   = off;
    m.scales_bytes = s_bytes;
    off += s_bytes;

    m.biases_off   = off;
    m.biases_bytes = s_bytes;
    off += s_bytes;

    m.rows_logical = rows_logical;
    m.rows_padded  = rows_padded;
    m.cols         = cols;
    m.group_size   = group_size;
    *cursor        = off;
    return m;
}

SG53_INLINE void sg53_plan_mhc(
    uint64_t* cursor,
    const StrataGlm53Config* cfg,
    StrataGlm53MhcDesc* mhc
) {
    const uint32_t hc     = cfg->hc_mult;
    const uint32_t mix_sz = (2U + hc) * hc;          /* 24 */
    const uint32_t hc_dim = hc * cfg->hidden_size;   /* 16384 */
    mhc->fn    = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, mix_sz, hc_dim);
    mhc->base  = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, mix_sz, 1U);
    mhc->scale = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, 3U,     1U);
}

SG53_INLINE uint64_t sg53_plan_layer_spine_ex(
    uint64_t* cursor,
    const StrataGlm53Config* cfg,
    uint32_t global_layer_idx,
    uint32_t attn_type,
    uint32_t mlp_type,
    uint32_t has_mhc,
    int32_t  moe_layer_idx,
    bool     drop_dup_q4_kvb,
    StrataGlm53LayerDesc* ld
) {
    memset(ld, 0, sizeof(*ld));
    const uint64_t start_off = sg53_align_up_u64(*cursor, STRATA_GLM53_PAGE_BYTES);
    *cursor = start_off;

    ld->global_layer_idx = global_layer_idx;
    ld->attn_type        = attn_type;
    ld->mlp_type         = mlp_type;
    ld->has_mhc          = has_mhc;
    ld->moe_layer_idx    = moe_layer_idx;
    ld->layer_page_off   = start_off;

    const uint32_t d  = cfg->hidden_size;
    const uint32_t gs = cfg->quant_group_size;

    if (has_mhc) {
        sg53_plan_mhc(cursor, cfg, &ld->hc_attn);
        sg53_plan_mhc(cursor, cfg, &ld->hc_ffn);
    }

    ld->input_layernorm          = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, d, 1U);
    ld->post_attention_layernorm = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, d, 1U);

    uint64_t inactive_dup_bytes = 0ULL;

    if (attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
        const uint32_t nh       = cfg->linear_num_heads;    /* 64    */
        const uint32_t hd       = cfg->linear_head_dim;     /* 128   */
        const uint32_t qkv_dim  = nh * hd;                  /* 8192  */
        const uint32_t conv_dim = 3U * qkv_dim;             /* 24576 */
        const uint32_t fbg_dim  = 2U * hd + nh;             /* 320   */

        ld->lin_qkv_proj   = sg53_alloc_q4(cursor, conv_dim, d,       gs);
        ld->lin_fbg_a_proj = sg53_alloc_q4(cursor, fbg_dim,  d,       gs);
        ld->lin_f_b_proj   = sg53_alloc_q4(cursor, qkv_dim,  hd,      gs);
        ld->lin_g_b_proj   = sg53_alloc_q4(cursor, qkv_dim,  hd,      gs);
        ld->lin_o_proj     = sg53_alloc_q4(cursor, d,        qkv_dim, gs);
        ld->lin_qkv_conv1d = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, conv_dim, cfg->linear_conv_kernel_dim);
        ld->lin_A_log      = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, nh,      1U);
        ld->lin_dt_bias    = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, qkv_dim, 1U);
        ld->lin_o_norm     = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, hd,      1U);
    } else {
        const uint32_t nh      = cfg->num_attention_heads;            /* 64    */
        const uint32_t qr      = cfg->q_lora_rank;                    /* 1536  */
        const uint32_t kvr     = cfg->kv_lora_rank;                   /* 512   */
        const uint32_t q_nope  = cfg->qk_nope_head_dim;               /* 256   */
        const uint32_t v_dim   = cfg->v_head_dim;                     /* 256   */
        const uint32_t q_tot   = nh * q_nope;                         /* 16384 */
        const uint32_t v_tot   = nh * v_dim;                          /* 16384 */
        const uint32_t uk_rows = nh * kvr;                            /* 32768 */
        const uint32_t idx_nh  = cfg->index_n_heads;                  /* 32    */
        const uint32_t idx_hd  = cfg->index_head_dim;                 /* 128   */

        ld->mla_qkv_a_proj     = sg53_alloc_q4(cursor, qr + kvr, d,   gs);
        ld->mla_q_a_layernorm  = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, qr,  1U);
        ld->mla_q_b_proj       = sg53_alloc_q4(cursor, q_tot,    qr,  gs);
        ld->mla_kv_a_layernorm = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, kvr, 1U);

        if (!drop_dup_q4_kvb) {
            const uint64_t q4_kvb_start = *cursor;
            ld->mla_kv_b_k         = sg53_alloc_q4(cursor, q_tot,    kvr, gs);
            ld->mla_kv_b_v         = sg53_alloc_q4(cursor, v_tot,    kvr, gs);
            inactive_dup_bytes    += (*cursor - q4_kvb_start); /* Absorbed decode reads mla_w_uk + mla_w_uv instead */
        }

        ld->mla_w_uk           = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F16, uk_rows, q_nope);
        ld->mla_w_uv           = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F16, v_tot,   kvr);
        ld->mla_o_proj         = sg53_alloc_q4(cursor, d,        v_tot, gs);

        ld->indexer_wq_b          = sg53_alloc_q4(cursor, idx_nh * idx_hd, qr, gs);
        ld->indexer_wk            = sg53_alloc_q4(cursor, idx_hd,          d,  gs);
        ld->indexer_weights_proj  = sg53_alloc_q4(cursor, idx_nh,          d,  gs);
        ld->indexer_k_norm_weight = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, idx_hd, 1U);
        ld->indexer_k_norm_bias   = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, idx_hd, 1U);
        ld->indexer_compress_ape  = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, cfg->index_kpool, idx_hd);
        ld->indexer_compress_gate = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, idx_hd, d);
    }

    if (mlp_type == STRATA_GLM53_MLP_DENSE) {
        ld->dense_gate_up_proj = sg53_alloc_q4(cursor, 2U * cfg->intermediate_size, d, gs);
        ld->dense_down_proj    = sg53_alloc_q4(cursor, d, cfg->intermediate_size,      gs);
    } else {
        const uint32_t d_sh = cfg->moe_intermediate_size * cfg->n_shared_experts; /* 2048 */
        ld->moe_router_gate     = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F16, cfg->n_routed_experts, d);
        ld->moe_e_score_bias    = sg53_alloc_slice(cursor, STRATA_GLM53_DTYPE_F32, cfg->n_routed_experts, 1U);
        ld->shared_gate_up_proj = sg53_alloc_q4(cursor, 2U * d_sh, d,    gs);
        ld->shared_down_proj    = sg53_alloc_q4(cursor, d,         d_sh, gs);
    }

    const uint64_t raw_layer_active = (*cursor - start_off) - inactive_dup_bytes;
    const uint64_t end_off          = sg53_align_up_u64(*cursor, STRATA_GLM53_PAGE_BYTES);
    *cursor              = end_off;
    ld->layer_page_bytes = end_off - start_off;
    return raw_layer_active;
}

SG53_INLINE uint64_t sg53_plan_layer_spine(
    uint64_t* cursor,
    const StrataGlm53Config* cfg,
    uint32_t global_layer_idx,
    uint32_t attn_type,
    uint32_t mlp_type,
    uint32_t has_mhc,
    int32_t  moe_layer_idx,
    StrataGlm53LayerDesc* ld
) {
    return sg53_plan_layer_spine_ex(
        cursor, cfg, global_layer_idx, attn_type, mlp_type, has_mhc, moe_layer_idx, false, ld);
}

/*
 * Deterministically computes EVERY tensor offset in v1 `spine.strata` (`5.3611 GiB`) and exact file sizes for
 * BOTH `spine.strata` and `experts_nvme.strata` (`163.2656 GiB`) purely from `cfg` (100% v1 Regime II)!
 */
SG53_INLINE void strata_glm53_init_layout(
    const StrataGlm53Config* cfg,
    StrataGlm53SpineHeader*  spine_out
) {
    memset(spine_out, 0, sizeof(*spine_out));
    spine_out->magic               = STRATA_GLM53_SPINE_MAGIC_U64;
    spine_out->version             = STRATA_GLM53_FORMAT_VERSION_V1;
    spine_out->header_padded_bytes = STRATA_GLM53_HEADER_PADDED_BYTES;
    spine_out->cfg                 = *cfg;

    /* Per-expert record geometry (`14,155,776 B = 13.5000 MiB = 54 * 256 KiB` for canonical `2048 x 4096`) */
    const uint64_t d     = (uint64_t)cfg->hidden_size;
    const uint64_t d_moe = (uint64_t)cfg->moe_intermediate_size;
    const uint64_t gs    = (uint64_t)cfg->quant_group_size;
    const uint64_t w_b   = sg53_align_up_u64((d_moe * d) / 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t s_b   = sg53_align_up_u64((d_moe * (d / gs)) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t prj_b = w_b + 2ULL * s_b;
    const uint64_t rec_b = sg53_align_up_u64(3ULL * prj_b, STRATA_GLM53_NVME_STRIPE_BYTES);

    spine_out->expert_weight_bytes = w_b;
    spine_out->expert_scales_bytes = s_b;
    spine_out->expert_proj_bytes   = prj_b;
    spine_out->expert_record_bytes = rec_b;

    uint64_t cur = STRATA_GLM53_HEADER_PADDED_BYTES;

    /* 1. Global Embedding (`embed_tokens`) */
    spine_out->embed_page_off   = cur;
    spine_out->embed_tokens     = sg53_alloc_q4(&cur, cfg->vocab_size, cfg->hidden_size, cfg->quant_group_size);
    cur                         = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->embed_page_bytes = cur - spine_out->embed_page_off;

    uint64_t active_spine = (d / 2ULL) + (d / gs) * 4ULL; /* 1 token row lookup */

    /* 2. Main Hybrid Decoder Layers (`0 .. num_hidden_layers - 1`) */
    uint32_t n_kda     = 0U;
    uint32_t n_dsa     = 0U;
    uint32_t moe_count = 0U;
    for (uint32_t l = 0U; l < cfg->num_hidden_layers && l < STRATA_GLM53_MAX_LAYERS; ++l) {
        const uint32_t atype = cfg->layer_attn_types[l];
        const uint32_t mtype = cfg->layer_mlp_types[l];
        if (atype == STRATA_GLM53_ATTN_KDA_LINEAR) ++n_kda; else ++n_dsa;
        const int32_t moe_idx = (mtype == STRATA_GLM53_MLP_SPARSE_MOE) ? (int32_t)(moe_count++) : -1;
        active_spine += sg53_plan_layer_spine_ex(
            &cur, cfg, l, atype, mtype, cfg->mhc_enabled, moe_idx, false, &spine_out->layers[l]
        );
    }
    spine_out->cfg.num_kda_layers  = n_kda;
    spine_out->cfg.num_dsa_layers  = n_dsa;
    spine_out->num_main_moe_layers = moe_count;

    /* 3. Final RMSNorm + Untied LM Head */
    spine_out->head_page_off = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    cur = spine_out->head_page_off;
    spine_out->final_norm = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
    spine_out->lm_head    = sg53_alloc_q4(&cur, cfg->vocab_size, cfg->hidden_size, cfg->quant_group_size);
    const uint64_t head_raw_end = cur;
    cur = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->head_page_bytes = cur - spine_out->head_page_off;
    active_spine += (head_raw_end - spine_out->head_page_off);

    /* 4. Native 1-Layer Multi-Token Prediction (`layers.45`, single-stream DSA MLA + 288E MoE) */
    uint64_t mtp_active = 0ULL;
    if (cfg->num_nextn_predict_layers > 0U) {
        const int32_t mtp_moe_idx       = (int32_t)(moe_count++);
        spine_out->mtp.enabled          = 1U;
        spine_out->mtp.global_layer_idx = cfg->num_hidden_layers;
        spine_out->mtp.moe_layer_idx    = mtp_moe_idx;
        spine_out->mtp.mtp_page_off     = cur;

        const uint64_t eh_start = cur;
        spine_out->mtp.enorm   = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        spine_out->mtp.hnorm   = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        spine_out->mtp.eh_proj = sg53_alloc_q4(&cur, cfg->hidden_size, 2U * cfg->hidden_size, cfg->quant_group_size);
        mtp_active += (cur - eh_start);

        mtp_active += sg53_plan_layer_spine_ex(
            &cur, cfg, cfg->num_hidden_layers,
            STRATA_GLM53_ATTN_DSA_MLA, STRATA_GLM53_MLP_SPARSE_MOE,
            0U /* has_mhc == 0 for single-stream MTP! */, mtp_moe_idx, false, &spine_out->mtp.layer
        );

        const uint64_t sn_start = cur;
        spine_out->mtp.shared_head_norm = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        mtp_active += (cur - sn_start) +
                      spine_out->lm_head.weight_bytes + spine_out->lm_head.scales_bytes + spine_out->lm_head.biases_bytes +
                      (uint64_t)cfg->num_experts_per_tok * rec_b;

        cur = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
        spine_out->mtp.mtp_page_bytes = cur - spine_out->mtp.mtp_page_off;
    }

    spine_out->num_total_moe_layers       = moe_count;
    spine_out->total_routed_experts       = moe_count * cfg->n_routed_experts;
    spine_out->experts_nvme_bytes         = (uint64_t)spine_out->total_routed_experts * rec_b;
    spine_out->total_spine_bytes          = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->active_spine_bytes_per_tok = active_spine;
    spine_out->active_routed_bytes_per_tok= (uint64_t)spine_out->num_main_moe_layers *
                                            (uint64_t)cfg->num_experts_per_tok * rec_b;
    spine_out->active_bytes_per_tok       = spine_out->active_spine_bytes_per_tok +
                                            spine_out->active_routed_bytes_per_tok;
    spine_out->mtp_active_bytes_per_draft = mtp_active;
}

/*
 * Deterministically computes EVERY offset & file size for a v2 100% RAM-Resident `RHT-64 + E8S-18-V4`
 * bundle (`93.22 GiB` total UMA: `spine.strata` 5.256 GiB + `experts_uma_part0.strata` 42.082 GiB +
 * `experts_uma_part1.strata` 42.082 GiB + `mtp_experts.strata` 3.797 GiB; ZERO NVMe expert file!).
 */
SG53_INLINE void strata_glm53_init_layout_v2(
    const StrataGlm53Config* cfg,
    StrataGlm53SpineHeader*  spine_out
) {
    memset(spine_out, 0, sizeof(*spine_out));
    spine_out->magic               = STRATA_GLM53_SPINE_MAGIC_U64;
    spine_out->version             = STRATA_GLM53_FORMAT_VERSION_V2;
    spine_out->header_padded_bytes = STRATA_GLM53_HEADER_PADDED_BYTES;
    spine_out->cfg                 = *cfg;

    /* Q4_A per-expert geometry (used by Layer-45 MTP `mtp_experts.strata`: `14,155,776 B`) */
    const uint64_t d     = (uint64_t)cfg->hidden_size;
    const uint64_t d_moe = (uint64_t)cfg->moe_intermediate_size;
    const uint64_t gs    = (uint64_t)cfg->quant_group_size;
    const uint64_t w_b   = sg53_align_up_u64((d_moe * d) / 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t s_b   = sg53_align_up_u64((d_moe * (d / gs)) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t prj_b = w_b + 2ULL * s_b;
    const uint64_t rec_b = sg53_align_up_u64(3ULL * prj_b, STRATA_GLM53_NVME_STRIPE_BYTES);

    spine_out->expert_weight_bytes = w_b;
    spine_out->expert_scales_bytes = s_b;
    spine_out->expert_proj_bytes   = prj_b;
    spine_out->expert_record_bytes = rec_b;

    /* E8S-18-V4 per-expert geometry (`7,471,104 B = 7.125 MiB = 456 * 16 KiB` for canonical `2048 x 4096`) */
    const uint64_t n_w        = d_moe * d;
    const uint64_t e8_lo16_b  = sg53_align_up_u64((n_w / 8ULL) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t e8_hi2_b   = sg53_align_up_u64((n_w / 64ULL) * 2ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t e8_scale_b = sg53_align_up_u64((n_w / 64ULL) * 1ULL, STRATA_GLM53_TENSOR_ALIGN);
    const uint64_t e8_proj_b  = e8_lo16_b + e8_hi2_b + e8_scale_b;
    const uint64_t e8_rec_b   = sg53_align_up_u64(3ULL * e8_proj_b, STRATA_GLM53_PAGE_BYTES);

    uint64_t cur = STRATA_GLM53_HEADER_PADDED_BYTES;

    /* 1. Global Embedding (`embed_tokens`) */
    spine_out->embed_page_off   = cur;
    spine_out->embed_tokens     = sg53_alloc_q4(&cur, cfg->vocab_size, cfg->hidden_size, cfg->quant_group_size);
    cur                         = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->embed_page_bytes = cur - spine_out->embed_page_off;

    uint64_t active_spine = (d / 2ULL) + (d / gs) * 4ULL;

    /* 2. Main Hybrid Decoder Layers (`0 .. num_hidden_layers - 1`, dropping unused dup Q4 `mla_kv_b_k/v`) */
    uint32_t n_kda     = 0U;
    uint32_t n_dsa     = 0U;
    uint32_t moe_count = 0U;
    for (uint32_t l = 0U; l < cfg->num_hidden_layers && l < STRATA_GLM53_MAX_LAYERS; ++l) {
        const uint32_t atype = cfg->layer_attn_types[l];
        const uint32_t mtype = cfg->layer_mlp_types[l];
        if (atype == STRATA_GLM53_ATTN_KDA_LINEAR) ++n_kda; else ++n_dsa;
        const int32_t moe_idx = (mtype == STRATA_GLM53_MLP_SPARSE_MOE) ? (int32_t)(moe_count++) : -1;
        active_spine += sg53_plan_layer_spine_ex(
            &cur, cfg, l, atype, mtype, cfg->mhc_enabled, moe_idx, true, &spine_out->layers[l]
        );
    }
    spine_out->cfg.num_kda_layers  = n_kda;
    spine_out->cfg.num_dsa_layers  = n_dsa;
    spine_out->num_main_moe_layers = moe_count;

    /* 3. Final RMSNorm + Untied LM Head */
    spine_out->head_page_off = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    cur = spine_out->head_page_off;
    spine_out->final_norm = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
    spine_out->lm_head    = sg53_alloc_q4(&cur, cfg->vocab_size, cfg->hidden_size, cfg->quant_group_size);
    const uint64_t head_raw_end = cur;
    cur = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->head_page_bytes = cur - spine_out->head_page_off;
    active_spine += (head_raw_end - spine_out->head_page_off);

    /* 4. Native 1-Layer Multi-Token Prediction (`layers.45`, single-stream DSA MLA + 288E Q4_A MoE) */
    uint64_t mtp_active = 0ULL;
    if (cfg->num_nextn_predict_layers > 0U) {
        const int32_t mtp_moe_idx       = (int32_t)(moe_count++);
        spine_out->mtp.enabled          = 1U;
        spine_out->mtp.global_layer_idx = cfg->num_hidden_layers;
        spine_out->mtp.moe_layer_idx    = mtp_moe_idx;
        spine_out->mtp.mtp_page_off     = cur;

        const uint64_t eh_start = cur;
        spine_out->mtp.enorm   = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        spine_out->mtp.hnorm   = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        spine_out->mtp.eh_proj = sg53_alloc_q4(&cur, cfg->hidden_size, 2U * cfg->hidden_size, cfg->quant_group_size);
        mtp_active += (cur - eh_start);

        mtp_active += sg53_plan_layer_spine_ex(
            &cur, cfg, cfg->num_hidden_layers,
            STRATA_GLM53_ATTN_DSA_MLA, STRATA_GLM53_MLP_SPARSE_MOE,
            0U, mtp_moe_idx, true, &spine_out->mtp.layer
        );

        const uint64_t sn_start = cur;
        spine_out->mtp.shared_head_norm = sg53_alloc_slice(&cur, STRATA_GLM53_DTYPE_F32, cfg->hidden_size, 1U);
        mtp_active += (cur - sn_start) +
                      spine_out->lm_head.weight_bytes + spine_out->lm_head.scales_bytes + spine_out->lm_head.biases_bytes +
                      (uint64_t)cfg->num_experts_per_tok * rec_b;

        cur = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
        spine_out->mtp.mtp_page_bytes = cur - spine_out->mtp.mtp_page_off;
    }

    const uint32_t main_moe = spine_out->num_main_moe_layers;
    const uint32_t p0_layers = (main_moe + 1U) / 2U; /* 42 -> 21; 1 -> 1 */
    const uint32_t p1_layers = main_moe - p0_layers; /* 42 -> 21; 1 -> 0 */

    spine_out->num_total_moe_layers        = moe_count;
    spine_out->total_routed_experts        = moe_count * cfg->n_routed_experts;
    spine_out->experts_nvme_bytes          = 0ULL; /* ZERO NVMe expert offload in v2! */
    spine_out->total_spine_bytes           = sg53_align_up_u64(cur, STRATA_GLM53_PAGE_BYTES);
    spine_out->active_spine_bytes_per_tok  = active_spine;
    spine_out->active_routed_bytes_per_tok = (uint64_t)main_moe * (uint64_t)cfg->num_experts_per_tok * e8_rec_b;
    spine_out->active_bytes_per_tok        = spine_out->active_spine_bytes_per_tok +
                                             spine_out->active_routed_bytes_per_tok;
    spine_out->mtp_active_bytes_per_draft  = mtp_active;

    /* Populate v2 RAM-Resident `E8S-18-V4` header fields */
    spine_out->is_ram_resident   = 1U;
    spine_out->main_expert_dtype = STRATA_GLM53_EXPERT_DTYPE_E8S18_V4;
    spine_out->mtp_expert_dtype  = STRATA_GLM53_EXPERT_DTYPE_Q4_A;
    spine_out->part0_moe_layers  = p0_layers;
    spine_out->part1_moe_layers  = p1_layers;
    spine_out->part0_bytes       = sg53_align_up_u64(
        (uint64_t)p0_layers * (uint64_t)cfg->n_routed_experts * e8_rec_b, STRATA_GLM53_PAGE_BYTES);
    spine_out->part1_bytes       = sg53_align_up_u64(
        (uint64_t)p1_layers * (uint64_t)cfg->n_routed_experts * e8_rec_b, STRATA_GLM53_PAGE_BYTES);
    spine_out->mtp_experts_bytes = (cfg->num_nextn_predict_layers > 0U)
        ? sg53_align_up_u64((uint64_t)cfg->n_routed_experts * rec_b, STRATA_GLM53_PAGE_BYTES)
        : 0ULL;
    spine_out->e8_record_bytes   = e8_rec_b;
    spine_out->e8_proj_bytes     = e8_proj_b;
    spine_out->e8_lo16_bytes     = e8_lo16_b;
    spine_out->e8_hi2_bytes      = e8_hi2_b;
    spine_out->e8_scale_bytes    = e8_scale_b;
    spine_out->rht_seed          = STRATA_GLM53_RHT64_SEED;
    for (uint32_t ml = 0U; ml < main_moe && ml < STRATA_GLM53_MAX_LAYERS; ++ml) {
        spine_out->moe_layer_quant[ml] = STRATA_GLM53_EXPERT_DTYPE_E8S18_V4;
    }
    if (cfg->num_nextn_predict_layers > 0U && main_moe < STRATA_GLM53_MAX_LAYERS) {
        spine_out->moe_layer_quant[main_moe] = STRATA_GLM53_EXPERT_DTYPE_Q4_A;
    }
}

/*
 * Deterministic byte offset of a main MoE expert `(moe_layer_idx, expert_idx)` inside v2
 * `experts_uma_part0.strata` (`*part_idx_out = 0`) or `experts_uma_part1.strata` (`*part_idx_out = 1`).
 */
SG53_INLINE uint64_t strata_glm53_v2_expert_offset(
    const StrataGlm53SpineHeader* hdr,
    uint32_t                      moe_layer_idx,
    uint32_t                      expert_idx,
    uint32_t*                     part_idx_out
) {
    const uint32_t p0 = hdr->part0_moe_layers;
    if (moe_layer_idx < p0) {
        if (part_idx_out) *part_idx_out = 0U;
        return ((uint64_t)moe_layer_idx * (uint64_t)hdr->cfg.n_routed_experts + (uint64_t)expert_idx) *
               hdr->e8_record_bytes;
    }
    if (part_idx_out) *part_idx_out = 1U;
    return ((uint64_t)(moe_layer_idx - p0) * (uint64_t)hdr->cfg.n_routed_experts + (uint64_t)expert_idx) *
           hdr->e8_record_bytes;
}

/* ============================================================================
 * 4. Host Bundle Handle, Streaming HF Converter & BPE Tokenizer C ABI
 *    (Implemented by `@glm53-stream` in `src/model/strata_glm53_convert.cpp`)
 * ============================================================================ */

typedef struct StrataGlm53Tokenizer StrataGlm53Tokenizer;

typedef struct StrataGlm53Bundle {
    char                           bundle_dir[512];
    char                           spine_path[512];
    char                           experts_nvme_path[512];

    /* Zero-copy `16 KiB` page-aligned read-only `mmap` of `spine.strata` (`~5.36 GiB` v1 / `~5.26 GiB` v2) */
    int                            spine_fd;
    const uint8_t*                 spine_base;
    size_t                         spine_bytes;            /* Exact multiple of 16384 */
    const StrataGlm53SpineHeader*  header;                 /* Points directly to `spine_base` */

    /* v1 Regime II: `N = 8` parallel `O_RDONLY` + `fcntl(F_NOCACHE, 1)` FDs to `experts_nvme.strata` (`163.27 GiB`) */
    int                            experts_fds[STRATA_GLM53_NUM_NVME_FDS];
    uint32_t                       num_experts_fds;        /* 8 on v1 disk; 0 for v2 RAM-resident or synthetic */
    uint64_t                       experts_nvme_bytes;     /* 175.31 GB on v1; 0 on v2 */
    const uint8_t*                 synthetic_experts_base; /* Optional RAM pointer ONLY when v1 & num_experts_fds == 0 */

    StrataGlm53Config              cfg;                    /* Value copy of `header->cfg` */
    StrataGlm53Tokenizer*          tokenizer;              /* Loaded from `<bundle_dir>/tokenizer.json` */

    /* v2 RAM-Resident (`header->version == 2 && header->is_ram_resident == 1`):
     * Zero-copy `16 KiB`-aligned read-only `mmap`s of `part0`, `part1`, and `mtp_experts`! */
    char                           experts_part0_path[512];
    char                           experts_part1_path[512];
    char                           mtp_experts_path[512];
    int                            part0_fd;
    const uint8_t*                 part0_base;
    size_t                         part0_bytes;
    int                            part1_fd;
    const uint8_t*                 part1_base;
    size_t                         part1_bytes;
    int                            mtp_experts_fd;
    const uint8_t*                 mtp_experts_base;
    size_t                         mtp_experts_bytes;
    bool                           is_ram_resident;
} StrataGlm53Bundle;

SG53_INLINE const void* strata_glm53_spine_ptr(const StrataGlm53Bundle* b, uint64_t byte_offset) {
    return (const void*)(b->spine_base + byte_offset);
}

SG53_INLINE bool strata_glm53_is_eos_token(const StrataGlm53Config* cfg, uint32_t token_id) {
    for (uint32_t i = 0U; i < cfg->num_eos_token_ids; ++i) {
        if (token_id == cfg->eos_token_ids[i]) return true;
    }
    return false;
}

typedef struct StrataGlm53ConvertStats {
    uint32_t num_shards_total;             /* 43 (v1) or 62 (v2) */
    uint32_t num_shards_converted;
    uint32_t num_shards_resumed;
    uint32_t num_text_tensors_converted;   /* 111172 (v1) or 36663 (v2) */
    uint32_t num_mtp_tensors_converted;    /* 2641 (v1) or 1760 (v2) */
    uint32_t num_routed_experts_packed;    /* 12384  (43 * 288) */
    uint32_t num_vision_tensors_skipped;   /* 347    (model.visual.*) */
    uint64_t hf_safetensors_bytes;         /* 181.69 GB (v1) or 328.33 GB (v2) */
    uint64_t spine_bytes;                  /* 5.361 GiB (v1) or 5.256 GiB (v2), % 16384 == 0 */
    uint64_t experts_nvme_bytes;           /* 163.27 GiB (v1) or 0 (v2) */
    uint64_t total_bundle_bytes;           /* ~168.63 GiB (v1) or ~93.22 GiB (v2) */
    uint64_t active_bytes_per_tok;         /* ~9.91 GB (v1) or ~7.66 GB (v2) */
    double   elapsed_sec;
    double   network_mbps;                 /* MB/s across downloaded shards */
    double   peak_rss_gib;                 /* Target < 1.0 GiB */
    bool     page_aligned_verified;        /* Exact 16 KiB alignment across all output files */
    bool     staging_clean_verified;       /* `.staging/` / `.tmp_shards/` removed (0 files, 0 B) */
    bool     nan_inf_checked;              /* Zero NaN/Inf across all norms, mHC, MLA & routers */
    /* v2 RAM-Resident (`RHT-64 + E8S-18-V4`) stats */
    uint64_t part0_bytes;                  /* 45,185,236,992 B (42.082 GiB) */
    uint64_t part1_bytes;                  /* 45,185,236,992 B (42.082 GiB) */
    uint64_t mtp_experts_bytes;            /*  4,076,863,488 B ( 3.797 GiB) */
    double   avg_e8_snr_db;                /* Measured mean SNR (dB) across converted E8S-18-V4 experts */
    double   avg_expert_quant_ms;          /* Mean wall ms per E8S-18-V4 expert quantization */
} StrataGlm53ConvertStats;

/*
 * Pipelined Shard-by-Shard Downloader + ARM NEON `BF16 -> F16/F32` Packager + Immediate Shard Deleter (v1):
 *   - `hf_repo_or_dir`: HF repo ID (`"TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"`), base URL, OR local directory.
 *     If NULL or empty, defaults to `STRATA_GLM53_DEFAULT_HF_REPO`.
 *   - `out_bundle_dir`: Destination bundle directory (defaults to `STRATA_GLM53_DEFAULT_BUNDLE_DIR` if NULL).
 *   - `download_concurrency`: Background shard prefetch concurrency (`0` => default `1` ahead so `.staging < 8 GiB`).
 *   - `convert_threads`: Parallel NEON conversion threads (`0` => default hardware core count).
 *   - Crash-safe resumable via `<out_bundle_dir>/.convert_done_shards` and removes `<out_bundle_dir>/.staging` on finish.
 */
int strata_glm53_convert_hf(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    uint32_t    download_concurrency,
    uint32_t    convert_threads,
    StrataGlm53ConvertStats* stats_out
);

/* Convenience wrapper calling `strata_glm53_convert_hf(hf_repo_or_dir, out_bundle_dir, 0, 0, stats_out)`. */
int strata_glm53_convert_hf_directory(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    StrataGlm53ConvertStats* stats_out
);

/*
 * Streaming `zai-org/GLM-5.3-Flash` FP8/BF16 -> v2 100% RAM-Resident `RHT-64 + E8S-18-V4` Converter (`93.22 GiB` UMA):
 *   - `hf_repo_or_dir`: HF repo ID (`"zai-org/GLM-5.3-Flash"`), base URL, OR local directory (NULL => default repo).
 *   - `out_bundle_dir`: Destination directory (NULL => `STRATA_GLM53_DEFAULT_FAST_BUNDLE_DIR`).
 *   - `num_streams`: HTTP parallel range streams per shard (`0` => default `4`).
 *   - `ols_trials`: OLS scale search trials per 64w group (`0` => default `3`; valid `1..3`).
 *   - `keep_temp_shards`: `0` => unlink each downloaded shard immediately after `pwrite` (peak temp SSD <= 20 GiB).
 *   - `hf_token`: optional HF Bearer token (NULL/empty => falls back to `HF_TOKEN` env var).
 *   - 100% crash-safe resumable via `<out_bundle_dir>/.convert_done_shards`.
 */
int strata_glm53_convert_fast(
    const char*              hf_repo_or_dir,
    const char*              out_bundle_dir,
    uint32_t                 num_streams,
    uint32_t                 ols_trials,
    uint32_t                 keep_temp_shards,
    const char*              hf_token,
    StrataGlm53ConvertStats* stats_out
);

/*
 * Opens BOTH v1 Regime II (`spine.strata` mmap + 8 F_NOCACHE FDs to `experts_nvme.strata`) AND
 * v2 RAM-Resident (`spine.strata` + `experts_uma_part0.strata` + `experts_uma_part1.strata` +
 * `mtp_experts.strata` zero-copy `16 KiB`-aligned read-only `mmap`s, ZERO NVMe FDs), plus `tokenizer.json`.
 */
int strata_glm53_load_bundle(
    const char* bundle_dir,
    StrataGlm53Bundle* bundle_out
);

void strata_glm53_free_bundle(StrataGlm53Bundle* bundle);

/* Convenience aliases for `strata_glm53_load_bundle` / `strata_glm53_free_bundle`. */
SG53_INLINE int strata_glm53_bundle_open(const char* bundle_dir, StrataGlm53Bundle* bundle_out) {
    return strata_glm53_load_bundle(bundle_dir, bundle_out);
}
SG53_INLINE void strata_glm53_bundle_close(StrataGlm53Bundle* bundle) {
    strata_glm53_free_bundle(bundle);
}

/* ---- 4.1 Byte-Level BPE Tokenizer (`154,880` vocab) & Official GLM-5.3 Chat Formatter ---- */

StrataGlm53Tokenizer* strata_glm53_tokenizer_load(const char* tokenizer_json_path);
void strata_glm53_tokenizer_free(StrataGlm53Tokenizer* tok);
uint32_t strata_glm53_tokenizer_vocab_size(const StrataGlm53Tokenizer* tok);

/*
 * Encodes UTF-8 `text` into `out_ids`. When `parse_special == true`, ALL 36 `added_tokens`
 * (`154820..154855`, including `[gMASK]`, `<sop>`, `<|system|>`, `<|user|>`, `<|assistant|>`,
 * `<|observation|>`, `<think>`, `</think>`, `<|endoftext|>`) MUST be matched atomically!
 */
int strata_glm53_encode(
    const StrataGlm53Tokenizer* tok,
    const char* text,
    size_t      text_len,
    bool        parse_special,
    uint32_t*   out_ids,
    uint32_t    max_ids,
    uint32_t*   out_num_ids
);

/*
 * Formats Turn 1 `user_prompt` (and optional `system_prompt`, NULL/empty for none) per
 * `GLM-5.3-Flash`'s official `chat_template.jinja` (`reasoning_effort` in `{"low", "high", "max"}`,
 * case-insensitive; NULL defaults to `"low"` for concise interactive responses):
 *   - `enable_thinking == false`:
 *     `[gMASK]<sop><|system|>Reasoning Effort: {Effort}[<|system|>{system_prompt}]<|user|>{user_prompt}<|assistant|><think></think>`
 *   - `enable_thinking == true`:
 *     `[gMASK]<sop><|system|>Reasoning Effort: {Effort}[<|system|>{system_prompt}]<|user|>{user_prompt}<|assistant|><think>`
 */
int strata_glm53_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool        enable_thinking,
    const char* reasoning_effort,
    char*       out_buf,
    size_t      out_buf_cap,
    size_t*     out_len
);

/*
 * Standardized Turn 2+ continuation formatter for stateful multi-turn `chat-glm53`:
 * Appended directly onto the live engine state after Turn `t-1` stops on EOS (`154820`/`154827`/`154829`),
 * producing 100% bit-identical token sequences to a full multi-turn `chat_template.jinja` render!
 */
SG53_INLINE int strata_glm53_format_chat_turn_continuation(
    const char* user_prompt,
    bool        enable_thinking,
    char*       out_buf,
    size_t      out_buf_cap,
    size_t*     out_len
) {
    if (!user_prompt || !out_buf || out_buf_cap == 0U) return -1;
    const char* suffix = enable_thinking ? "<|assistant|><think>" : "<|assistant|><think></think>";
    const int n = snprintf(out_buf, out_buf_cap, "<|user|>%s%s", user_prompt, suffix);
    if (n < 0 || (size_t)n >= out_buf_cap) return -1;
    if (out_len) *out_len = (size_t)n;
    return 0;
}

int strata_glm53_decode(
    const StrataGlm53Tokenizer* tok,
    const uint32_t* ids,
    uint32_t        num_ids,
    bool            skip_special,
    char*           out_utf8,
    size_t          out_cap,
    size_t*         out_len
);

typedef struct StrataGlm53StreamDecoder {
    uint8_t pending_bytes[8];
    uint8_t pending_len;
} StrataGlm53StreamDecoder;

SG53_INLINE void strata_glm53_stream_decoder_init(StrataGlm53StreamDecoder* dec) {
    for (int i = 0; i < 8; ++i) dec->pending_bytes[i] = 0;
    dec->pending_len = 0;
}

int strata_glm53_stream_decode_step(
    const StrataGlm53Tokenizer* tok,
    StrataGlm53StreamDecoder*   dec,
    uint32_t                    token_id,
    bool                        skip_special,
    char*                       out_utf8,
    size_t                      out_cap,
    size_t*                     out_len
);

/* ============================================================================
 * 5. Regime II Metal 4 (`NAX`) + 16-Core `ANE` (`MLComputePlan`) Router Lookahead
 *    + `56 GiB` UMA LFU Hot-Expert Cache + `F_NOCACHE` NVMe `Ring_DMA` + Native `MTP` ABI
 *    (Implemented by `@glm53-gpu` in `src/gpu/strata_glm53_gpu.mm` & `strata_glm53_kernels.metal`)
 * ============================================================================ */

typedef struct StrataGlm53GpuEngine StrataGlm53GpuEngine;

typedef enum StrataGlm53ExecMode {
    /*
     * 100% Exact `top-8` Experts (`K=0` & `MTP K>0` bit-exact ground truth):
     * Evaluates ALL `8/8` top-8 routed experts per MoE layer/token across `b_hot_experts` (UMA hits)
     * and `b_ring_dma` (parallel `F_NOCACHE` NVMe cold fetches overlapped with GPU & ANE lookahead).
     */
    STRATA_GLM53_MODE_LOSSLESS = 0,

    /*
     * Importance-Gated Adaptive Regime II Decode (`Mode B`, blazing interactive throughput):
     * Evaluates `100%` of UMA-resident `top-8` hot hits + cold NVMe misses whose scaled router
     * weight `w[i] >= gate_tau` (up to `cold_cap` cold experts/layer), optionally renormalizing
     * selected routed weights to `routed_scaling_factor = 2.5f` while always evaluating `100%`
     * of dense layers `0..2` + `shared_experts` + dominant experts!
     */
    STRATA_GLM53_MODE_ADAPTIVE = 1
} StrataGlm53ExecMode;

typedef struct StrataGlm53SamplingParams {
    bool     greedy;             /* true => deterministic GPU argmax (`temperature` ignored) */
    float    temperature;        /* e.g. 0.6f when `!greedy` */
    uint32_t top_k;              /* e.g. 20 (0 => disabled / default) */
    float    top_p;              /* e.g. 0.95f (<= 0 or >= 1 => disabled) */
    uint64_t rng_seed;           /* PRNG seed/state for Leviathan stochastic verification */
    float    presence_penalty;   /* OpenAI presence penalty [-2..2]; 0 => disabled */
    float    frequency_penalty;  /* OpenAI frequency penalty [-2..2]; 0 => disabled */
    float    repetition_penalty; /* HF multiplicative penalty (> 0; 0 or 1.0 => disabled) */
    uint32_t penalty_window;     /* 0 => full completion; >0 => sliding last max(N, 8) completion tokens */
} StrataGlm53SamplingParams;

typedef struct StrataGlm53GpuInfo {
    char                device_name[128];
    bool                zero_copy_spine;
    bool                ane_verified_physical;       /* `MLComputePlan` confirmed physical `MLNeuralEngineComputeDevice` */
    uint32_t            ane_lookahead_horizon;       /* Lookahead layers ahead (`l+1 .. l+H`, e.g. `H = 4`) */
    uint32_t            num_nvme_fds;                /* Parallel `F_NOCACHE` NVMe FDs (`8` on v1, `0` on v2) */
    StrataGlm53ExecMode exec_mode;
    float               adaptive_gate_tau;
    uint32_t            adaptive_cold_cap;
    double              load_time_ms;
    uint32_t            max_seq_len;

    /* Disaggregated Regime II (v1) & RAM-Resident E8S-18-V4 (v2) Memory & Residency Accounting */
    uint64_t            spine_mapped_bytes;          /* ~5.36 GiB (v1) / ~5.26 GiB (v2) zero-copy `newBufferWithBytesNoCopy:` */
    uint64_t            hot_cache_bytes;             /* v1 UMA hot-expert cache (`0` on v2) */
    uint32_t            hot_cache_capacity_experts;  /* v1 hot slots (`0` on v2) */
    uint32_t            hot_cache_resident_experts;  /* v1 populated slots (`0` on v2) */
    uint32_t            total_routed_experts;        /* `12,384` (`43 * 288`) */
    uint32_t            has_mtp;                     /* `1` when Layer-45 Native MTP enabled */
    uint64_t            ring_dma_bytes;              /* v1 `Ring_DMA` cold staging buffer (`0` on v2) */
    uint32_t            ring_dma_slots;              /* v1 `Ring_DMA` slots (`0` on v2) */
    uint32_t            reserved_pad1;
    uint64_t            experts_nvme_file_bytes;     /* `163.27 GiB` on v1 (`0` on v2!) */
    uint64_t            total_uma_wired_bytes;       /* Total GPU-mapped UMA (`spine + part0 + part1 + mtp + states` on v2) */
    uint64_t            active_spine_bytes_per_token;/* Always-active main spine bytes/token */
    uint64_t            active_bytes_per_token;      /* ~9.91 GB (v1) / ~7.66 GB (v2) active per K=0 token */
    uint64_t            mtp_active_bytes_per_draft;  /* ~0.60 GiB per 1L MTP draft step */
    uint64_t            kda_state_bytes;             /* 34-layer conv + FP32 `[64, 128, 128]` (~141 MiB) */
    uint64_t            kv_cache_bytes_allocated;    /* 12-layer FP16 `[512]` latent MLA + 4-tok pool indexer */
    uint64_t            kv_cache_bytes_per_token;    /* 11 main DSA layers: `11 * 1024 B = 11,264 B` + pool */
    uint64_t            workspace_bytes;             /* Pre-allocated scratch + speculative rollback arenas */

    /* v2 RAM-Resident `RHT-64 + E8S-18-V4` UMA Breakdown (`0` on v1) */
    uint32_t            is_ram_resident;             /* 1 for v2 RAM-resident, 0 for v1 Regime II */
    uint32_t            main_expert_dtype;           /* STRATA_GLM53_EXPERT_DTYPE_E8S18_V4 (1) or Q4_A (0) */
    uint64_t            uma_part0_mapped_bytes;      /* 45,185,236,992 B (~42.08 GiB) zero-copy `part0` */
    uint64_t            uma_part1_mapped_bytes;      /* 45,185,236,992 B (~42.08 GiB) zero-copy `part1` */
    uint64_t            mtp_experts_mapped_bytes;    /*  4,076,863,488 B (~3.80 GiB) zero-copy `mtp_experts` */
    uint32_t            int8_prefill_enabled;        /* 1 iff opt-in INT8 NAX prefill GEMMs enabled */
    uint32_t            adaptive_k_enabled;          /* 1 iff 3-state SpecCtl Adaptive-K enabled */
    uint32_t            frspec_enabled;              /* 1 iff 20K SLC-pinned FR-Spec draft shortlist active */
    uint32_t            frspec_shortlist_size;       /* Active FR-Spec shortlist vocab size (e.g. 20480) */
    uint32_t            metal_cache_hit;             /* 1 iff MTLBinaryArchive warm cache hit at startup */
    double              pso_load_ms;                 /* Wall-clock ms spent loading/compiling Metal PSOs */
} StrataGlm53GpuInfo;

typedef struct StrataGlm53PerfCounters {
    uint64_t prefill_tokens;
    uint64_t decode_steps;
    uint64_t spec_verify_steps;
    uint64_t mtp_drafted_tokens;
    uint64_t mtp_accepted_tokens;

    /* UMA Hot-Expert Cache & `F_NOCACHE` NVMe `Ring_DMA` Telemetry */
    uint64_t uma_hot_hits;              /* Routed expert selections served at 0 NVMe I/O from `b_hot_experts` */
    uint64_t nvme_cold_reads;           /* Cold `13.50 MiB` expert `pread`s streamed from `experts_nvme.strata` */
    uint64_t nvme_bytes_read;           /* Total cold bytes transferred over ANS Gen5 NVMe (`nvme_cold_reads * 13.50 MiB`) */
    double   nvme_io_time_sec;          /* Cumulative wall-time spent in multi-FD `F_NOCACHE` NVMe DMA batches */
    double   nvme_bandwidth_gbps;       /* Measured physical NVMe DMA throughput (`(nvme_bytes_read / 1e9) / nvme_io_time_sec`) */
    uint64_t lfu_promotions;            /* Dynamic cold -> UMA hot-cache fills / LFU promotions */

    /* Physical 16-Core `ANE` (`MLComputePlan`) Multi-Horizon Router Lookahead Telemetry */
    uint64_t ane_lookahead_calls;       /* CoreML ANE multi-horizon router probe invocations */
    uint64_t ane_lookahead_hits;        /* Cold experts accurately prefetched via ANE lookahead before layer execution */
    uint64_t ran_on_ane;                /* Physical `MLNeuralEngineComputeDevice` verified execution count (`>= 1`!) */
    double   ane_total_time_ms;         /* Cumulative ANE probe wall-time (overlapped with GPU/NVMe) */

    /* Active Expert & DRAM Accounting */
    uint64_t active_experts_evaluated;  /* Total routed experts evaluated (`336/tok` main K=0 lossless) */
    uint64_t adaptive_cold_skipped;     /* Low-weight (`< gate_tau` or `> cold_cap`) cold experts skipped in adaptive mode */
    uint64_t physical_dram_bytes_read;  /* Cumulative physical UMA DRAM bytes read by Metal 4 GPU */
} StrataGlm53PerfCounters;

/*
 * Initializes the Regime II Tri-Engine Runtime on `Mac17,10`:
 *   - Maps `bundle->spine_base` (`~5.36 GiB`) zero-copy via `newBufferWithBytesNoCopy:` + `MTLResidencySet`.
 *   - Allocates `b_hot_experts` (`hot_cache_gib`, `<= 0.0f` => default `56.0 GiB` = `4,248` `13.50 MiB` slots `< 80.64 GiB`)
 *     + `[43][288]` `O(1)` slot residency table + LFU frequency counters + `b_ring_dma` staging buffer.
 *   - Compiles pure-C++20 CoreML spec-v7 multi-horizon `FP16 [288, 4096]` router lookahead probes on the 16-core `ANE`
 *     (`MLComputeUnitsCPUAndNeuralEngine`, verified via `MLComputePlan` -> `MLNeuralEngineComputeDevice`).
 *   - Compiles `strata_glm53_kernels.metal` (`MTLLanguageVersion4_0`) and allocates 34L `KDA` + 12L `DSA MLA` latent
 *     KV (`1 KiB/tok/layer`!) & 4-token pool indexer caches (`max_seq_len == 0` => `STRATA_GLM53_DEFAULT_MAX_SEQ_LEN`).
 */
StrataGlm53GpuEngine* strata_glm53_gpu_create(
    const StrataGlm53Bundle* bundle,
    uint32_t                 max_seq_len,
    float                    hot_cache_gib,
    StrataGlm53ExecMode      exec_mode
);

void strata_glm53_gpu_destroy(StrataGlm53GpuEngine* eng);

/* Resets sequence state (`seq_pos = 0`, `KDA` conv/recurrent state, `MLA`/indexer & `MTP` state) while KEEPING warm UMA hot-expert cache! */
void strata_glm53_gpu_reset(StrataGlm53GpuEngine* eng);

/* Optional: clears dynamic LFU hot-expert slots back to initial cold state for cold-cache benchmarks. */
void strata_glm53_gpu_reset_hot_cache(StrataGlm53GpuEngine* eng);

uint32_t strata_glm53_gpu_seq_pos(const StrataGlm53GpuEngine* eng);

/* Zero-cost runtime switch between `STRATA_GLM53_MODE_LOSSLESS` and `STRATA_GLM53_MODE_ADAPTIVE` (no buffer realloc!). */
void strata_glm53_gpu_set_exec_mode(
    StrataGlm53GpuEngine* eng,
    StrataGlm53ExecMode   exec_mode
);

StrataGlm53ExecMode strata_glm53_gpu_get_exec_mode(const StrataGlm53GpuEngine* eng);

/* Configures importance-gated adaptive parameters (`gate_tau <= 0` => default `0.20f`, `cold_cap` max cold experts/layer). */
void strata_glm53_gpu_set_adaptive_params(
    StrataGlm53GpuEngine* eng,
    float                 gate_tau,
    uint32_t              cold_cap,
    bool                  renorm_selected
);

/* Toggle opt-in INT8 NAX Prefill GEMMs (prefill M >= 32 only; decode M <= 5 remains 100% bit-exact FP16/FP32). */
void strata_glm53_gpu_set_int8_prefill(StrataGlm53GpuEngine* eng, bool enable);

/* Toggle 3-State SpecCtl Adaptive-K (CooldownK0 + HighConfFused + ActiveProbe; 100% bit-exact K>0 == K=0). */
void strata_glm53_gpu_set_adaptive_k(StrataGlm53GpuEngine* eng, bool enable);

/* Toggle 100% Lossless 20K SLC-Pinned Draft-Head Shortlist (`FR-Spec`, drafting only; verify always full 155K). */
void strata_glm53_gpu_set_frspec(StrataGlm53GpuEngine* eng, bool enable);

void strata_glm53_gpu_get_info(
    const StrataGlm53GpuEngine* eng,
    StrataGlm53GpuInfo*         info_out
);

void strata_glm53_gpu_get_counters(
    const StrataGlm53GpuEngine* eng,
    StrataGlm53PerfCounters*    counters_out
);

void strata_glm53_gpu_reset_counters(StrataGlm53GpuEngine* eng);

/*
 * Reset or append completion-scoped token frequency counters used when
 * `presence_penalty != 0`, `frequency_penalty != 0`, or `repetition_penalty != 1.0`.
 */
void strata_glm53_gpu_penalties_reset(StrataGlm53GpuEngine* eng);
void strata_glm53_gpu_penalties_record(
    StrataGlm53GpuEngine* eng,
    const uint32_t*       tokens,
    uint32_t              num_tokens
);

typedef void (*StrataGlm53PrefillProgressFn)(
    uint32_t done_tokens,
    uint32_t total_tokens,
    uint32_t seq_pos,
    void*    user_ctx
);

void strata_glm53_gpu_set_prefill_progress_cb(
    StrataGlm53GpuEngine*        eng,
    StrataGlm53PrefillProgressFn cb,
    void*                        user_ctx
);

/*
 * Incremental Batched Prefill over `token_ids[0 .. num_tokens - 1]` starting at current `seq_pos`,
 * advancing `seq_pos` by `num_tokens`, warming UMA LFU hot-expert slots, synchronizing/priming Layer-45
 * `MTP` state on the shifted sequence (`token_embed = 0` at `mtp_pos == 0`!), and writing the sampled/argmaxed
 * next token into `*next_token_out`. `last_logits_out` may be NULL.
 */
int strata_glm53_gpu_prefill(
    StrataGlm53GpuEngine*            eng,
    const uint32_t*                  token_ids,
    uint32_t                         num_tokens,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        next_token_out,
    float*                           last_logits_out
);

/*
 * Standard `K = 0` Single-Token Regime II Decode (`M = 1`): overlaps Metal 4 GPU spine/hot-expert execution
 * with 16-Core `ANE` (`MLComputePlan`) multi-horizon router lookahead + 8-FD parallel `F_NOCACHE` NVMe
 * `pread` into `Ring_DMA`, advancing `seq_pos` by 1 and writing `*next_token_out`. `logits_out` may be NULL.
 */
int strata_glm53_gpu_decode_step(
    StrataGlm53GpuEngine*            eng,
    uint32_t                         token_id,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        next_token_out,
    float*                           logits_out
);

/*
 * Native Layer-45 `MTP` Speculative Decode Step (`draft_k = 1 .. 3`, or `0` -> delegates to `decode_step`):
 *   1. Given `pending_token_id` (the uncommitted token emitted by `prefill` or the previous decode/spec step),
 *      runs Layer-45 `mtp` `draft_k` times to draft `draft_tokens[0 .. draft_k - 1]`.
 *   2. Verifies `[pending_token_id, draft_tokens[0 .. draft_k - 1]]` (`M = 1 + draft_k`) in ONE batched
 *      45-layer Main Model pass (under the active `exec_mode`) with exact Leviathan greedy/stochastic verification.
 *   3. Suppose `a` (`0 <= a <= draft_k`) draft tokens are accepted:
 *      - Rolls back rejected `KDA` conv/recurrent states + `MLA`/indexer tail states (`seq_pos` advances by `1 + a`).
 *      - Synchronizes MTP cache/state on the accepted true `(emitted, H_verify)` pairs.
 *      - Writes `*num_emitted_out = 1 + a` committed tokens into `emitted_tokens_out[0 .. a]`, where
 *        `emitted_tokens_out[a]` is the bonus/resampled token (pass as next step's `pending_token_id`!).
 *      - Sets `*num_draft_accepted_out = a` if non-NULL.
 *   Guarantees `100%` bit-identical greedy token IDs between `draft_k == 0` and `draft_k >= 1`!
 */
int strata_glm53_gpu_spec_decode_step(
    StrataGlm53GpuEngine*            eng,
    uint32_t                         pending_token_id,
    uint32_t                         draft_k,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        emitted_tokens_out,    /* capacity >= `1 + draft_k` */
    uint32_t*                        num_emitted_out,       /* in `[1 .. 1 + draft_k]`   */
    uint32_t*                        num_draft_accepted_out /* optional, in `[0 .. draft_k]` */
);

/*
 * O(1) Turn-Boundary GPU Checkpoint / Restore (`~142 MiB` GPU->GPU blit, `< 0.5 ms`):
 *   - Allocated lazily ONLY when `strata_glm53_checkpoint_create` is called (`0` bytes overhead in `bench-glm53`).
 *   - Snapshots `seq_pos`, `mtp_seq_pos`, `sample_step`, 34-layer `KDA` `b_conv_state` + `b_rec_state`,
 *     12-layer `Glm5NextIndexer` partial `< 4`-token tail window (`[12, 4, 2, 128]`, critical when `seq_pos % 4 != 0`!),
 *     and `MTP` pending hidden/seed state (`b_mla_kv_latent` & completed `pool_keys` are strictly append-only!).
 *   - Enables instant `/undo`, `/reset`, and `100%` bit-exact `--verify-multiturn` in `chat-glm53`!
 */
typedef struct StrataGlm53Checkpoint StrataGlm53Checkpoint;

StrataGlm53Checkpoint* strata_glm53_checkpoint_create(const StrataGlm53GpuEngine* eng);
void strata_glm53_checkpoint_destroy(StrataGlm53Checkpoint* ckpt);
int strata_glm53_checkpoint_save(StrataGlm53GpuEngine* eng, StrataGlm53Checkpoint* ckpt);
int strata_glm53_checkpoint_restore(StrataGlm53GpuEngine* eng, const StrataGlm53Checkpoint* ckpt);
uint64_t strata_glm53_checkpoint_bytes(const StrataGlm53Checkpoint* ckpt);

/* ============================================================================
 * 6. Step 4 — Two-Tier Zero-Pause Realtime Compaction & Session State ABI (`GLM-5.3-Flash`)
 *    Tier 2: Global-`pos`-Invariant Paged `DSA MLA` Latent KV (`b_block_loc`, 4 KiB/pool-block) +
 *            `F_NOCACHE` NVMe `kv_swap` + 16-Core ANE `indexer_wq_b` cold-pool lookahead + exact recall.
 *    Tier 1: Dual `StrataGlm53ContextSlot` (`live` / `shadow` / `rollback`) sharing the zero-copy spine &
 *            `56 GiB` UMA Hot-Expert Cache (`0 B` weight duplication) with `< 0.05 ms` O(1) RCU swap.
 * ============================================================================ */

typedef struct StrataGlm53CompactionStats {
    uint32_t mode;                     /* StrataCompactionMode of the slot */
    uint32_t logical_seq_pos;
    uint32_t uma_kv_blocks_capacity;
    uint32_t uma_kv_blocks_used;
    uint32_t logical_blocks;
    uint32_t nvme_blocks_now_paged;
    uint64_t nvme_blocks_paged_out;
    uint64_t nvme_blocks_recalled;
    uint64_t nvme_bytes_written;
    uint64_t nvme_bytes_read;
    double   nvme_write_ms;
    double   nvme_read_ms;
    uint64_t paged_sync_points;
    uint64_t sync_recall_misses;
    uint64_t ane_probe_calls;
    uint64_t ane_probe_on_ane;
    uint32_t ran_on_ane;
    uint32_t reserved0;
    uint64_t ane_prefetch_issued;
    uint64_t ane_prefetch_hits;
    double   ane_probe_us_avg;
    char     nvme_swap_path[256];
    uint32_t episodes_distilled;
    uint32_t shadow_seq_pos;
    bool     shadow_dual_synced;
    uint32_t rcu_swaps_completed;
    double   last_switchover_ms;
    double   max_user_stall_ms;
} StrataGlm53CompactionStats;

typedef struct StrataGlm53ContextSlot StrataGlm53ContextSlot;

StrataGlm53ContextSlot* strata_glm53_slot_create(
    StrataGlm53GpuEngine* eng,
    uint32_t              uma_kv_pool_tokens,
    const char*           nvme_swap_path
);
void strata_glm53_slot_destroy(StrataGlm53ContextSlot* slot);
void strata_glm53_bind_slot(StrataGlm53GpuEngine* eng, StrataGlm53ContextSlot* slot);
StrataGlm53ContextSlot* strata_glm53_bound_slot(const StrataGlm53GpuEngine* eng);
uint32_t strata_glm53_slot_seq_pos(const StrataGlm53ContextSlot* slot);
bool     strata_glm53_slot_is_paged(const StrataGlm53ContextSlot* slot);
uint64_t strata_glm53_slot_bytes(const StrataGlm53ContextSlot* slot);
void strata_glm53_slot_get_stats(const StrataGlm53ContextSlot* slot, StrataGlm53CompactionStats* stats_out);
void strata_glm53_slot_reset_stats(StrataGlm53ContextSlot* slot);
int strata_glm53_ane_kv_probe_ready(const StrataGlm53GpuEngine* eng, char* note_out, size_t note_cap);

int strata_glm53_gpu_prefill_slice(
    StrataGlm53GpuEngine* eng,
    const uint32_t*       token_ids,
    uint32_t              slice_tokens,
    bool                  compute_final_logits,
    uint32_t*             next_token_out
);

/* Canonical mode-independent GPU/NVMe sequence state serialization on the BOUND slot (`0..seq_pos` MLA KV + 34L KDA). */
int strata_glm53_state_save_fd(StrataGlm53GpuEngine* eng, int fd, uint64_t* bytes_written_out);
int strata_glm53_state_load_fd(StrataGlm53GpuEngine* eng, int fd, uint64_t* bytes_read_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_GLM53_H */
