/*
 * STRATA-M5 v2.0 — Causal Mode A / Zero-Stall Mode B Expert Planner,
 *                  FR-Spec 32K Shortlist, & Unbiased Leviathan Rejection Sampler
 * Owned by: models-mlx (src/model/strata_sampler.cpp)
 */

#include "../../include/strata_abi.h"

#include <errno.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace {

struct ColdCandidate {
    uint16_t expert_id;
    float    score_sum;
};

static inline double clamp_unit_rand(float u_rand_f32) {
    double u = static_cast<double>(u_rand_f32);
    if (u < 0.0) return 0.0;
    if (u >= 1.0) return 0.9999999999999999;
    return u;
}

static uint32_t sample_probs_f32_2pass(
    const float* probs,
    uint32_t n,
    float u_rand_f32
) {
    if (n == 0) return 0U;
    double sum_p = 0.0;
    uint32_t last_nonzero = 0U;
    for (uint32_t v = 0; v < n; ++v) {
        const float pv = probs[v];
        if (pv > 0.0f) {
            sum_p += static_cast<double>(pv);
            last_nonzero = v;
        }
    }
    if (sum_p <= 0.0) return 0U;

    const double target = clamp_unit_rand(u_rand_f32) * sum_p;
    double cum = 0.0;
    for (uint32_t v = 0; v < n; ++v) {
        const float pv = probs[v];
        if (pv > 0.0f) {
            cum += static_cast<double>(pv);
            if (cum > target) {
                return v;
            }
        }
    }
    return last_nonzero;
}

}  // namespace

extern "C" {

int strata_plan_layer_experts(
    StrataExecMode mode,
    uint32_t batch_B,
    uint32_t top_k,
    const uint16_t* topk_expert_ids,
    const float*    topk_gates,
    const bool*     is_uma_hot,
    float           mode_a_tau_tok,
    uint32_t        mode_b_cold_cap,
    StrataExpertPlan* plan_out
) {
    if (!topk_expert_ids || !topk_gates || !is_uma_hot || !plan_out ||
        batch_B == 0 || batch_B > STRATA_MAX_SPEC_BATCH ||
        top_k == 0 || top_k > STRATA_MAX_TOP_K) {
        return -EINVAL;
    }

    memset(plan_out, 0, sizeof(*plan_out));

    if (mode == STRATA_MODE_A_STATIONARY) {
        /*
         * MODE A (Strictly Prefix-Stationary Target, #12 §1.3):
         *   - Token t's refinement mask bit j depends ONLY on (topk_gates[t, j] >= mode_a_tau_tok).
         *   - Strictly invariant to future draft tokens t' != t AND invariant to transient
         *     LFU/ring cache state `is_uma_hot`, guaranteeing 100% unbiased Leviathan sampling!
         *   - Cold NVMe fetches collect the deduplicated union of non-UMA experts needed
         *     by at least one causal-refined (t, j) pair.
         */
        for (uint32_t t = 0; t < batch_B; ++t) {
            uint16_t mask = 0U;
            for (uint32_t j = 0; j < top_k; ++j) {
                const size_t idx = static_cast<size_t>(t) * top_k + j;
                const uint16_t e = topk_expert_ids[idx];
                const float    g = topk_gates[idx];

                if (g >= mode_a_tau_tok) {
                    mask = static_cast<uint16_t>(mask | (1U << j));
                    if (is_uma_hot[e]) {
                        plan_out->num_uma_hot_hits++;
                    } else {
                        bool already_queued = false;
                        for (uint32_t c = 0; c < plan_out->num_cold_fetches; ++c) {
                            if (plan_out->cold_fetch_experts[c] == e) {
                                already_queued = true;
                                break;
                            }
                        }
                        if (!already_queued) {
                            plan_out->cold_fetch_experts[plan_out->num_cold_fetches++] = e;
                        }
                    }
                }
            }
            plan_out->refine_mask_per_tok[t] = mask;
        }
        return 0;
    }

    if (mode == STRATA_MODE_B_ZERO_STALL) {
        /*
         * MODE B (MoE-Spec + HOBBIT Zero-Stall Mode, #12 §1.3):
         *   1. 100% of UMA-resident hot/pinned hits run at full 4-bit precision for FREE.
         *   2. Cold experts accumulate cross-batch importance S_e = sum_{t: e in TopK(t)} g_{t,e}.
         *   3. Only top `mode_b_cold_cap` cold experts by S_e are scheduled for NVMe DMA;
         *      all remaining cold tail experts execute on UMA QTIP Base / W6A16 MoE-SVD in 0.00 ms.
         */
        ColdCandidate cold_cands[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
        uint32_t num_unique_cold = 0U;

        for (uint32_t t = 0; t < batch_B; ++t) {
            for (uint32_t j = 0; j < top_k; ++j) {
                const size_t idx = static_cast<size_t>(t) * top_k + j;
                const uint16_t e = topk_expert_ids[idx];
                const float    g = topk_gates[idx];

                if (is_uma_hot[e]) {
                    plan_out->refine_mask_per_tok[t] =
                        static_cast<uint16_t>(plan_out->refine_mask_per_tok[t] | (1U << j));
                    plan_out->num_uma_hot_hits++;
                } else if (g > 0.0f) {
                    bool found = false;
                    for (uint32_t c = 0; c < num_unique_cold; ++c) {
                        if (cold_cands[c].expert_id == e) {
                            cold_cands[c].score_sum += g;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        cold_cands[num_unique_cold].expert_id = e;
                        cold_cands[num_unique_cold].score_sum = g;
                        ++num_unique_cold;
                    }
                }
            }
        }

        std::sort(cold_cands, cold_cands + num_unique_cold,
                  [](const ColdCandidate& a, const ColdCandidate& b) {
                      if (a.score_sum != b.score_sum) {
                          return a.score_sum > b.score_sum;
                      }
                      return a.expert_id < b.expert_id;
                  });

        const uint32_t admitted = std::min<uint32_t>(num_unique_cold, mode_b_cold_cap);
        plan_out->num_cold_fetches = admitted;
        for (uint32_t c = 0; c < admitted; ++c) {
            plan_out->cold_fetch_experts[c] = cold_cands[c].expert_id;
        }

        /* Mark admitted cold experts in refine_mask_per_tok (GPU kernel still guards slot_ready!) */
        if (admitted > 0) {
            for (uint32_t t = 0; t < batch_B; ++t) {
                for (uint32_t j = 0; j < top_k; ++j) {
                    const uint16_t e = topk_expert_ids[static_cast<size_t>(t) * top_k + j];
                    if (!is_uma_hot[e]) {
                        for (uint32_t c = 0; c < admitted; ++c) {
                            if (plan_out->cold_fetch_experts[c] == e) {
                                plan_out->refine_mask_per_tok[t] = static_cast<uint16_t>(
                                    plan_out->refine_mask_per_tok[t] | (1U << j));
                                break;
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }

    return -EINVAL;
}

void strata_frspec_shortlist_probs(
    const float* draft_shortlist_logits,
    const uint32_t* shortlist_vocab_ids,
    uint32_t shortlist_size,
    uint32_t full_vocab_size,
    float* full_draft_probs_out
) {
    if (!draft_shortlist_logits || !shortlist_vocab_ids || !full_draft_probs_out ||
        shortlist_size == 0 || full_vocab_size == 0) {
        return;
    }

    memset(full_draft_probs_out, 0, static_cast<size_t>(full_vocab_size) * sizeof(float));

    float max_logit = draft_shortlist_logits[0];
    for (uint32_t i = 1; i < shortlist_size; ++i) {
        if (draft_shortlist_logits[i] > max_logit) {
            max_logit = draft_shortlist_logits[i];
        }
    }

    thread_local std::vector<float> exp_vals;
    if (exp_vals.size() < shortlist_size) {
        exp_vals.resize(shortlist_size);
    }
    double sum_exp = 0.0;
    for (uint32_t i = 0; i < shortlist_size; ++i) {
        const float e = expf(draft_shortlist_logits[i] - max_logit);
        exp_vals[i] = e;
        sum_exp += static_cast<double>(e);
    }

    const float inv_sum = (sum_exp > 0.0)
                              ? static_cast<float>(1.0 / sum_exp)
                              : (1.0f / static_cast<float>(shortlist_size));
    for (uint32_t i = 0; i < shortlist_size; ++i) {
        const uint32_t vid = shortlist_vocab_ids[i];
        if (vid < full_vocab_size) {
            full_draft_probs_out[vid] += exp_vals[i] * inv_sum;
        }
    }
}

int strata_leviathan_verify_step(
    uint32_t draft_K,
    uint32_t full_vocab_size,
    const uint32_t* draft_tokens,
    const float* draft_probs_KxV,
    const float* target_probs_BxV,
    const float* uniform_rands_K,
    StrataVerifyResult* result_out
) {
    if (!target_probs_BxV || !uniform_rands_K || !result_out ||
        full_vocab_size == 0 || (draft_K + 1U) > STRATA_MAX_SPEC_BATCH ||
        (draft_K > 0 && (!draft_tokens || !draft_probs_KxV))) {
        return -EINVAL;
    }

    memset(result_out, 0, sizeof(*result_out));

    for (uint32_t m = 0; m < draft_K; ++m) {
        const uint32_t x_m = draft_tokens[m];
        if (x_m >= full_vocab_size) {
            return -EINVAL;
        }

        const float* q_m = draft_probs_KxV + static_cast<size_t>(m) * full_vocab_size;
        const float* p_m = target_probs_BxV + static_cast<size_t>(m) * full_vocab_size;

        const double q_x = static_cast<double>(q_m[x_m]);
        const double p_x = static_cast<double>(p_m[x_m]);
        const double accept_prob = (q_x > 0.0)
                                       ? ((p_x >= q_x) ? 1.0 : (p_x / q_x))
                                       : 0.0;

        if (static_cast<double>(uniform_rands_K[m]) < accept_prob) {
            result_out->emitted_tokens[result_out->accepted_count++] = x_m;
        } else {
            /*
             * Zero-allocation 2-pass residual sampling: x'_m ~ norm(max(0, p_m - q_m)).
             * Under FR-Spec (32K shortlist -> 155K/248K full vocab), every out-of-shortlist
             * token v has q_m[v] == 0, so max(0, p_m[v] - 0) == p_m[v], preserving 100%
             * of rare tail probability mass with zero bias.
             */
            double sum_pos = 0.0;
            uint32_t last_pos = 0U;
            for (uint32_t v = 0; v < full_vocab_size; ++v) {
                const double diff = static_cast<double>(p_m[v]) - static_cast<double>(q_m[v]);
                if (diff > 0.0) {
                    sum_pos += diff;
                    last_pos = v;
                }
            }

            uint32_t resampled = 0U;
            if (sum_pos > 1e-15) {
                const double target = clamp_unit_rand(uniform_rands_K[draft_K]) * sum_pos;
                double cum = 0.0;
                resampled = last_pos;
                for (uint32_t v = 0; v < full_vocab_size; ++v) {
                    const double diff = static_cast<double>(p_m[v]) - static_cast<double>(q_m[v]);
                    if (diff > 0.0) {
                        cum += diff;
                        if (cum > target) {
                            resampled = v;
                            break;
                        }
                    }
                }
            } else {
                resampled = sample_probs_f32_2pass(p_m, full_vocab_size, uniform_rands_K[draft_K]);
            }

            result_out->emitted_tokens[result_out->accepted_count++] = resampled;
            return 0;
        }
    }

    /* All draft_K tokens accepted! Emit 1 bonus token from target_probs_BxV[draft_K] */
    const float* p_bonus = target_probs_BxV + static_cast<size_t>(draft_K) * full_vocab_size;
    const uint32_t bonus_tok = sample_probs_f32_2pass(
        p_bonus, full_vocab_size, uniform_rands_K[draft_K]);
    result_out->emitted_tokens[result_out->accepted_count++] = bonus_tok;
    return 0;
}

}  // extern "C"
