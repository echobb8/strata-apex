/*
 * STRATA-M5 v2.0 — Real Qwen3.8-27B Hugging Face CLI & Long-Context Benchmark Suite
 * File: src/cli/strata_qwen38_bench.mm
 *
 * Implements:
 *   - convert-hf --hf-dir <dir> --bundle <dir>
 *   - run-hf     --bundle <dir> [--prompt "<text>" | --prompt-file <path>]
 *                [--max-tokens 256] [--greedy | --temp 0.6] [--no-think]
 *   - bench-hf   --bundle <dir> [--contexts 512,4096,16384,32768] [--gen-tokens 128]
 */

#import <Foundation/Foundation.h>

#include "../../include/strata_qwen38.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static double sq38_now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static uint64_t sq38_dir_total_bytes(const std::string& dir_path) {
    std::error_code ec;
    uint64_t total = 0ULL;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
        if (!ec && entry.is_regular_file(ec)) {
            total += static_cast<uint64_t>(entry.file_size(ec));
        }
    }
    return total;
}

static bool sq38_read_file_utf8(const std::string& path, std::string* out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out = oss.str();
    return true;
}

static bool sq38_parse_contexts_csv(const std::string& csv, std::vector<uint32_t>* out) {
    out->clear();
    size_t start = 0;
    while (start <= csv.size()) {
        size_t comma = csv.find(',', start);
        std::string part = (comma == std::string::npos)
            ? csv.substr(start)
            : csv.substr(start, comma - start);
        if (!part.empty()) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(part.c_str(), &endp, 10);
            if (endp == part.c_str() || *endp != '\0' || v < 64UL || v > 262144UL) {
                return false;
            }
            out->push_back(static_cast<uint32_t>(v));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return !out->empty();
}

static std::string sq38_preview_line(const std::string& s, size_t max_len) {
    std::string out;
    out.reserve(std::min(s.size(), max_len + 4));
    bool prev_space = true;
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
        } else {
            out.push_back(c);
            prev_space = false;
        }
        if (out.size() >= max_len) {
            out += "...";
            break;
        }
    }
    return out;
}

static bool sq38_encode_piece(
    const StrataQwen38Tokenizer* tok,
    const std::string& text,
    bool parse_special,
    std::vector<uint32_t>* out
) {
    const uint32_t cap = static_cast<uint32_t>(text.size() + 64U);
    out->assign(cap, 0U);
    uint32_t n = 0U;
    if (strata_qwen38_encode(tok, text.c_str(), text.size(), parse_special, out->data(), cap, &n) != 0) {
        return false;
    }
    out->resize(n);
    return true;
}

/* Chat-mode prompt tokenization for the one-shot command.
 *
 * Only the TEMPLATE may spell control tokens. The user's own text is encoded as plain text (as the REPL and the
 * server already do), so `--prompt "$(cat untrusted.txt)"` cannot forge a role boundary or an end-of-turn marker.
 *
 * `rendered_a` / `rendered_b` are the chat template rendered around the one-byte probes "A" and "B". The formatter
 * emits `prefix + user_text + suffix`, so the single byte where the two renderings differ is the user slot. */
static bool sq38_encode_chat_prompt_plain_user(
    const StrataQwen38Tokenizer* tok,
    const std::string& rendered_a,
    const std::string& rendered_b,
    const std::string& user_text,
    std::vector<uint32_t>* out_ids
) {
    if (rendered_a.size() != rendered_b.size()) return false;
    size_t slot = 0U;
    while (slot < rendered_a.size() && rendered_a[slot] == rendered_b[slot]) ++slot;
    if (slot == rendered_a.size() ||
        rendered_a.compare(slot + 1U, std::string::npos, rendered_b, slot + 1U, std::string::npos) != 0) {
        return false;
    }
    const std::string parts[3] = {rendered_a.substr(0U, slot), user_text, rendered_a.substr(slot + 1U)};
    out_ids->clear();
    std::vector<uint32_t> piece;
    for (size_t i = 0U; i < 3U; ++i) {
        if (parts[i].empty()) continue;
        if (!sq38_encode_piece(tok, parts[i], /*parse_special=*/(i != 1U), &piece)) return false;
        out_ids->insert(out_ids->end(), piece.begin(), piece.end());
    }
    return true;
}

/*
 * Builds an exact `target_tokens`-token Qwen3.8 ChatML prompt (`--no-think` direct assistant turn)
 * containing a unique 6-digit passkey needle at ~15% token depth, instructing the model to state
 * the exact passkey first and immediately follow with an in-depth Unified Memory technical essay.
 */
static bool sq38_build_passkey_prompt(
    const StrataQwen38Tokenizer* tok,
    uint32_t target_tokens,
    std::vector<uint32_t>* out_ids,
    std::string* out_passkey,
    uint32_t* out_needle_pos
) {
    const uint32_t code = 100000U + static_cast<uint32_t>(
        (static_cast<uint64_t>(target_tokens) * 2654435761ULL) % 899999ULL);
    *out_passkey = std::to_string(code);

    const std::string prefix_full =
        "<|im_start|>user\nRead the following hardware architecture telemetry log carefully and "
        "remember the embedded secret passkey.\n\n";
    const std::string needle_str =
        "\n\n[CRITICAL PASSKEY RECORD] The secret passkey is " + *out_passkey +
        ". Remember this exact passkey: " + *out_passkey + ".\n\n";
    const std::string suffix_full =
        "\n\nBased on the telemetry log above, first state the exact 6-digit secret passkey on "
        "the very first line. Immediately after, write a comprehensive, detailed multi-section "
        "technical explanation (at least 600 words) of how Apple Silicon Unified Memory "
        "Architecture (UMA), Metal 4 Neural Accelerators (NAX), 16 KiB hardware page tables, "
        "and zero-copy memory mapping eliminate PCIe bottlenecks in LLM inference."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const std::string filler_str =
        "Telemetry entry: unified memory fabric maintains coherent shared residency across "
        "graphics compute clusters, matrix pipelines, and system cache hierarchy without "
        "external host bus staging.\n";

    std::vector<uint32_t> prefix_ids, needle_ids, suffix_ids, filler_ids;
    if (!sq38_encode_piece(tok, prefix_full, true, &prefix_ids) ||
        !sq38_encode_piece(tok, needle_str, false, &needle_ids) ||
        !sq38_encode_piece(tok, suffix_full, true, &suffix_ids) ||
        !sq38_encode_piece(tok, filler_str, false, &filler_ids) ||
        filler_ids.empty()) {
        return false;
    }

    if (prefix_ids.size() + needle_ids.size() + suffix_ids.size() > target_tokens) {
        const std::string p_short = "<|im_start|>user\nLog:\n";
        const std::string s_short =
            "\nState the exact passkey first, then explain Unified Memory in detail."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (!sq38_encode_piece(tok, p_short, true, &prefix_ids) ||
            !sq38_encode_piece(tok, s_short, true, &suffix_ids)) {
            return false;
        }
    }

    const size_t fixed = prefix_ids.size() + needle_ids.size() + suffix_ids.size();
    if (fixed > target_tokens) return false;

    const uint32_t budget = target_tokens - static_cast<uint32_t>(fixed);
    const uint32_t desired_pos = static_cast<uint32_t>(
        (static_cast<uint64_t>(target_tokens) * 15ULL) / 100ULL);
    uint32_t before_toks = (desired_pos > prefix_ids.size())
        ? (desired_pos - static_cast<uint32_t>(prefix_ids.size()))
        : 0U;
    if (before_toks > budget) before_toks = budget / 2U;
    const uint32_t after_toks = budget - before_toks;

    out_ids->clear();
    out_ids->reserve(target_tokens);
    out_ids->insert(out_ids->end(), prefix_ids.begin(), prefix_ids.end());
    for (uint32_t i = 0; i < before_toks; ++i) {
        out_ids->push_back(filler_ids[i % filler_ids.size()]);
    }
    *out_needle_pos = static_cast<uint32_t>(out_ids->size());
    out_ids->insert(out_ids->end(), needle_ids.begin(), needle_ids.end());
    for (uint32_t i = 0; i < after_toks; ++i) {
        out_ids->push_back(filler_ids[i % filler_ids.size()]);
    }
    out_ids->insert(out_ids->end(), suffix_ids.begin(), suffix_ids.end());
    return out_ids->size() == target_tokens;
}

} /* namespace */

extern "C" {

int strata_qwen38_cmd_convert_hf(int argc, char** argv) {
    std::string hf_dir;
    std::string bundle_dir;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--hf-dir" || a == "--in") && i + 1 < argc) {
            hf_dir = argv[++i];
        } else if ((a == "--bundle" || a == "--out") && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s convert-hf --hf-dir <dir> --bundle <dir>\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for convert-hf: %s\n", a.c_str());
            return 1;
        }
    }

    if (hf_dir.empty() || bundle_dir.empty()) {
        std::fprintf(stderr, "Error: `convert-hf` requires `--hf-dir <dir> --bundle <dir>`\n");
        return 1;
    }

    StrataQwen38ConvertStats stats{};
    const int rc = strata_qwen38_convert_hf_directory(hf_dir.c_str(), bundle_dir.c_str(), &stats);
    if (rc != 0) {
        std::fprintf(stderr, "[strata_cli] convert-hf failed (rc=%d)\n", rc);
        return 1;
    }

    const uint64_t bundle_total_bytes = sq38_dir_total_bytes(bundle_dir);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double bw_gib_s = (stats.elapsed_sec > 0.0)
        ? ((static_cast<double>(stats.out_weights_bytes) / gib) / stats.elapsed_sec)
        : 0.0;

    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 Qwen3.8-27B Zero-Copy HF Conversion Complete\n");
    std::printf("================================================================================\n");
    std::printf("  Source HF Dir      : %s\n", hf_dir.c_str());
    std::printf("  Output Bundle Dir  : %s\n", bundle_dir.c_str());
    std::printf("  Shards & Tensors   : %u shards | %u text converted | %u vision skipped\n",
                stats.num_shards,
                stats.num_text_tensors_converted,
                stats.num_vision_tensors_skipped);
    std::printf("  Source Safetensors : %.3f GiB (%llu B)\n",
                static_cast<double>(stats.hf_safetensors_bytes) / gib,
                static_cast<unsigned long long>(stats.hf_safetensors_bytes));
    std::printf("  weights.strata     : %.3f GiB (%llu B | 16 KiB Page-Aligned=%s)\n",
                static_cast<double>(stats.out_weights_bytes) / gib,
                static_cast<unsigned long long>(stats.out_weights_bytes),
                stats.page_aligned_verified ? "PASS" : "FAIL");
    std::printf("  Total Bundle Size  : %.3f GiB (weights.strata + tokenizer.json + manifest)\n",
                static_cast<double>(bundle_total_bytes) / gib);
    std::printf("  Active Decode Foot.: %.3f GB/tok (%.3f GiB/tok)\n",
                static_cast<double>(stats.active_bytes_per_tok) / 1e9,
                static_cast<double>(stats.active_bytes_per_tok) / gib);
    std::printf("  Wall Time & RSS    : %.2f s (%.2f GiB/s write) | Peak RSS %.3f GiB\n",
                stats.elapsed_sec, bw_gib_s, stats.peak_rss_gib);
    std::printf("================================================================================\n");
    return stats.page_aligned_verified ? 0 : 1;
}

int strata_qwen38_cmd_run_hf(int argc, char** argv) {
    std::string bundle_dir;
    std::string prompt = "Explain why Apple Silicon Unified Memory Architecture eliminates PCIe bottlenecks for large language models.";
    std::string prompt_file;
    std::string system_prompt;
    uint32_t max_tokens = 256U;
    bool no_think       = false;
    StrataQwen38SamplingParams sampling{};
    sampling.greedy      = false;
    sampling.temperature = 0.6f;
    sampling.top_k       = 20U;
    sampling.top_p       = 0.95f;
    sampling.rng_seed    = 42ULL;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (a == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (a == "--max-tokens" && i + 1 < argc) {
            max_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--greedy") {
            sampling.greedy      = true;
            sampling.temperature = 0.0f;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            sampling.temperature = std::strtof(argv[++i], nullptr);
            sampling.greedy      = (sampling.temperature <= 1e-6f);
        } else if (a == "--top-k" && i + 1 < argc) {
            sampling.top_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--top-p" && i + 1 < argc) {
            sampling.top_p = std::strtof(argv[++i], nullptr);
        } else if (a == "--seed" && i + 1 < argc) {
            sampling.rng_seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--no-think") {
            no_think = true;
        } else if (a == "--think") {
            no_think = false;
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: %s run-hf --bundle <dir> [--prompt \"<text>\" | --prompt-file <path>]\n"
                "                 [--max-tokens 256] [--greedy | --temp 0.6] [--no-think]\n",
                argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for run-hf: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty()) {
        std::fprintf(stderr, "Error: `run-hf` requires `--bundle <dir>`\n");
        return 1;
    }
    if (!prompt_file.empty() && !sq38_read_file_utf8(prompt_file, &prompt)) {
        std::fprintf(stderr, "Error: failed to read `--prompt-file %s`\n", prompt_file.c_str());
        return 1;
    }

    StrataQwen38Bundle bundle{};
    if (strata_qwen38_load_bundle(bundle_dir.c_str(), &bundle) != 0) {
        std::fprintf(stderr, "[strata_cli] Failed to load Qwen3.8 bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    /* The user's text is plain text; only the template may spell control tokens (see
     * `sq38_encode_chat_prompt_plain_user`). */
    const char* sys_ptr = system_prompt.empty() ? nullptr : system_prompt.c_str();
    auto render = [&](const char* user_text, std::string* out) -> bool {
        std::vector<char> buf(std::strlen(user_text) + system_prompt.size() + 1024U, '\0');
        size_t len = 0U;
        if (strata_qwen38_format_chat_prompt(user_text, sys_ptr, !no_think, buf.data(), buf.size(), &len) != 0) {
            return false;
        }
        out->assign(buf.data(), len);
        return true;
    };
    std::string rendered_a, rendered_b;
    if (!render("A", &rendered_a) || !render("B", &rendered_b)) {
        std::fprintf(stderr, "[strata_cli] Failed to format ChatML prompt\n");
        strata_qwen38_free_bundle(&bundle);
        return 1;
    }

    std::vector<uint32_t> prompt_ids;
    if (!sq38_encode_chat_prompt_plain_user(bundle.tokenizer, rendered_a, rendered_b, prompt, &prompt_ids) ||
        prompt_ids.empty()) {
        std::fprintf(stderr, "[strata_cli] BPE prompt encoding failed\n");
        strata_qwen38_free_bundle(&bundle);
        return 1;
    }
    const uint32_t prompt_len = static_cast<uint32_t>(prompt_ids.size());

    const uint32_t max_seq_len = prompt_len + max_tokens + 64U;
    StrataQwen38GpuEngine* eng = strata_qwen38_gpu_create(&bundle, max_seq_len);
    if (!eng) {
        std::fprintf(stderr, "[strata_cli] Metal 4 Qwen3.8 GPU engine creation failed\n");
        strata_qwen38_free_bundle(&bundle);
        return 1;
    }

    StrataQwen38GpuInfo info{};
    strata_qwen38_gpu_get_info(eng, &info);

    std::printf("================================================================================\n");
    std::printf("  STRATA-M5 Qwen3.8-27B Metal 4 (NAX + Hybrid DeltaNet/GQA) Live Inference\n");
    std::printf("  Device: %s | Zero-Copy mmap: %.2f GiB | %s | %s\n",
                info.device_name,
                static_cast<double>(info.weights_mapped_bytes) / (1024.0 * 1024.0 * 1024.0),
                no_think ? "Direct (--no-think)" : "Thinking (<think>)",
                sampling.greedy ? "Greedy" : "Stochastic");
    std::printf("  Prompt: %u tokens | Max New Tokens: %u | Max Seq Alloc: %u\n",
                prompt_len, max_tokens, info.max_seq_len);
    std::printf("================================================================================\n");
    if (!no_think) {
        std::printf("<think>\n");
    }
    std::fflush(stdout);

    struct rusage ru_init{}, ru_steady{}, ru_end{};
    getrusage(RUSAGE_SELF, &ru_init);

    uint32_t cur_tok = 0U;
    const double t_pre0 = sq38_now_sec();
    int rc = strata_qwen38_gpu_prefill(eng, prompt_ids.data(), prompt_len, &sampling, &cur_tok, nullptr);
    const double t_pre1 = sq38_now_sec();
    if (rc != 0) {
        std::fprintf(stderr, "\n[strata_cli] GPU prefill failed (rc=%d)\n", rc);
        strata_qwen38_gpu_destroy(eng);
        strata_qwen38_free_bundle(&bundle);
        return 1;
    }

    const double ttft_ms = (t_pre1 - t_pre0) * 1000.0;
    const double prefill_tok_s = (t_pre1 > t_pre0)
        ? (static_cast<double>(prompt_len) / (t_pre1 - t_pre0))
        : 0.0;

    StrataQwen38StreamDecoder sdec{};
    strata_qwen38_stream_decoder_init(&sdec);
    char piece[256];
    size_t piece_len = 0U;

    uint32_t generated_tokens = 0U;
    uint32_t decode_steps     = 0U;
    uint64_t total_kv_read    = 0ULL;

    if (max_tokens > 0U && !strata_qwen38_is_eos_token(&bundle.cfg, cur_tok)) {
        if (strata_qwen38_stream_decode_step(
                bundle.tokenizer, &sdec, cur_tok, true, piece, sizeof(piece), &piece_len) == 0 &&
            piece_len > 0U) {
            std::fwrite(piece, 1, piece_len, stdout);
            std::fflush(stdout);
        }
        generated_tokens = 1U;
    }

    getrusage(RUSAGE_SELF, &ru_steady);
    const double t_dec0 = sq38_now_sec();

    if (generated_tokens == 1U) {
        while (generated_tokens < max_tokens) {
            const uint32_t seq_after = prompt_len + decode_steps + 1U;
            uint32_t next_tok = 0U;
            rc = strata_qwen38_gpu_decode_step(eng, cur_tok, &sampling, &next_tok, nullptr);
            if (rc != 0) {
                std::fprintf(stderr, "\n[strata_cli] GPU decode_step failed (rc=%d)\n", rc);
                strata_qwen38_gpu_destroy(eng);
                strata_qwen38_free_bundle(&bundle);
                return 1;
            }
            ++decode_steps;
            total_kv_read += static_cast<uint64_t>(seq_after) * info.kv_cache_bytes_per_token;
            cur_tok = next_tok;

            if (strata_qwen38_is_eos_token(&bundle.cfg, cur_tok)) {
                break;
            }
            if (strata_qwen38_stream_decode_step(
                    bundle.tokenizer, &sdec, cur_tok, true, piece, sizeof(piece), &piece_len) == 0 &&
                piece_len > 0U) {
                std::fwrite(piece, 1, piece_len, stdout);
                std::fflush(stdout);
            }
            ++generated_tokens;
        }
    }

    const double t_dec1 = sq38_now_sec();
    getrusage(RUSAGE_SELF, &ru_end);

    const double decode_sec  = t_dec1 - t_dec0;
    const double decode_ms   = decode_sec * 1000.0;
    const double decode_tps  = (decode_sec > 0.0 && decode_steps > 0U)
        ? (static_cast<double>(decode_steps) / decode_sec)
        : 0.0;
    const uint64_t total_dram_bytes =
        static_cast<uint64_t>(decode_steps) * info.active_bytes_per_token + total_kv_read;
    const double dram_gb_s = (decode_sec > 0.0 && decode_steps > 0U)
        ? ((static_cast<double>(total_dram_bytes) / 1e9) / decode_sec)
        : 0.0;
    const long init_majflt   = ru_steady.ru_majflt - ru_init.ru_majflt;
    const long steady_majflt = ru_end.ru_majflt - ru_steady.ru_majflt;
    const double peak_rss_gib = static_cast<double>(ru_end.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t final_seq = strata_qwen38_gpu_seq_pos(eng);
    const double state_mib = static_cast<double>(
        info.deltanet_state_bytes + static_cast<uint64_t>(final_seq) * info.kv_cache_bytes_per_token) /
        (1024.0 * 1024.0);

    std::printf("\n--------------------------------------------------------------------------------\n");
    std::printf("  Qwen3.8-27B Metal 4 Telemetry (%s | init %.1f ms)\n", info.device_name, info.load_time_ms);
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  TTFT & Prefill     : %8.2f ms | %8.2f tok/s (%u prompt tokens)\n",
                ttft_ms, prefill_tok_s, prompt_len);
    std::printf("  Decode Throughput  : %8.2f tok/s | %u decode steps in %.2f ms (%u output toks)\n",
                decode_tps, decode_steps, decode_ms, generated_tokens);
    std::printf("  Modeled DRAM BW    : %8.2f GB/s modeled (%.3f GB/tok weights + FP16 KV read)\n",
                dram_gb_s, static_cast<double>(info.active_bytes_per_token) / 1e9);
    std::printf("  Hybrid State Foot. : %8.2f MiB (DeltaNet %.2f MiB O(1) + KV %.2f MiB @ %u)\n",
                state_mib,
                static_cast<double>(info.deltanet_state_bytes) / (1024.0 * 1024.0),
                static_cast<double>(static_cast<uint64_t>(final_seq) * info.kv_cache_bytes_per_token) / (1024.0 * 1024.0),
                final_seq);
    std::printf("  Memory & Faults    : Peak RSS %.3f GiB | Steady MajFlt=%ld (Init=%ld)\n",
                peak_rss_gib, steady_majflt, init_majflt);
    std::printf("================================================================================\n");

    strata_qwen38_gpu_destroy(eng);
    strata_qwen38_free_bundle(&bundle);
    return 0;
}

int strata_qwen38_cmd_bench_hf(int argc, char** argv) {
    std::string bundle_dir;
    std::string contexts_csv = "512,4096,16384,32768";
    uint32_t gen_tokens      = 128U;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        } else if (a == "--contexts" && i + 1 < argc) {
            contexts_csv = argv[++i];
        } else if (a == "--gen-tokens" && i + 1 < argc) {
            gen_tokens = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: %s bench-hf --bundle <dir> [--contexts 512,4096,16384,32768] [--gen-tokens 128]\n",
                argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option for bench-hf: %s\n", a.c_str());
            return 1;
        }
    }

    if (bundle_dir.empty() || gen_tokens == 0U) {
        std::fprintf(stderr, "Error: `bench-hf` requires `--bundle <dir>` and `--gen-tokens > 0`\n");
        return 1;
    }

    std::vector<uint32_t> contexts;
    if (!sq38_parse_contexts_csv(contexts_csv, &contexts)) {
        std::fprintf(stderr, "Error: invalid `--contexts %s` (expected comma-separated integers >= 64)\n",
                     contexts_csv.c_str());
        return 1;
    }

    StrataQwen38Bundle bundle{};
    if (strata_qwen38_load_bundle(bundle_dir.c_str(), &bundle) != 0) {
        std::fprintf(stderr, "[strata_cli] Failed to load Qwen3.8 bundle `%s`\n", bundle_dir.c_str());
        return 1;
    }

    const uint32_t max_context = *std::max_element(contexts.begin(), contexts.end());
    const uint32_t max_seq_len = max_context + gen_tokens + 128U;
    StrataQwen38GpuEngine* eng = strata_qwen38_gpu_create(&bundle, max_seq_len);
    if (!eng) {
        std::fprintf(stderr, "[strata_cli] Metal 4 Qwen3.8 GPU engine creation failed\n");
        strata_qwen38_free_bundle(&bundle);
        return 1;
    }

    StrataQwen38GpuInfo info{};
    strata_qwen38_gpu_get_info(eng, &info);

    StrataQwen38SamplingParams greedy{};
    greedy.greedy      = true;
    greedy.temperature = 0.0f;

    /* Single warmup pass to fault zero-copy mmap pages & warm Metal 4 PSOs */
    {
        std::vector<uint32_t> warm_ids;
        std::string warm_pk;
        uint32_t warm_pos = 0U;
        if (sq38_build_passkey_prompt(bundle.tokenizer, 128U, &warm_ids, &warm_pk, &warm_pos)) {
            uint32_t tok = 0U;
            (void)strata_qwen38_gpu_prefill(eng, warm_ids.data(), 128U, &greedy, &tok, nullptr);
            for (int w = 0; w < 4; ++w) {
                uint32_t nxt = 0U;
                (void)strata_qwen38_gpu_decode_step(eng, tok, &greedy, &nxt, nullptr);
                tok = nxt;
            }
        }
        strata_qwen38_gpu_reset(eng);
    }

    std::printf("========================================================================================================\n");
    std::printf("  STRATA-M5 Qwen3.8-27B Long-Context Benchmark & Passkey Needle Retrieval (%s)\n",
                info.device_name);
    std::printf("  Zero-Copy mmap: %.2f GiB | Active Weights: %.3f GB/tok | DeltaNet O(1): %.2f MiB | Gen: %u toks\n",
                static_cast<double>(info.weights_mapped_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(info.active_bytes_per_token) / 1e9,
                static_cast<double>(info.deltanet_state_bytes) / (1024.0 * 1024.0),
                gen_tokens);
    std::printf("========================================================================================================\n");
    std::printf("  %7s | %11s | %9s | %10s | %9s | %10s | %8s | %6s | %s\n",
                "Context", "Prefill t/s", "TTFT (ms)", "Decode t/s", "DRAM GB/s",
                "KV+DN MiB", "Peak RSS", "MajFlt", "Needle (~15%)");
    std::printf("  --------+-------------+-----------+------------+-----------+------------+----------+--------+---------------\n");
    std::fflush(stdout);

    struct PreviewItem {
        uint32_t    context;
        uint32_t    needle_pos;
        std::string passkey;
        bool        passed;
        std::string preview;
    };
    std::vector<PreviewItem> previews;
    bool all_passed = true;

    for (uint32_t ctx_len : contexts) {
        strata_qwen38_gpu_reset(eng);

        std::vector<uint32_t> prompt_ids;
        std::string passkey;
        uint32_t needle_pos = 0U;
        if (!sq38_build_passkey_prompt(bundle.tokenizer, ctx_len, &prompt_ids, &passkey, &needle_pos)) {
            std::fprintf(stderr, "[strata_cli] Failed to build %u-token passkey prompt\n", ctx_len);
            all_passed = false;
            break;
        }

        uint32_t cur_tok = 0U;
        const double t_pre0 = sq38_now_sec();
        int rc = strata_qwen38_gpu_prefill(eng, prompt_ids.data(), ctx_len, &greedy, &cur_tok, nullptr);
        const double t_pre1 = sq38_now_sec();
        if (rc != 0) {
            std::fprintf(stderr, "[strata_cli] Prefill failed at context %u (rc=%d)\n", ctx_len, rc);
            all_passed = false;
            break;
        }

        const double ttft_ms       = (t_pre1 - t_pre0) * 1000.0;
        const double prefill_tok_s = (t_pre1 > t_pre0)
            ? (static_cast<double>(ctx_len) / (t_pre1 - t_pre0))
            : 0.0;

        std::vector<uint32_t> out_ids;
        out_ids.reserve(gen_tokens + 1U);
        bool hit_eos = strata_qwen38_is_eos_token(&bundle.cfg, cur_tok);
        if (!hit_eos) out_ids.push_back(cur_tok);

        struct rusage ru_dec0{}, ru_dec1{};
        getrusage(RUSAGE_SELF, &ru_dec0);
        uint64_t total_kv_read = 0ULL;

        const double t_dec0 = sq38_now_sec();
        for (uint32_t s = 0; s < gen_tokens; ++s) {
            const uint32_t seq_after = ctx_len + s + 1U;
            uint32_t next_tok = 0U;
            rc = strata_qwen38_gpu_decode_step(eng, cur_tok, &greedy, &next_tok, nullptr);
            if (rc != 0) {
                std::fprintf(stderr, "[strata_cli] Decode failed at context %u step %u (rc=%d)\n",
                             ctx_len, s, rc);
                all_passed = false;
                break;
            }
            total_kv_read += static_cast<uint64_t>(seq_after) * info.kv_cache_bytes_per_token;
            cur_tok = next_tok;
            if (!hit_eos) {
                if (strata_qwen38_is_eos_token(&bundle.cfg, cur_tok)) {
                    hit_eos = true;
                } else {
                    out_ids.push_back(cur_tok);
                }
            }
        }
        const double t_dec1 = sq38_now_sec();
        getrusage(RUSAGE_SELF, &ru_dec1);
        if (!all_passed) break;

        const double decode_sec = t_dec1 - t_dec0;
        const double decode_tps = (decode_sec > 0.0)
            ? (static_cast<double>(gen_tokens) / decode_sec)
            : 0.0;
        const uint64_t total_dram_bytes =
            static_cast<uint64_t>(gen_tokens) * info.active_bytes_per_token + total_kv_read;
        const double dram_gb_s = (decode_sec > 0.0)
            ? ((static_cast<double>(total_dram_bytes) / 1e9) / decode_sec)
            : 0.0;

        const uint32_t seq_end = strata_qwen38_gpu_seq_pos(eng);
        const double state_mib = static_cast<double>(
            info.deltanet_state_bytes + static_cast<uint64_t>(seq_end) * info.kv_cache_bytes_per_token) /
            (1024.0 * 1024.0);
        const double peak_rss_gib = static_cast<double>(ru_dec1.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
        const long steady_majflt  = ru_dec1.ru_majflt - ru_dec0.ru_majflt;

        std::vector<char> decoded(out_ids.size() * 16U + 64U, '\0');
        size_t decoded_len = 0U;
        (void)strata_qwen38_decode(
            bundle.tokenizer, out_ids.data(), static_cast<uint32_t>(out_ids.size()),
            true, decoded.data(), decoded.size(), &decoded_len);
        const std::string text_out(decoded.data(), decoded_len);
        const bool needle_ok = (text_out.find(passkey) != std::string::npos);
        if (!needle_ok) all_passed = false;

        std::printf("  %7u | %11.2f | %9.2f | %10.2f | %9.2f | %10.2f | %6.2f G | %6ld | %s (%s)\n",
                    ctx_len, prefill_tok_s, ttft_ms, decode_tps, dram_gb_s,
                    state_mib, peak_rss_gib, steady_majflt,
                    needle_ok ? "PASS" : "FAIL", passkey.c_str());
        std::fflush(stdout);

        previews.push_back({ctx_len, needle_pos, passkey, needle_ok, sq38_preview_line(text_out, 78U)});
    }

    std::printf("========================================================================================================\n");
    for (const auto& p : previews) {
        const double pct = 100.0 * static_cast<double>(p.needle_pos) / static_cast<double>(p.context);
        std::printf("  [C=%5u | needle @ tok %u (%.1f%%) | key=%s | %s] %s\n",
                    p.context, p.needle_pos, pct, p.passkey.c_str(),
                    p.passed ? "PASS" : "FAIL", p.preview.c_str());
    }
    std::printf("========================================================================================================\n");

    strata_qwen38_gpu_destroy(eng);
    strata_qwen38_free_bundle(&bundle);
    return all_passed ? 0 : 1;
}

} /* extern "C" */
