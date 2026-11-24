#!/usr/bin/env python3
"""
STRATA-M5 3D Procedural Eval Runner with 1-Pass Check & Self-Repair.

Workflow per model (strictly sequential for 128 GiB UMA safety):
  1. Draft Pass   : Generate single-file Three.js HTML from prompt.
  2. Verify Pass  : Run `scripts/check_3d_html.mjs` (HTML structure + `node --check`
                    + live headless Chrome runtime/WebGL/console/blank-canvas check).
  3. 1-Pass Fix   : If >= 1 error was found, hand the exact error report + numbered
                    file back to the SAME model for ONE surgical SEARCH/REPLACE fix pass,
                    apply patches, and re-verify.
  4. Scorecard    : Print + save `report.md` and `report.json` showing errors found,
                    fix outcome, pure inference fix time, and total wall fix time.

Examples:
  python3 scripts/eval_3d.py --out evals/hogwarts_run4
  python3 scripts/eval_3d.py --models inkling2,qwen180b,glm53 --out evals/hogwarts_run4
  python3 scripts/eval_3d.py --repair-only evals/hogwarts_run3 --models inkling2
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATA_CLI = (ROOT / "build" / "strata-apex") if (ROOT / "build" / "strata-apex").exists() else (ROOT / "build" / "strata_cli")
CHECKER_JS = ROOT / "scripts" / "check_3d_html.mjs"
TELEMETRY_SEP = "\n" + ("-" * 60)

def _resolve_cache_bundle(rel: str) -> str:
    env_root = os.environ.get("STRATA_APEX_CACHE_DIR") or os.environ.get("STRATA_CACHE_DIR")
    if env_root:
        p = Path(env_root) / rel
        if p.exists():
            return str(p)
    for base in (Path.home() / ".cache" / "strata-apex", Path.home() / ".cache" / "strata"):
        p = base / rel
        if p.exists():
            return str(p)
    return str(Path.home() / ".cache" / "strata-apex" / rel)

MODELS = {
    "inkling2": {
        "label": "Inkling-2-Mini (26B)",
        "subcmd": "run-inkling2",
        "bundle": _resolve_cache_bundle("inkling-2-mini-fast"),
        "draft_extra": ["--draft-k", "7", "--effort", "medium"],
        "repair_extra": ["--draft-k", "7", "--effort", "low"],
    },
    "qwen180b": {
        "label": "Qwen3.8-Flash-Next (180B)",
        "subcmd": "run-flash",
        "bundle": _resolve_cache_bundle("qwen3.8-flash-next-fast"),
        "draft_extra": ["--draft-k", "4", "--effort", "medium"],
        "repair_extra": ["--draft-k", "4", "--effort", "low"],
    },
    "glm53": {
        "label": "GLM-5.3-Flash (320B)",
        "subcmd": "run-glm53",
        "bundle": _resolve_cache_bundle("glm-5.3-flash-fast"),
        "draft_extra": ["--mode", "lossless", "--draft-k", "4", "--effort", "low", "--max-think-tokens", "12288"],
        "repair_extra": ["--mode", "lossless", "--draft-k", "4", "--effort", "low", "--max-think-tokens", "2048"],
    },
}


def strip_think(raw_text: str) -> tuple[str, int, bool]:
    """Return (answer_after_think, think_chars, stuck_in_think)."""
    body = raw_text.split(TELEMETRY_SEP)[0]
    t0 = body.find("<think>", 300)
    t1 = body.find("</think>", t0) if t0 != -1 else body.find("</think>")
    if t0 != -1 and t1 == -1:
        return "", len(body) - t0, True
    if t1 != -1:
        think_chars = (t1 - t0) if t0 != -1 else t1
        return body[t1 + len("</think>"):], max(0, think_chars), False
    return body, 0, False


def extract_html(raw_text: str) -> tuple[str, int, bool]:
    answer, think_chars, stuck = strip_think(raw_text)
    if stuck or not answer:
        return "", think_chars, stuck
    matches = re.findall(r"```(?:html)?\s*\n(.*?)```", answer, flags=re.S | re.I)
    if matches:
        # Prefer the largest fenced block containing <!doctype html or <html
        html_blocks = [m.strip() for m in matches if "<html" in m.lower() or "<!doctype" in m.lower()]
        if html_blocks:
            return max(html_blocks, key=len), think_chars, False
        return matches[-1].strip(), think_chars, False
    low = answer.lower()
    if "<!doctype html" in low:
        return answer[low.rfind("<!doctype html"):].strip(), think_chars, False
    return "", think_chars, False


def parse_telemetry(raw_text: str) -> dict:
    info = {
        "prefill_tok_s": None,
        "ttft_ms": None,
        "decode_tok_s": None,
        "decode_tokens": None,
        "decode_ms": None,
        "mtp_tok_per_step": None,
        "mtp_accept_pct": None,
        "uma_gib": None,
        "steady_majflt": None,
        "think_tokens": None,
        "think_state": None,
    }
    if TELEMETRY_SEP not in raw_text:
        return info
    footer = raw_text.split(TELEMETRY_SEP)[-1]
    if m := re.search(r"Prefill & TTFT\s*:\s*([\d.]+)\s*tok/s\s*\|\s*TTFT\s*([\d.]+)\s*ms", footer):
        info["prefill_tok_s"] = float(m.group(1))
        info["ttft_ms"] = float(m.group(2))
    if m := re.search(r"Decode Throughput\s*:\s*([\d.]+)\s*tok/s\s*\|\s*(\d+)\s*decode toks.*?\(([\d.]+)\s*ms", footer):
        info["decode_tok_s"] = float(m.group(1))
        info["decode_tokens"] = int(m.group(2))
        info["decode_ms"] = float(m.group(3))
    if m := re.search(r"Native MTP.*?:\s*([\d.]+)\s*tok/step.*?\(([\d.]+)%\)", footer):
        info["mtp_tok_per_step"] = float(m.group(1))
        info["mtp_accept_pct"] = float(m.group(2))
    if m := re.search(r"Total UMA\s*([\d.]+)\s*GiB.*?Steady MajFlt=(\d+)", footer):
        info["uma_gib"] = float(m.group(1))
        info["steady_majflt"] = int(m.group(2))
    if m := re.search(r"Reasoning \(<think>\)\s*:\s*(\d+)\s*tokens\s*\(([^)]+)\)", footer):
        info["think_tokens"] = int(m.group(1))
        info["think_state"] = m.group(2)
    return info


def infer_sec_from_tel(tel: dict) -> float | None:
    if tel.get("ttft_ms") is not None and tel.get("decode_ms") is not None:
        return (tel["ttft_ms"] + tel["decode_ms"]) / 1000.0
    return None


def run_strata_streaming(cmd: list[str], out_raw_path: Path, quiet: bool = False) -> tuple[int, str, float]:
    """Run strata_cli, streaming output live to terminal while saving raw output."""
    t0 = time.monotonic()
    chunks = []
    with out_raw_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fout.write(line)
            fout.flush()
            chunks.append(line)
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
    elapsed = time.monotonic() - t0
    return rc, "".join(chunks), elapsed


def run_checker(html_path: Path, screenshot_path: Path | None = None, no_browser: bool = False) -> dict:
    cmd = ["node", str(CHECKER_JS), str(html_path)]
    if screenshot_path is not None:
        cmd += ["--screenshot", str(screenshot_path)]
    if no_browser:
        cmd.append("--no-browser")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return {
            "passed": False,
            "total_errors": 1,
            "runtime_errors": [{"line": None, "message": f"Checker crashed (rc={r.returncode}): {r.stderr[:300]}"}],
            "syntax_errors": [],
            "webgl_errors": [],
            "console_errors": [],
            "structural_errors": [],
            "blank_errors": [],
            "browser_checked": False,
            "check_ms": 0,
        }
    return json.loads(r.stdout)


def collect_error_items(check: dict) -> list[tuple[str, dict]]:
    items = []
    for cat in ("structural_errors", "syntax_errors", "runtime_errors", "webgl_errors", "console_errors", "blank_errors"):
        for err in check.get(cat, []):
            items.append((cat.replace("_errors", "").upper(), err))
    return items


def format_error_summary(check: dict) -> str:
    items = collect_error_items(check)
    if not items:
        return "None"
    parts = []
    for cat, err in items[:4]:
        ln = f"L{err['line']}" if err.get("line") else ""
        msg = str(err.get("message", "")).splitlines()[0]
        if len(msg) > 90:
            msg = msg[:87] + "..."
        parts.append(f"[{cat}{(' ' + ln) if ln else ''}] {msg}")
    if len(items) > 4:
        parts.append(f"(+{len(items) - 4} more)")
    return "; ".join(parts)


def build_repair_prompt(html: str, check: dict) -> str:
    lines = html.splitlines()
    items = collect_error_items(check)

    err_lines = []
    context_blocks = []
    seen_ctx = set()
    for idx, (cat, err) in enumerate(items, 1):
        ln = err.get("line")
        col = err.get("col")
        loc = f"line {ln}" + (f", col {col}" if col else "") if ln else "global"
        err_lines.append(f"{idx}. [{cat}] ({loc}): {err.get('message')}")
        if isinstance(ln, int) and 1 <= ln <= len(lines):
            lo = max(1, ln - 8)
            hi = min(len(lines), ln + 8)
            key = (lo, hi)
            if key not in seen_ctx:
                seen_ctx.add(key)
                snippet = "\n".join(f"{n:4d} | {lines[n - 1]}" for n in range(lo, hi + 1))
                context_blocks.append(f"Lines {lo}-{hi}:\n```html\n{snippet}\n```")

    numbered_full = "\n".join(f"{n:4d} | {lines[n - 1]}" for n in range(1, len(lines) + 1))
    ctx_section = ("\n\n### Context Around Error Lines\n" + "\n\n".join(context_blocks)) if context_blocks else ""

    return f"""Your single-file Three.js HTML page failed automated syntax/runtime/WebGL verification with {len(items)} error(s).

### Reported Errors
{chr(10).join(err_lines)}{ctx_section}

### Instructions (One Surgical Fix Pass)
- Fix ALL reported errors (and any directly related bug that would immediately fail next, such as missing `const`, mistyped variable names, `ShaderMaterial` built-in uniform redeclarations, `BufferGeometryUtils.mergeGeometries` indexed/attribute mismatches, `THREE.Color`/`Vector3` mood-interpolation bugs, or `.add(null)` calls).
- Keep internal reasoning very brief (a few short bullet points), then output ONLY surgical SEARCH/REPLACE patch blocks in this exact format:

<<<<<<< SEARCH
exact existing lines from the file (NO line numbers!)
=======
fixed replacement lines (NO line numbers!)
>>>>>>> REPLACE

- Rules for SEARCH/REPLACE blocks:
  1. Copy exact raw characters from the file into `<<<<<<< SEARCH` (include 1-3 unchanged surrounding lines so each match is unique).
  2. NEVER include line-number prefixes like ` 197 | ` inside SEARCH or REPLACE.
  3. Do NOT rewrite the whole file — output ONLY the minimal patch blocks needed to fix the errors cleanly.

### Full Line-Numbered HTML File (for reference)
```html
{numbered_full}
```
"""


def _strip_line_prefix(block: str) -> str:
    out = []
    for ln in block.splitlines():
        out.append(re.sub(r"^\s*\d+\s*\|\s?", "", ln))
    return "\n".join(out)


def apply_search_replace_patches(orig_html: str, raw_repair: str) -> tuple[str, int, list[str]]:
    """Parse and apply <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks.
    Returns (new_html, applied_count, notes)."""
    answer, _, stuck = strip_think(raw_repair)
    if stuck or not answer:
        return orig_html, 0, ["repair_stuck_in_think"]

    pattern = re.compile(
        r"<{5,9}\s*SEARCH[^\n]*\n(.*?)\n={5,9}[^\n]*\n(.*?)\n>{5,9}\s*REPLACE",
        flags=re.S,
    )
    matches = pattern.findall(answer)
    notes = []

    if not matches:
        # Fallback: did the model output a full repaired HTML file?
        full_html, _, _ = extract_html(raw_repair)
        if full_html and "<!doctype html" in full_html.lower() and "</html>" in full_html.lower():
            return full_html, 1, ["used_full_html_fallback"]
        return orig_html, 0, ["no_patch_blocks_found"]

    cur = orig_html
    applied = 0
    for idx, (search_blk, replace_blk) in enumerate(matches, 1):
        # Attempt 1: exact substring match
        if search_blk in cur:
            cur = cur.replace(search_blk, replace_blk, 1)
            applied += 1
            continue
        # Attempt 2: strip accidental ` 197 | ` line-number prefixes
        s_clean = _strip_line_prefix(search_blk)
        r_clean = _strip_line_prefix(replace_blk)
        if s_clean != search_blk and s_clean in cur:
            cur = cur.replace(s_clean, r_clean, 1)
            applied += 1
            notes.append(f"hunk_{idx}_stripped_line_nums")
            continue
        # Attempt 3: trailing-whitespace-tolerant line-by-line match
        s_lines = [l.rstrip() for l in s_clean.splitlines()]
        cur_lines = cur.splitlines()
        matched_at = -1
        for i in range(0, len(cur_lines) - len(s_lines) + 1):
            if all(cur_lines[i + k].rstrip() == s_lines[k] for k in range(len(s_lines))):
                matched_at = i
                break
        if matched_at != -1:
            rep_lines = r_clean.splitlines()
            cur_lines = cur_lines[:matched_at] + rep_lines + cur_lines[matched_at + len(s_lines):]
            cur = "\n".join(cur_lines) + ("\n" if cur.endswith("\n") else "")
            applied += 1
            notes.append(f"hunk_{idx}_ws_tolerant")
            continue
        notes.append(f"hunk_{idx}_unmatched")

    return cur, applied, notes


def build_strata_cmd(model_key: str, prompt_file: Path, is_repair: bool, args) -> list[str]:
    cfg = MODELS[model_key]
    cmd = ["caffeinate", "-dimsu", str(STRATA_CLI), cfg["subcmd"], "--bundle", cfg["bundle"]]
    cmd += cfg["repair_extra"] if is_repair else cfg["draft_extra"]
    cmd += ["--prompt-file", str(prompt_file), "--think"]
    if is_repair:
        cmd += [
            "--max-tokens", str(args.repair_max_tokens),
            "--temp", "0.2",
            "--top-k", "20",
            "--top-p", "0.95",
            "--penalty-window", "256",
            "--presence-penalty", "0.05",
            "--frequency-penalty", "0.0",
            "--seed", str(args.seed + 1),
        ]
    else:
        cmd += [
            "--max-tokens", "0",
            "--temp", str(args.temp),
            "--top-k", "20",
            "--top-p", "0.95",
            "--penalty-window", "256",
            "--presence-penalty", str(args.presence_penalty),
            "--frequency-penalty", str(args.frequency_penalty),
            "--seed", str(args.seed),
        ]
    return cmd


def process_model(model_key: str, out_dir: Path, prompt_file: Path, args) -> dict:
    cfg = MODELS[model_key]
    print("\n" + "=" * 96)
    print(f"  MODEL: {cfg['label']} ({model_key})")
    print("=" * 96)

    draft_raw_path = out_dir / f"{model_key}.draft.raw.txt"
    draft_html_path = out_dir / f"{model_key}.draft.html"
    draft_png_path = out_dir / f"{model_key}.draft.png"
    final_html_path = out_dir / f"{model_key}.html"
    final_png_path = out_dir / f"{model_key}.png"

    result = {
        "model": model_key,
        "label": cfg["label"],
        "draft_wall_sec": 0.0,
        "draft_infer_sec": None,
        "draft_telemetry": {},
        "draft_think_chars": 0,
        "draft_html_bytes": 0,
        "draft_check": None,
        "repair_needed": False,
        "repair_wall_sec": 0.0,
        "repair_infer_sec": None,
        "repair_telemetry": {},
        "repair_hunks_applied": 0,
        "repair_notes": [],
        "repair_diff_lines": 0,
        "final_check": None,
        "status": "FAIL",
    }

    # --- Stage 1: Draft generation (or load existing draft in --repair-only mode) ---
    if args.repair_only:
        src_dir = Path(args.repair_only).resolve()
        src_html = src_dir / f"{model_key}.draft.html"
        if not src_html.exists():
            src_html = src_dir / f"{model_key}.html"
        src_raw = src_dir / f"{model_key}.draft.raw.txt"
        if not src_raw.exists():
            src_raw = src_dir / f"{model_key}.raw.txt"
        if not src_html.exists():
            print(f"  [SKIP] No existing HTML found in {src_dir} for {model_key}")
            result["status"] = "MISSING"
            return result
        draft_html = src_html.read_text(encoding="utf-8", errors="replace")
        draft_html_path.write_text(draft_html, encoding="utf-8")
        if src_raw.exists():
            raw_txt = src_raw.read_text(encoding="utf-8", errors="replace")
            result["draft_telemetry"] = parse_telemetry(raw_txt)
            result["draft_infer_sec"] = infer_sec_from_tel(result["draft_telemetry"])
            _, think_chars, _ = strip_think(raw_txt)
            result["draft_think_chars"] = think_chars
        result["draft_html_bytes"] = len(draft_html.encode("utf-8"))
        print(f"  [Stage 1] Loaded existing draft from {src_html} ({result['draft_html_bytes']:,} B)")
    else:
        print(f"  [Stage 1] Generating draft ({cfg['subcmd']})...")
        cmd = build_strata_cmd(model_key, prompt_file, is_repair=False, args=args)
        rc, raw_txt, wall_sec = run_strata_streaming(cmd, draft_raw_path, quiet=args.quiet)
        result["draft_wall_sec"] = round(wall_sec, 2)
        result["draft_telemetry"] = parse_telemetry(raw_txt)
        result["draft_infer_sec"] = infer_sec_from_tel(result["draft_telemetry"])
        draft_html, think_chars, stuck = extract_html(raw_txt)
        result["draft_think_chars"] = think_chars
        if draft_html:
            draft_html_path.write_text(draft_html, encoding="utf-8")
            result["draft_html_bytes"] = len(draft_html.encode("utf-8"))
        print(
            f"\n  [Stage 1 Done] rc={rc} | wall={wall_sec:.1f}s | think={think_chars:,} chars"
            f" | html={result['draft_html_bytes']:,} B"
            + (" | STUCK_IN_THINK!" if stuck else "")
        )

    # --- Stage 2: Verify Draft ---
    if not draft_html_path.exists() or draft_html_path.stat().st_size == 0:
        result["draft_check"] = {
            "passed": False,
            "total_errors": 1,
            "structural_errors": [{"line": None, "message": "No HTML extracted from draft output"}],
            "syntax_errors": [],
            "runtime_errors": [],
            "webgl_errors": [],
            "console_errors": [],
            "blank_errors": [],
            "browser_checked": False,
            "check_ms": 0,
        }
        result["final_check"] = result["draft_check"]
        result["status"] = "NO_HTML"
        print("  [Stage 2] FAIL: No HTML generated in draft; cannot run repair pass.")
        return result

    print(f"  [Stage 2] Checking {draft_html_path.name} (syntax + headless Chrome WebGL/runtime)...")
    draft_check = run_checker(draft_html_path, draft_png_path, no_browser=args.no_browser)
    result["draft_check"] = draft_check
    d_errs = draft_check.get("total_errors", 0)
    print(
        f"  [Stage 2 Done] errors={d_errs} | browser_checked={draft_check.get('browser_checked')}"
        f" | check={draft_check.get('check_ms', 0)} ms | {format_error_summary(draft_check)}"
    )

    # Clean on first pass!
    if d_errs == 0 and (args.no_browser or draft_check.get("browser_checked")):
        shutil.copy2(draft_html_path, final_html_path)
        if draft_png_path.exists():
            shutil.copy2(draft_png_path, final_png_path)
        result["repair_needed"] = False
        result["final_check"] = draft_check
        result["status"] = "CLEAN_1SHOT"
        print(f"  [Result] CLEAN ON FIRST PASS — 0 errors found, 0.00s fix time!")
        return result

    if d_errs == 0 and not args.no_browser and not draft_check.get("browser_checked"):
        print(f"  [ERROR] Headless Chrome check failed to run: {draft_check.get('browser_skip_reason')}")
        result["status"] = "CHECKER_UNAVAILABLE"
        result["final_check"] = draft_check
        return result

    # --- Stage 3: Single Self-Repair Pass ---
    result["repair_needed"] = True
    t_fix0 = time.monotonic()
    draft_html = draft_html_path.read_text(encoding="utf-8", errors="replace")
    repair_prompt = build_repair_prompt(draft_html, draft_check)
    repair_prompt_path = out_dir / f"{model_key}.repair_prompt.txt"
    repair_raw_path = out_dir / f"{model_key}.repair.raw.txt"
    patch_diff_path = out_dir / f"{model_key}.patch.diff"
    repair_prompt_path.write_text(repair_prompt, encoding="utf-8")

    print(f"  [Stage 3] Running 1-pass self-fix on {cfg['label']} ({d_errs} error(s) found)...")
    rep_cmd = build_strata_cmd(model_key, repair_prompt_path, is_repair=True, args=args)
    rep_rc, rep_raw, _ = run_strata_streaming(rep_cmd, repair_raw_path, quiet=args.quiet)
    result["repair_telemetry"] = parse_telemetry(rep_raw)
    result["repair_infer_sec"] = infer_sec_from_tel(result["repair_telemetry"])

    fixed_html, hunks_applied, patch_notes = apply_search_replace_patches(draft_html, rep_raw)
    result["repair_hunks_applied"] = hunks_applied
    result["repair_notes"] = patch_notes
    final_html_path.write_text(fixed_html, encoding="utf-8")

    diff_lines = list(
        difflib.unified_diff(
            draft_html.splitlines(),
            fixed_html.splitlines(),
            fromfile=f"{model_key}.draft.html",
            tofile=f"{model_key}.html",
            lineterm="",
        )
    )
    patch_diff_path.write_text("\n".join(diff_lines) + ("\n" if diff_lines else ""), encoding="utf-8")
    changed = sum(1 for ln in diff_lines if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")))
    result["repair_diff_lines"] = changed

    print(f"\n  [Stage 3 Patch] rc={rep_rc} | hunks={hunks_applied} | diff_lines={changed} | notes={patch_notes}")
    print(f"  [Stage 3 Re-Check] Verifying repaired {final_html_path.name}...")
    final_check = run_checker(final_html_path, final_png_path, no_browser=args.no_browser)
    result["final_check"] = final_check
    result["repair_wall_sec"] = round(time.monotonic() - t_fix0, 2)

    f_errs = final_check.get("total_errors", 0)
    if f_errs == 0 and (args.no_browser or final_check.get("browser_checked")):
        result["status"] = "FIXED_IN_1_PASS"
    elif f_errs < d_errs:
        result["status"] = f"PARTIAL_FIX ({d_errs}->{f_errs})"
    else:
        result["status"] = f"STILL_BROKEN ({f_errs} errs)"

    inf_str = f"{result['repair_infer_sec']:.1f}s infer / " if result["repair_infer_sec"] is not None else ""
    print(
        f"  [Stage 3 Done] status={result['status']} | fix_time={inf_str}{result['repair_wall_sec']:.1f}s wall"
        f" | remaining={f_errs} | {format_error_summary(final_check)}"
    )
    return result


def write_reports(results: list[dict], out_dir: Path) -> None:
    (out_dir / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    md = [
        "# STRATA-M5 3D Eval Report (Draft + 1-Pass Syntax/Runtime Check & Fix)",
        "",
        "| Model | Draft Speed / Time | Draft Errors Found | 1-Pass Self-Fix Result | Fix Time (Infer / Wall) | Patch Size | Final Status |",
        "| :--- | :--- | :---: | :--- | :---: | :---: | :---: |",
    ]
    for r in results:
        dt = r["draft_telemetry"]
        spd = f"{dt['decode_tok_s']:.1f} tok/s" if dt.get("decode_tok_s") else "—"
        dwall = f"{r['draft_wall_sec']:.1f}s" if r["draft_wall_sec"] else (f"{r['draft_infer_sec']:.1f}s*" if r["draft_infer_sec"] else "loaded")
        d_cnt = r["draft_check"]["total_errors"] if r["draft_check"] else -1
        if not r["repair_needed"]:
            fix_res = "Not needed (clean draft)"
            fix_t = "0.0s"
            psize = "0 lines"
        else:
            f_cnt = r["final_check"]["total_errors"] if r["final_check"] else -1
            fix_res = f"{d_cnt} -> {f_cnt} errs ({r['repair_hunks_applied']} hunk(s))"
            inf_s = f"{r['repair_infer_sec']:.1f}s" if r["repair_infer_sec"] is not None else "?"
            fix_t = f"{inf_s} / {r['repair_wall_sec']:.1f}s"
            psize = f"{r['repair_diff_lines']} lines"
        md.append(
            f"| **{r['label']}** | {spd} · {dwall} | `{d_cnt}` | {fix_res} | {fix_t} | {psize} | **{r['status']}** |"
        )

    md += ["", "## Error & Repair Details", ""]
    for r in results:
        md.append(f"### {r['label']} (`{r['model']}`)")
        if r["draft_check"]:
            md.append(f"- **Draft check ({r['draft_check'].get('check_ms', 0)} ms):** {format_error_summary(r['draft_check'])}")
        if r["repair_needed"]:
            md.append(f"- **Repair hunks applied:** `{r['repair_hunks_applied']}` ({r['repair_diff_lines']} changed lines, notes: `{r['repair_notes']}`)")
            if r["final_check"]:
                md.append(f"- **Post-fix check ({r['final_check'].get('check_ms', 0)} ms):** {format_error_summary(r['final_check'])}")
        md.append(f"- **Final HTML:** `{out_dir / (r['model'] + '.html')}`")
        md.append("")

    report_md = "\n".join(md)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 96)
    print(report_md)
    print("=" * 96)
    print(f"Saved: {out_dir / 'report.md'} and {out_dir / 'report.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="STRATA-M5 3D Procedural Eval Runner with 1-Pass Check & Self-Fix")
    ap.add_argument("--prompt-file", default=str(ROOT / "evals" / "prompts" / "hogwarts.txt"))
    ap.add_argument("--models", default="inkling2,qwen180b,glm53", help="Comma-separated: inkling2,qwen180b,glm53")
    ap.add_argument("--out", default=str(ROOT / "evals" / "hogwarts_run4"), help="Output directory (never bench_data/)")
    ap.add_argument("--repair-only", default=None, help="Load existing drafts from <dir> and run check + 1-pass self-fix only")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--presence-penalty", type=float, default=0.15)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--repair-max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-browser", action="store_true", help="Syntax/static check only (skip headless Chrome)")
    ap.add_argument("--quiet", action="store_true", help="Do not mirror token stream to stdout")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    if "bench_data" in out_dir.parts:
        print("ERROR: Refusing to write inside bench_data/ (repository invariant). Use evals/.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = Path(args.prompt_file).resolve()
    if not args.repair_only and not prompt_file.exists():
        print(f"ERROR: Prompt file not found: {prompt_file}", file=sys.stderr)
        return 2

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODELS:
            print(f"ERROR: Unknown model `{mk}` (expected: {', '.join(MODELS)})", file=sys.stderr)
            return 2

    results = []
    for mk in model_keys:
        results.append(process_model(mk, out_dir, prompt_file, args))

    write_reports(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
