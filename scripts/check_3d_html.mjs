#!/usr/bin/env node
/**
 * STRATA-M5 3D HTML Verifier (Syntax + Live Headless Chrome Runtime/WebGL Check)
 * Zero external npm dependencies (uses Node 22+ built-in WebSocket/fetch + local Chrome CDP).
 *
 * Usage:
 *   node scripts/check_3d_html.mjs <file.html> [--timeout-ms 12000] [--no-browser] [--screenshot <out.png>]
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';

function parseArgs(argv) {
  const opts = {
    file: null,
    timeoutMs: 12000,
    noBrowser: false,
    screenshot: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--timeout-ms' && i + 1 < argv.length) {
      opts.timeoutMs = Math.max(2000, parseInt(argv[++i], 10) || 12000);
    } else if (a === '--no-browser') {
      opts.noBrowser = true;
    } else if (a === '--screenshot' && i + 1 < argv.length) {
      opts.screenshot = argv[++i];
    } else if (!a.startsWith('-') && !opts.file) {
      opts.file = a;
    }
  }
  return opts;
}

function findChrome() {
  const candidates = [
    process.env.STRATA_CHROME_BIN,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

function extractModuleScript(html) {
  const re = /<script\b[^>]*type=["']module["'][^>]*>([\s\S]*?)<\/script>/i;
  const m = re.exec(html);
  if (!m) return null;
  const openTagEnd = m.index + m[0].indexOf('>') + 1;
  const startLine = html.slice(0, openTagEnd).split('\n').length;
  return { code: m[1], startLine };
}

function runNodeSyntaxCheck(code, startLine) {
  const tmpFile = path.join(os.tmpdir(), `strata_check_${process.pid}_${Date.now()}.mjs`);
  try {
    fs.writeFileSync(tmpFile, code, 'utf8');
    const r = spawnSync(process.execPath, ['--check', tmpFile], {
      encoding: 'utf8',
      timeout: 5000,
    });
    if (r.status === 0) return { ok: true, errors: [] };
    const stderr = (r.stderr || '').trim();
    let htmlLine = null;
    const lm = stderr.match(/:(\d+)\s*\n/);
    if (lm) {
      htmlLine = startLine + parseInt(lm[1], 10) - 1;
    }
    const cleanMsg = stderr
      .split('\n')
      .filter((l) => !l.includes('at checkSyntax') && !l.startsWith('Node.js v'))
      .join('\n')
      .replaceAll(tmpFile, `html_line_${htmlLine ?? '?'}`);
    return {
      ok: false,
      errors: [{ line: htmlLine, message: cleanMsg || 'JavaScript SyntaxError' }],
    };
  } finally {
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

function runStaticHeuristics(code, startLine) {
  const warnings = [];
  const lines = code.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const htmlLine = startLine + i;
    if (/ShaderMaterial\s*\(/.test(ln)) {
      // Checked in context below
    }
    if (/\buniform\s+mat[34]\s+(viewMatrix|projectionMatrix|modelMatrix|modelViewMatrix|normalMatrix)\b/.test(ln)) {
      warnings.push({
        line: htmlLine,
        message: 'GLSL ShaderMaterial redeclares Three.js built-in uniform (viewMatrix/projectionMatrix/normalMatrix)',
      });
    }
    if (/\.map\.repeat\b/.test(ln) && !/\.material\.map\.repeat\b/.test(ln) && /\bmesh\.map\.repeat\b/.test(ln)) {
      warnings.push({
        line: htmlLine,
        message: 'Probable mesh.map.repeat instead of mesh.material.map.repeat',
      });
    }
  }
  return warnings;
}

function startStaticServer(dirPath) {
  const mime = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
  };
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const urlPath = (req.url || '/').split('?')[0];
      if (urlPath === '/favicon.ico') {
        res.writeHead(204);
        res.end();
        return;
      }
      const rel = path.normalize(decodeURIComponent(urlPath)).replace(/^(\.\.(\/|\\|$))+/, '');
      const full = path.join(dirPath, rel);
      if (!full.startsWith(dirPath) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(full).toLowerCase();
      res.writeHead(200, {
        'Content-Type': mime[ext] || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(full).pipe(res);
    });
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

class CdpClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.ws = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      this.ws = ws;
      ws.addEventListener('open', () => resolve());
      ws.addEventListener('error', (e) => reject(new Error(`CDP WebSocket error: ${e.message || e}`)));
      ws.addEventListener('message', (ev) => {
        let msg;
        try { msg = JSON.parse(String(ev.data)); } catch { return; }
        if (msg.id && this.pending.has(msg.id)) {
          const { res, rej } = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          if (msg.error) rej(new Error(msg.error.message || JSON.stringify(msg.error)));
          else res(msg.result);
        } else if (msg.method) {
          const cbs = this.listeners.get(msg.method);
          if (cbs) for (const cb of cbs) cb(msg.params);
        }
      });
    });
  }

  on(method, cb) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(cb);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    try { this.ws?.close(); } catch {}
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function runHeadlessBrowserCheck(absPath, timeoutMs, screenshotPath) {
  const chromeBin = findChrome();
  if (!chromeBin) {
    return { checked: false, reason: 'Chrome binary not found' };
  }

  const dirPath = path.dirname(absPath);
  const baseName = path.basename(absPath);
  const { server, port: httpPort } = await startStaticServer(dirPath);
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'strata_chrome_'));

  let chromeProc = null;
  let cdp = null;

  try {
    const chromeArgs = [
      '--headless=new',
      '--remote-debugging-port=0',
      `--user-data-dir=${userDataDir}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-background-networking',
      '--disable-crash-reporter',
      '--disable-sync',
      '--disable-extensions',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--window-size=1280,800',
      'about:blank',
    ];

    chromeProc = spawn(chromeBin, chromeArgs, { stdio: ['ignore', 'pipe', 'pipe'] });

    const debugPort = await new Promise((resolve, reject) => {
      let stderrBuf = '';
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for Chrome DevTools port. stderr=${stderrBuf.slice(-300)}`)), 6000);
      chromeProc.stderr.on('data', (chunk) => {
        stderrBuf += chunk.toString('utf8');
        const m = stderrBuf.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//);
        if (m) {
          clearTimeout(timer);
          resolve(parseInt(m[1], 10));
        }
      });
      chromeProc.on('exit', (code) => {
        clearTimeout(timer);
        reject(new Error(`Chrome exited early (code=${code}): ${stderrBuf.slice(-300)}`));
      });
    });

    const listResp = await fetch(`http://127.0.0.1:${debugPort}/json/list`);
    const targets = await listResp.json();
    const pageTarget = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl) || targets[0];
    if (!pageTarget?.webSocketDebuggerUrl) {
      throw new Error('No DevTools page target found');
    }

    cdp = new CdpClient(pageTarget.webSocketDebuggerUrl);
    await cdp.connect();

    const runtimeErrors = [];
    const consoleErrors = [];
    const webglErrors = [];
    const failedRequests = [];

    await cdp.send('Runtime.enable');
    await cdp.send('Log.enable');
    await cdp.send('Page.enable');
    await cdp.send('Network.enable');

    cdp.on('Runtime.exceptionThrown', (params) => {
      const det = params.exceptionDetails || {};
      const desc = det.exception?.description || det.text || 'Uncaught exception';
      const firstLine = String(desc).split('\n')[0];
      runtimeErrors.push({
        line: typeof det.lineNumber === 'number' ? det.lineNumber + 1 : null,
        col: typeof det.columnNumber === 'number' ? det.columnNumber + 1 : null,
        message: firstLine,
      });
    });

    cdp.on('Runtime.consoleAPICalled', (params) => {
      const text = (params.args || [])
        .map((a) => (a.value !== undefined ? String(a.value) : a.description || ''))
        .join(' ')
        .trim();
      if (!text) return;
      let frameLine = null;
      let frameCol = null;
      const frames = params.stackTrace?.callFrames || [];
      const pageFrame = frames.find((f) => String(f.url || '').includes(encodeURIComponent(baseName)) || String(f.url || '').includes(baseName));
      if (pageFrame && typeof pageFrame.lineNumber === 'number') {
        frameLine = pageFrame.lineNumber + 1;
        frameCol = typeof pageFrame.columnNumber === 'number' ? pageFrame.columnNumber + 1 : null;
      }
      const isWebgl = /WebGLProgram|Shader Error|GL_INVALID|ERROR:\s*0:|program not valid|THREE\.BufferGeometryUtils/i.test(text);
      if (isWebgl) {
        webglErrors.push({ line: frameLine, col: frameCol, message: text.slice(0, 500) });
      } else if (params.type === 'error') {
        consoleErrors.push({ line: frameLine, col: frameCol, message: text.slice(0, 400) });
      }
    });

    cdp.on('Log.entryAdded', (params) => {
      const e = params.entry || {};
      if (e.level === 'error' && !String(e.url || '').includes('favicon.ico')) {
        const msg = `${e.text || 'Log error'}${e.lineNumber ? ` (line ${e.lineNumber})` : ''}`;
        if (/WebGL|Shader|GL_INVALID/i.test(msg)) webglErrors.push({ message: msg.slice(0, 400) });
        else consoleErrors.push({ message: msg.slice(0, 400) });
      }
    });

    cdp.on('Network.loadingFailed', (params) => {
      if (params.canceled) return;
      failedRequests.push({ errorText: params.errorText || 'network_failed' });
    });

    let loadFired = false;
    const loadPromise = new Promise((resolve) => {
      cdp.on('Page.loadEventFired', () => { loadFired = true; resolve(); });
    });

    const pageUrl = `http://127.0.0.1:${httpPort}/${encodeURIComponent(baseName)}`;
    await cdp.send('Page.navigate', { url: pageUrl });

    // Wait for page load + Three.js module execution + several animation frames
    await Promise.race([loadPromise, sleep(Math.min(timeoutMs, 7000))]);
    await sleep(2200);

    // Probe DOM, canvas, and click a bright mood button (Autumn Dusk / Scottish Dawn) to exercise mood lerp & lighting
    const probeRes = await cdp.send('Runtime.evaluate', {
      returnByValue: true,
      expression: `(() => {
        const c = document.querySelector('canvas');
        const btns = [...document.querySelectorAll('button, .pill, [role="button"]')];
        const clicked = [];
        for (const label of ['Autumn Dusk', 'Scottish Dawn', 'Highland Storm']) {
          const b = btns.find(x => (x.textContent || '').trim().toLowerCase().includes(label.toLowerCase()));
          if (b) { b.click(); clicked.push(label); break; }
        }
        return {
          canvasPresent: !!c,
          canvasWidth: c ? c.width : 0,
          canvasHeight: c ? c.height : 0,
          buttonCount: btns.length,
          clicked,
        };
      })()`,
    });
    const probe = probeRes?.result?.value || {};
    await sleep(1200);

    // Capture compositor screenshot to detect NaN-blackout / blank canvas & optionally save preview PNG
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
    const pngBuf = Buffer.from(shot.data, 'base64');
    if (screenshotPath) {
      fs.writeFileSync(screenshotPath, pngBuf);
    }

    // Deduplicate identical messages
    const dedup = (arr) => {
      const seen = new Set();
      return arr.filter((x) => {
        const k = `${x.line || ''}|${x.message}`;
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    };

    const rt = dedup(runtimeErrors);
    const ce = dedup(consoleErrors);
    const wg = dedup(webglErrors);

    // A rendered 1280x800 3D Hogwarts scene compresses to ~120-500+ KB PNG (especially in Dusk/Dawn);
    // a blank/NaN solid canvas with HUD text is typically < 50 KB (or < 30 KB with no HUD pills).
    const pngBytes = pngBuf.length;
    const blankThresh = (probe.clicked && probe.clicked.length > 0) ? 55000 : 30000;
    const suspectedBlank = !probe.canvasPresent || probe.canvasWidth === 0 || (pngBytes < blankThresh && rt.length === 0 && wg.length === 0);

    return {
      checked: true,
      load_fired: loadFired,
      runtime_errors: rt,
      console_errors: ce,
      webgl_errors: wg,
      failed_requests: failedRequests,
      canvas_present: !!probe.canvasPresent,
      canvas_size: [probe.canvasWidth || 0, probe.canvasHeight || 0],
      button_count: probe.buttonCount || 0,
      mood_clicked: probe.clicked || [],
      screenshot_png_bytes: pngBytes,
      suspected_blank_canvas: suspectedBlank,
    };
  } finally {
    cdp?.close();
    if (chromeProc && !chromeProc.killed) {
      try { chromeProc.kill('SIGKILL'); } catch {}
    }
    try { server.close(); } catch {}
    try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch {}
  }
}

async function main() {
  const t0 = Date.now();
  const opts = parseArgs(process.argv);
  if (!opts.file) {
    console.error('Usage: node scripts/check_3d_html.mjs <file.html> [--timeout-ms 12000] [--no-browser] [--screenshot <out.png>]');
    process.exit(2);
  }
  const absPath = path.resolve(opts.file);
  if (!fs.existsSync(absPath)) {
    console.error(JSON.stringify({ error: `File not found: ${absPath}` }));
    process.exit(2);
  }

  const html = fs.readFileSync(absPath, 'utf8');
  const completeHtml = /<!doctype\s+html/i.test(html) && /<\/html>/i.test(html);
  const hasImportMap = /<script\b[^>]*type=["']importmap["']/i.test(html);
  const mod = extractModuleScript(html);

  const structuralErrors = [];
  if (!completeHtml) structuralErrors.push({ line: null, message: 'Incomplete HTML (missing <!DOCTYPE html> or </html>)' });
  if (!hasImportMap) structuralErrors.push({ line: null, message: 'Missing <script type="importmap"> for Three.js' });
  if (!mod) structuralErrors.push({ line: null, message: 'Missing <script type="module"> block' });

  let syntaxRes = { ok: false, errors: [] };
  let staticWarnings = [];
  if (mod) {
    syntaxRes = runNodeSyntaxCheck(mod.code, mod.startLine);
    staticWarnings = runStaticHeuristics(mod.code, mod.startLine);
  }

  let browserRes = { checked: false, reason: 'skipped' };
  if (!opts.noBrowser && structuralErrors.length === 0 && syntaxRes.ok) {
    try {
      browserRes = await runHeadlessBrowserCheck(absPath, opts.timeoutMs, opts.screenshot);
    } catch (e) {
      browserRes = { checked: false, reason: String(e.message || e) };
    }
  }

  const runtimeErrors = browserRes.runtime_errors || [];
  const consoleErrors = browserRes.console_errors || [];
  const webglErrors = browserRes.webgl_errors || [];
  const blankErrors = browserRes.suspected_blank_canvas
    ? [{ line: null, message: `Canvas rendered blank/uniform (screenshot ${browserRes.screenshot_png_bytes} B) — likely NaN color/light/camera or failed mergeGeometries returning null` }]
    : [];

  const allErrors = [
    ...structuralErrors,
    ...syntaxRes.errors,
    ...runtimeErrors,
    ...webglErrors,
    ...consoleErrors,
    ...blankErrors,
  ];

  const out = {
    file: opts.file,
    complete_html: completeHtml,
    has_importmap: hasImportMap,
    has_module_script: !!mod,
    script_start_line: mod ? mod.startLine : null,
    syntax_ok: syntaxRes.ok,
    structural_errors: structuralErrors,
    syntax_errors: syntaxRes.errors,
    static_warnings: staticWarnings,
    browser_checked: !!browserRes.checked,
    browser_skip_reason: browserRes.checked ? null : (browserRes.reason || null),
    runtime_errors: runtimeErrors,
    webgl_errors: webglErrors,
    console_errors: consoleErrors,
    blank_errors: blankErrors,
    canvas_present: browserRes.canvas_present ?? null,
    canvas_size: browserRes.canvas_size ?? null,
    button_count: browserRes.button_count ?? null,
    mood_clicked: browserRes.mood_clicked ?? [],
    screenshot_png_bytes: browserRes.screenshot_png_bytes ?? null,
    total_errors: allErrors.length,
    passed: allErrors.length === 0 && (opts.noBrowser || !!browserRes.checked),
    check_ms: Date.now() - t0,
  };

  console.log(JSON.stringify(out, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ error: String(e?.stack || e) }));
  process.exit(1);
});
