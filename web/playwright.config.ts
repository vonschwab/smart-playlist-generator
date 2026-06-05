import { defineConfig } from "@playwright/test";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

const repoRoot = resolve(__dirname, "..").replace(/\\/g, "/");
const webDir = __dirname.replace(/\\/g, "/").replace(/\/$/, "");
// On Windows, python3 may not exist; use python
const pythonCmd = process.platform === "win32" ? "python" : "python3";
const fakePath = `${repoRoot}/tests/fixtures/fake_worker.py`;
const workerCmd = `${pythonCmd} "${fakePath}"`;
const serverScript = `${repoRoot}/tools/serve_web.py`;

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:8771" },
  webServer: {
    // npm --prefix runs the build from the web/ directory; python serves from repo root
    command: `npm --prefix "${webDir}" run build && ${pythonCmd} "${serverScript}" --port 8771 --no-browser`,
    url: "http://127.0.0.1:8771/api/health",
    timeout: 120000,
    reuseExistingServer: false,
    env: { PG_WEB_WORKER_CMD: workerCmd },
    cwd: repoRoot,
  },
});
