import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Borg } from "borg";

import { createDemoServerApp } from "../demo/server/src/app.js";
import { createLiveBridge } from "../demo/server/src/live.js";

const dataDir = mkdtempSync(join(tmpdir(), "borg-dist-smoke-"));
const live = createLiveBridge();
const borg = await Borg.open({
  dataDir,
  tracer: live.tracer,
  onStreamAppend: live.onStreamAppend,
  liveExtraction: false,
});
const { app } = createDemoServerApp({ borg, live });

try {
  const response = await app.request("/api/state");

  if (response.status !== 200) {
    throw new Error(`/api/state returned ${response.status}`);
  }
} finally {
  live.broadcaster.closeAll();
  await borg.close();
  rmSync(dataDir, { recursive: true, force: true });
}
