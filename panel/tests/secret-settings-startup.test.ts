import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

function read(path: string): string {
  return readFileSync(resolve(process.cwd(), path), "utf8");
}

describe("secret setting presence checks", () => {
  it("exposes a non-decrypting settings.has IPC path", () => {
    const database = read("src/main/database.ts");
    const main = read("src/main/index.ts");
    const preload = read("src/preload/index.ts");

    expect(database).toContain("hasSetting(key: string): boolean");
    expect(database).toContain("SELECT 1 FROM settings WHERE key = ? AND value != '' LIMIT 1");
    expect(main).toContain("ipcMain.handle('settings:has'");
    expect(preload).toContain("has: (key: string) => ipcRenderer.invoke('settings:has', key)");
  });

  it("does not decrypt saved HF/Brave secrets during mount-time UI presence checks", () => {
    const startupSurfaces = [
      read("src/renderer/src/App.tsx"),
      read("src/renderer/src/components/image/ImageModelPicker.tsx"),
      read("src/renderer/src/components/sessions/DownloadTab.tsx"),
      read("src/renderer/src/components/chat/ChatSettings.tsx"),
    ].join("\n");

    expect(startupSurfaces).toContain("settings.has('hf_api_key')");
    expect(startupSurfaces).toContain("settings.has('braveApiKey')");
    expect(startupSurfaces).not.toContain("settings.get('hf_api_key')");
    expect(startupSurfaces).not.toContain("settings.get('braveApiKey')");
  });

  it("does not decrypt the HF token for local bundle session startup", () => {
    const sessions = read("src/main/sessions.ts");

    expect(sessions).toContain("function shouldPassHfTokenToEngine");
    expect(sessions).toContain("if (existsSync(value)) return false");
    expect(sessions).toContain("if (shouldPassHfTokenToEngine(config.modelPath))");
    // Assert the PROPERTY this test exists for -- that the decrypting read
    // happens inside the guard -- not the exact spelling of the read. Pinning
    // the literal line broke the moment the read was wrapped in
    // normalizeHfTokenSetting, a change that preserved the guard entirely.
    const guardIndex = sessions.indexOf(
      "if (shouldPassHfTokenToEngine(config.modelPath))",
    );
    const reads = [...sessions.matchAll(/db\.getSetting\(\s*'hf_api_key'\s*\)/g)];
    expect(reads.length).toBeGreaterThan(0);
    for (const read of reads) {
      expect(read.index).toBeGreaterThan(guardIndex);
      // and close enough to sit in that block rather than somewhere later
      expect(read.index! - guardIndex).toBeLessThan(800);
    }
  });
});
