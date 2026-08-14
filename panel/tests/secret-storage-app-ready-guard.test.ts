import { beforeEach, describe, expect, it, vi } from "vitest";

let ready = false;
const encryptString = vi.fn(() => Buffer.from("ciphertext"));
const decryptString = vi.fn(() => "plaintext");

vi.mock("electron", () => ({
  app: { isReady: () => ready },
  safeStorage: {
    isEncryptionAvailable: () => true,
    encryptString: (v: string) => encryptString(v),
    decryptString: (b: Buffer) => decryptString(b),
  },
}));

import { decryptValue, encryptValue } from "../src/main/secretStorage";

// vMLX 1.6.28 shipped hung on launch for anyone with a stored remote API key:
// the session-migration ran at module scope, decrypted that key, and the
// synchronous keychain call sat behind a macOS password prompt for "Chromium
// Safe Storage" before any window existed. These pin the guard that keeps
// safeStorage unreachable until the app name — and thus our own keychain
// item — actually resolves.
describe("secretStorage app-ready guard", () => {
  beforeEach(() => {
    ready = false;
    encryptString.mockClear();
    decryptString.mockClear();
  });

  it("does not touch the keychain to decrypt before the app is ready", () => {
    expect(decryptValue("enc:Y2lwaGVy")).toBe("");
    expect(decryptString).not.toHaveBeenCalled();
  });

  it("refuses to encrypt before ready rather than storing plaintext", () => {
    expect(() => encryptValue("sk-secret")).toThrow(/before app ready/);
    expect(encryptString).not.toHaveBeenCalled();
  });

  it("still passes through values that need no keychain access", () => {
    expect(decryptValue("")).toBe("");
    expect(decryptValue("legacy-plaintext")).toBe("legacy-plaintext");
    expect(encryptValue("")).toBe("");
    expect(decryptString).not.toHaveBeenCalled();
    expect(encryptString).not.toHaveBeenCalled();
  });

  it("uses the keychain normally once the app is ready", () => {
    ready = true;
    expect(encryptValue("sk-secret")).toBe(
      "enc:" + Buffer.from("ciphertext").toString("base64"),
    );
    expect(decryptValue("enc:Y2lwaGVy")).toBe("plaintext");
    expect(encryptString).toHaveBeenCalledOnce();
    expect(decryptString).toHaveBeenCalledOnce();
  });
});
