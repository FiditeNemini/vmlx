import { beforeEach, describe, expect, it, vi } from "vitest";

// mlxstudio#131 / #127: after an app update/re-sign the macOS keychain ACL can
// deny safeStorage decryption for previously stored secrets. The throw used to
// propagate out of DatabaseManager.mapSessionRow / getSetting and crash the app
// at startup; the only user "fix" was deleting app-support (losing every
// session). Contract under test: a throwing decryptString degrades to an empty
// secret (same as encryption-unavailable) and never throws.

const safeStorageMock = vi.hoisted(() => ({
  isEncryptionAvailable: vi.fn(() => true),
  encryptString: vi.fn((value: string) => Buffer.from(value, "utf8")),
  decryptString: vi.fn((buf: Buffer) => buf.toString("utf8")),
}));

vi.mock("electron", () => ({ safeStorage: safeStorageMock }));

import { decryptValue, encryptValue } from "../src/main/secretStorage";

describe("safeStorage decrypt failure resilience (mlxstudio#131/#127)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    safeStorageMock.isEncryptionAvailable.mockReturnValue(true);
    safeStorageMock.decryptString.mockImplementation((buf: Buffer) =>
      buf.toString("utf8"),
    );
  });

  it("round-trips a secret when the keychain works", () => {
    const stored = encryptValue("sk-secret-key");
    expect(stored.startsWith("enc:")).toBe(true);
    expect(decryptValue(stored)).toBe("sk-secret-key");
  });

  it("returns empty string instead of throwing when decryptString throws", () => {
    const stored = encryptValue("sk-secret-key");
    safeStorageMock.decryptString.mockImplementation(() => {
      throw new Error(
        "Error while decrypting the ciphertext provided to safeStorage.decryptString.",
      );
    });
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    try {
      // Crashed the app at startup before the decryptValue try/catch.
      expect(decryptValue(stored)).toBe("");
      // Repeated failures (every session row) must not spam the log.
      expect(decryptValue(stored)).toBe("");
      expect(consoleError).toHaveBeenCalledTimes(1);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("keeps the legacy plaintext and encryption-unavailable contracts", () => {
    expect(decryptValue("plaintext-key")).toBe("plaintext-key");
    safeStorageMock.isEncryptionAvailable.mockReturnValue(false);
    expect(decryptValue("enc:whatever")).toBe("");
  });
});
