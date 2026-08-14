import { app, safeStorage } from "electron";

let decryptFailureWarned = false;
let prematureAccessWarned = false;

/**
 * safeStorage picks its macOS keychain service name from the app name, which
 * is only resolved once the app is ready; called earlier it binds to Electron's
 * default "Chromium Safe Storage" — an item other Chromium apps own. Reaching
 * for someone else's item makes macOS prompt for the login password, and since
 * every safeStorage call is synchronous that prompt blocks the main thread
 * before any window exists, so the app just looks hung. Callers must therefore
 * wait for ready; this reports whether that has happened.
 */
function keychainReady(operation: string): boolean {
  if (app.isReady()) return true;
  if (!prematureAccessWarned) {
    prematureAccessWarned = true;
    console.error(
      `[SECRET] Refusing to ${operation} before app ready — this would bind ` +
        "to the wrong keychain item and hang startup. Move the call into the " +
        "app-ready path, or read the row without its secret.",
    );
  }
  return false;
}

export function encryptValue(value: string): string {
  if (!value) return value;
  // Never fall through to storing a secret as plaintext — a caller that lands
  // here is a bug to fix at the call site, not a value to silently downgrade.
  if (!keychainReady("encrypt")) {
    throw new Error("Cannot encrypt before app ready");
  }
  if (!safeStorage.isEncryptionAvailable()) return value;
  return "enc:" + safeStorage.encryptString(value).toString("base64");
}

export function decryptValue(value: string): string {
  if (!value || !value.startsWith("enc:")) return value; // legacy plaintext
  // Same contract as an ACL denial below: the secret reads as absent and the
  // user re-enters it, which beats hanging the app behind a keychain prompt.
  if (!keychainReady("decrypt")) return "";
  if (!safeStorage.isEncryptionAvailable()) return "";
  try {
    return safeStorage.decryptString(Buffer.from(value.slice(4), "base64"));
  } catch (err) {
    // Keychain ACL can deny decryption after an app update/re-sign changes the
    // code signature (mlxstudio#131/#127). Treat the secret as unrecoverable —
    // same contract as encryption-unavailable — so startup survives and the
    // user only has to re-enter the API key instead of losing every session.
    if (!decryptFailureWarned) {
      decryptFailureWarned = true;
      console.error(
        "[DB] Failed to decrypt stored secret (keychain access denied or " +
          "corrupt ciphertext); returning empty value — re-enter the API key",
        err,
      );
    }
    return "";
  }
}
