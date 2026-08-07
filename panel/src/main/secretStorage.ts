import { safeStorage } from "electron";

let decryptFailureWarned = false;

export function encryptValue(value: string): string {
  if (!value || !safeStorage.isEncryptionAvailable()) return value;
  return "enc:" + safeStorage.encryptString(value).toString("base64");
}

export function decryptValue(value: string): string {
  if (!value || !value.startsWith("enc:")) return value; // legacy plaintext
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
