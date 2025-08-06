//! Ed25519 sign/verify wrappers.

use ed25519_dalek::{
    SIGNATURE_LENGTH, Signature, Signer, SigningKey, Verifier as DalekVerifier, VerifyingKey,
};

use crate::extension::ExtensionError;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PublicKey(pub [u8; 32]);

impl PublicKey {
    pub fn from_hex(s: &str) -> Result<Self, ExtensionError> {
        let s = s.strip_prefix("ed25519:").unwrap_or(s);
        let bytes = hex::decode(s)
            .map_err(|e| ExtensionError::AttestationParse(format!("bad pubkey hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(ExtensionError::AttestationParse(format!(
                "pubkey is {} bytes, expected 32",
                bytes.len()
            )));
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        Ok(Self(out))
    }

    pub fn fingerprint(&self) -> String {
        let h = blake3::hash(&self.0);
        hex::encode(&h.as_bytes()[..16])
    }

    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), ExtensionError> {
        if signature.len() != SIGNATURE_LENGTH {
            return Err(ExtensionError::AttestationParse(format!(
                "signature is {} bytes, expected {}",
                signature.len(),
                SIGNATURE_LENGTH
            )));
        }
        let mut sig_bytes = [0u8; SIGNATURE_LENGTH];
        sig_bytes.copy_from_slice(signature);
        let sig = Signature::from_bytes(&sig_bytes);
        let vk = VerifyingKey::from_bytes(&self.0)
            .map_err(|e| ExtensionError::AttestationParse(e.to_string()))?;
        vk.verify(message, &sig)
            .map_err(|_| ExtensionError::AttestationParse("signature verification failed".into()))
    }
}

pub struct SecretKey(SigningKey);

impl SecretKey {
    pub fn generate() -> Self {
        use rand_core::OsRng;
        Self(SigningKey::generate(&mut OsRng))
    }

    /// Reconstruct a `SecretKey` from its raw 32-byte seed.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self(SigningKey::from_bytes(bytes))
    }

    pub fn public(&self) -> PublicKey {
        PublicKey(self.0.verifying_key().to_bytes())
    }

    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        self.0.sign(message).to_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_then_verify_succeeds() {
        let sk = SecretKey::generate();
        let pk = sk.public();
        let msg = b"hello ptwm";
        let sig = sk.sign(msg);
        pk.verify(msg, &sig).expect("verify should succeed");
    }

    #[test]
    fn tampered_message_fails_verify() {
        let sk = SecretKey::generate();
        let pk = sk.public();
        let sig = sk.sign(b"original");
        let res = pk.verify(b"tampered", &sig);
        assert!(res.is_err());
    }

    #[test]
    fn tampered_signature_fails_verify() {
        let sk = SecretKey::generate();
        let pk = sk.public();
        let msg = b"some message";
        let mut sig = sk.sign(msg);
        sig[0] ^= 0xAA;
        let res = pk.verify(msg, &sig);
        assert!(res.is_err());
    }

    #[test]
    fn rejects_wrong_length_pubkey_hex() {
        assert!(PublicKey::from_hex("ed25519:00").is_err());
        assert!(PublicKey::from_hex("ed25519:zz").is_err());
    }

    #[test]
    fn accepts_pubkey_with_or_without_prefix() {
        let hex = "00".repeat(32);
        let pk1 = PublicKey::from_hex(&hex).unwrap();
        let pk2 = PublicKey::from_hex(&format!("ed25519:{hex}")).unwrap();
        assert_eq!(pk1, pk2);
    }

    #[test]
    fn fingerprint_is_stable() {
        let pk = PublicKey([0x11; 32]);
        let fp1 = pk.fingerprint();
        let fp2 = pk.fingerprint();
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 32); // 16 bytes hex-encoded
    }
}
