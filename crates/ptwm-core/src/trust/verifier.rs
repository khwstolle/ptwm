//! Decides whether an ExtensionTableEntry is *trusted* given the active
//! keyring and the bundled-key status. Built-ins (whose canonical_id was
//! derived from BUILTIN_PUBKEY) are trusted by construction.

use super::keyring::{Keyring, TrustEntry};
use super::signature::PublicKey;
use crate::extension::{Attestation, ExtensionError, ExtensionTableEntry, builtins::is_builtin};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrustVerdict {
    Trusted,
    Untrusted { reason: String },
}

pub struct Verifier {
    keyring: Keyring,
}

impl Verifier {
    pub fn new(keyring: Keyring) -> Self {
        Self { keyring }
    }

    pub fn keyring(&self) -> &Keyring {
        &self.keyring
    }

    /// Decide whether `entry` is trusted.
    ///
    /// `signed_material` is the bytes that the contribution's signature
    /// covers (typically: the canonical-CBOR encoding of the manifest +
    /// the binary). `pubkey` is the author key claimed by the
    /// contribution. The caller is responsible for supplying both
    /// consistently — for in-tree built-ins, `pubkey` doesn't matter
    /// because `is_builtin(canonical_id)` short-circuits first.
    pub fn verify(
        &self,
        entry: &ExtensionTableEntry,
        signed_material: &[u8],
        pubkey: &PublicKey,
    ) -> Result<TrustVerdict, ExtensionError> {
        // 1. Built-ins are trusted by construction.
        if is_builtin(&entry.canonical_id) {
            return Ok(TrustVerdict::Trusted);
        }

        // 2. Exact contribution-hash pin in the keyring wins.
        if self.keyring.trusts_contribution_id(&entry.canonical_id) {
            return Ok(TrustVerdict::Trusted);
        }

        // 3. Author pubkey must be trusted.
        let trust = match self.keyring.trusts_pubkey(pubkey) {
            Some(t) => t,
            None => {
                return Ok(TrustVerdict::Untrusted {
                    reason: format!("author key {} not in keyring", pubkey.fingerprint()),
                });
            }
        };

        // 4. Capability constraints apply, if any.
        if let TrustEntry::KeyWithCapabilityConstraints {
            allowed_capabilities,
            ..
        } = trust
        {
            for key in entry.capabilities.0.keys() {
                if !allowed_capabilities.contains(key) {
                    return Ok(TrustVerdict::Untrusted {
                        reason: format!("capability '{key}' not in trust scope"),
                    });
                }
            }
        }

        // 5. The signature must verify against the author key.
        let Attestation::PgpSignature(sig) = &entry.attestation;
        if sig.is_empty() {
            return Ok(TrustVerdict::Untrusted {
                reason: "no signature attached".into(),
            });
        }
        pubkey.verify(signed_material, sig)?;

        Ok(TrustVerdict::Trusted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{
        Attestation, CanonicalId, CapabilityMap, Kind, Lifecycle, builtin_canonical_id,
        capability::CapabilityValue, table::FLAVOR_WASM,
    };
    use crate::trust::keyring::TrustEntry as KE;
    use crate::trust::signature::SecretKey;

    fn entry_signed_by(
        sk: &SecretKey,
        canonical_id: CanonicalId,
        signed_material: &[u8],
        capabilities: CapabilityMap,
    ) -> ExtensionTableEntry {
        ExtensionTableEntry {
            canonical_id,
            human_label: "io.example.test".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities,
            attestation: Attestation::PgpSignature(sk.sign(signed_material)),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn builtin_is_trusted_without_keyring_entries() {
        let v = Verifier::new(Keyring::default());
        let id = builtin_canonical_id("huffman");
        let entry = ExtensionTableEntry {
            canonical_id: id,
            human_label: "io.ptwm.builtin.huffman@x".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        };
        let pk = PublicKey([0; 32]);
        let v = v.verify(&entry, b"", &pk).unwrap();
        assert_eq!(v, TrustVerdict::Trusted);
    }

    #[test]
    fn untrusted_author_yields_untrusted() {
        let sk = SecretKey::generate();
        let id = CanonicalId::derive(&sk.public().0, "thing", "1.0.0");
        let entry = entry_signed_by(&sk, id, b"material", CapabilityMap::new());
        let v = Verifier::new(Keyring::default());
        let pk = sk.public();
        match v.verify(&entry, b"material", &pk).unwrap() {
            TrustVerdict::Untrusted { reason } => assert!(reason.contains("not in keyring")),
            _ => panic!("expected Untrusted"),
        }
    }

    #[test]
    fn trusted_author_yields_trusted() {
        let sk = SecretKey::generate();
        let id = CanonicalId::derive(&sk.public().0, "thing", "1.0.0");
        let entry = entry_signed_by(&sk, id, b"material", CapabilityMap::new());

        let mut k = Keyring::default();
        k.entries.push(KE::AuthorKey {
            pubkey: format!("ed25519:{}", hex::encode(sk.public().0)),
            label: Some("test".into()),
        });
        let v = Verifier::new(k);
        let verdict = v.verify(&entry, b"material", &sk.public()).unwrap();
        assert_eq!(verdict, TrustVerdict::Trusted);
    }

    #[test]
    fn tampered_signature_yields_untrusted_via_err() {
        let sk = SecretKey::generate();
        let id = CanonicalId::derive(&sk.public().0, "thing", "1.0.0");
        let mut entry = entry_signed_by(&sk, id, b"material", CapabilityMap::new());
        // Flip a byte in the signature.
        let Attestation::PgpSignature(sig) = &mut entry.attestation;
        sig[0] ^= 0xFF;

        let mut k = Keyring::default();
        k.entries.push(KE::AuthorKey {
            pubkey: format!("ed25519:{}", hex::encode(sk.public().0)),
            label: None,
        });
        let v = Verifier::new(k);
        let res = v.verify(&entry, b"material", &sk.public());
        assert!(res.is_err(), "tampered signature should bubble verify err");
    }

    #[test]
    fn capability_outside_scope_yields_untrusted() {
        let sk = SecretKey::generate();
        let id = CanonicalId::derive(&sk.public().0, "thing", "1.0.0");
        let mut caps = CapabilityMap::new();
        caps.set("network", CapabilityValue::Bool(true));
        let entry = entry_signed_by(&sk, id, b"material", caps);

        let mut k = Keyring::default();
        k.entries.push(KE::KeyWithCapabilityConstraints {
            pubkey: format!("ed25519:{}", hex::encode(sk.public().0)),
            allowed_capabilities: vec!["determinism".into(), "hardware_class".into()],
            label: None,
        });
        let v = Verifier::new(k);
        match v.verify(&entry, b"material", &sk.public()).unwrap() {
            TrustVerdict::Untrusted { reason } => assert!(reason.contains("network")),
            _ => panic!("expected Untrusted"),
        }
    }

    #[test]
    fn contribution_hash_pin_overrides_pubkey_check() {
        let sk = SecretKey::generate();
        let id = CanonicalId::derive(&sk.public().0, "thing", "1.0.0");
        // Empty signature — would normally fail.
        let entry = ExtensionTableEntry {
            canonical_id: id,
            human_label: "io.example.test".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        };
        let mut k = Keyring::default();
        k.entries.push(KE::ContributionHash {
            canonical_id: id.to_string(),
            label: Some("pinned".into()),
        });
        let v = Verifier::new(k);
        // Wrong pubkey, missing signature — but the hash pin wins.
        let v_out = v.verify(&entry, b"", &PublicKey([0xFF; 32])).unwrap();
        assert_eq!(v_out, TrustVerdict::Trusted);
    }
}
