//! OCI artifact install source: `oci://<registry>/<path>:<tag>`.
//!
//! Pulls the manifest, expects a single layer carrying the bundle
//! `.tar.zst`, and unpacks into the staging directory.

use std::str::FromStr;

use oci_client::{Client, Reference, client::ClientConfig, secrets::RegistryAuth};
use tempfile::TempDir;
use tokio::runtime::Runtime;

use crate::extension::ExtensionError;

pub fn stage_oci(ref_str: &str) -> Result<TempDir, ExtensionError> {
    let staging =
        tempfile::tempdir().map_err(|e| ExtensionError::ManifestParse(format!("tempdir: {e}")))?;

    let rt = Runtime::new().map_err(|e| ExtensionError::ManifestParse(format!("tokio rt: {e}")))?;

    let bytes = rt.block_on(async {
        let reference = Reference::from_str(ref_str)
            .map_err(|e| ExtensionError::ManifestParse(format!("oci ref {ref_str}: {e}")))?;
        let client = Client::new(ClientConfig::default());
        let auth = RegistryAuth::Anonymous;
        let (manifest, _digest) = client
            .pull_image_manifest(&reference, &auth)
            .await
            .map_err(|e| ExtensionError::ManifestParse(format!("pull manifest: {e}")))?;
        let layer = manifest
            .layers
            .first()
            .ok_or_else(|| ExtensionError::ManifestParse("oci manifest has no layers".into()))?
            .clone();
        let mut blob: Vec<u8> = Vec::new();
        client
            .pull_blob(&reference, &layer, &mut blob)
            .await
            .map_err(|e| ExtensionError::ManifestParse(format!("pull blob: {e}")))?;
        Ok::<_, ExtensionError>(blob)
    })?;

    let dec = zstd::Decoder::new(&bytes[..])
        .map_err(|e| ExtensionError::ManifestParse(format!("zstd: {e}")))?;
    tar::Archive::new(dec)
        .unpack(staging.path())
        .map_err(|e| ExtensionError::ManifestParse(format!("untar: {e}")))?;
    Ok(staging)
}
