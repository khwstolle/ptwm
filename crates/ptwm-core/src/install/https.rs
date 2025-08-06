//! HTTPS install source. Fetches a `.tar.zst` from the given URL,
//! unpacks into a staging directory, returns the TempDir.

use std::io::Read;
use std::time::Duration;

use tempfile::TempDir;

use crate::extension::ExtensionError;

const MAX_BUNDLE_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB safety cap

pub fn stage_https(url: &str) -> Result<TempDir, ExtensionError> {
    let staging =
        tempfile::tempdir().map_err(|e| ExtensionError::ManifestParse(format!("tempdir: {e}")))?;
    let resp = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(60))
        .build()
        .get(url)
        .call()
        .map_err(|e| ExtensionError::ManifestParse(format!("GET {url}: {e}")))?;

    let mut buf = Vec::new();
    let mut reader = resp.into_reader().take(MAX_BUNDLE_BYTES);
    reader
        .read_to_end(&mut buf)
        .map_err(|e| ExtensionError::ManifestParse(format!("read body: {e}")))?;

    // Decompress + untar.
    let dec = zstd::Decoder::new(&buf[..])
        .map_err(|e| ExtensionError::ManifestParse(format!("zstd: {e}")))?;
    tar::Archive::new(dec)
        .unpack(staging.path())
        .map_err(|e| ExtensionError::ManifestParse(format!("untar: {e}")))?;
    Ok(staging)
}
