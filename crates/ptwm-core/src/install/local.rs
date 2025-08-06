//! Local install sources: directory and `.tar.zst` archive.

use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::extension::ExtensionError;

pub fn stage_local_dir(dir: &Path) -> Result<TempDir, ExtensionError> {
    let staging =
        tempfile::tempdir().map_err(|e| ExtensionError::ManifestParse(format!("tempdir: {e}")))?;
    let target = staging.path().to_path_buf();
    let opts = fs_extra::dir::CopyOptions {
        content_only: true,
        ..Default::default()
    };
    fs_extra::dir::copy(dir, &target, &opts)
        .map_err(|e| ExtensionError::ManifestParse(format!("copy: {e}")))?;
    Ok(staging)
}

pub fn stage_local_archive(path: &Path) -> Result<TempDir, ExtensionError> {
    let staging =
        tempfile::tempdir().map_err(|e| ExtensionError::ManifestParse(format!("tempdir: {e}")))?;
    let f = std::fs::File::open(path)
        .map_err(|e| ExtensionError::ManifestParse(format!("open archive: {e}")))?;
    let dec =
        zstd::Decoder::new(f).map_err(|e| ExtensionError::ManifestParse(format!("zstd: {e}")))?;
    tar::Archive::new(dec)
        .unpack(staging.path())
        .map_err(|e| ExtensionError::ManifestParse(format!("untar: {e}")))?;
    Ok(staging)
}

pub fn _resolve_target_root(_: PathBuf) -> PathBuf {
    // Placeholder kept so this file has more than one symbol; the
    // installer logic in mod.rs computes targets directly.
    PathBuf::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn stage_local_dir_copies_into_tempdir() {
        let src = tempdir().unwrap();
        fs::write(src.path().join("manifest.toml"), "x").unwrap();
        let staging = stage_local_dir(src.path()).unwrap();
        assert!(staging.path().join("manifest.toml").exists());
    }

    #[test]
    fn stage_local_archive_unpacks_tar_zst() {
        // Build a tiny tar.zst with one file.
        let dir = tempdir().unwrap();
        let archive_path = dir.path().join("bundle.tar.zst");
        {
            let f = fs::File::create(&archive_path).unwrap();
            let enc = zstd::Encoder::new(f, 3).unwrap();
            let mut tar = tar::Builder::new(enc);
            let mut header = tar::Header::new_gnu();
            let payload = b"file contents".to_vec();
            header.set_size(payload.len() as u64);
            header.set_cksum();
            tar.append_data(&mut header, "manifest.toml", &payload[..])
                .unwrap();
            tar.into_inner().unwrap().finish().unwrap();
        }
        let staging = stage_local_archive(&archive_path).unwrap();
        let extracted = staging.path().join("manifest.toml");
        assert!(extracted.exists());
    }
}
