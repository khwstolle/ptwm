#!/usr/bin/env bash
# Extract the ptwm-core Rust API as rustdoc JSON.
#
# Requires the docs Nix shell (`nix develop .#docs`), which provides a
# nightly Rust toolchain via fenix. Rustdoc JSON is gated behind
# `-Z unstable-options --output-format json`, which stable rustc rejects.
#
# Output: docs/site/.docgen/rust.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SITE_DIR/../.." && pwd)"
OUT_DIR="$SITE_DIR/.docgen"

mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

# Sanity check: confirm we're on a nightly toolchain. The default shell
# ships stable, which silently rejects --output-format=json.
RUSTC_VERSION="$(rustc --version 2>&1 || echo '')"
if [[ $RUSTC_VERSION != *nightly* ]]; then
  echo "extract-rust.sh: nightly Rust required (got: $RUSTC_VERSION)" >&2
  echo "  Enter the docs shell: nix develop .#docs" >&2
  echo "  Or skip rust docgen: PTWM_SKIP_RUST_DOCGEN=1 pnpm run docgen" >&2
  if [[ ${PTWM_SKIP_RUST_DOCGEN:-} == "1" ]]; then
    echo '{"generator":"rustdoc","items":[],"skipped":true}' >"$OUT_DIR/rust.json"
    exit 0
  fi
  exit 1
fi

echo "extract-rust.sh: using $RUSTC_VERSION" >&2

cargo rustdoc \
  --quiet \
  --package ptwm-core \
  --target-dir "$SITE_DIR/.docgen/target" \
  -- \
  -Z unstable-options \
  --output-format json

SRC_JSON="$SITE_DIR/.docgen/target/doc/ptwm_core.json"
if [[ ! -f $SRC_JSON ]]; then
  echo "extract-rust.sh: expected rustdoc JSON at $SRC_JSON but it does not exist" >&2
  exit 2
fi

cp "$SRC_JSON" "$OUT_DIR/rust.json"
echo "extract-rust.sh: wrote $OUT_DIR/rust.json ($(wc -c <"$OUT_DIR/rust.json") bytes)" >&2
