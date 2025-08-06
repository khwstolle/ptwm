"""Tests for `CompressionConfig.from_resolved_policy`."""

from __future__ import annotations

from pathlib import Path

from ptwm import CompressionConfig
from ptwm._rust import builtin_canonical_id
from ptwm._rust.policy import PolicyFile
from ptwm.codecs import CodecId


def test_default_empty_policy_preserves_unconstrained_menu() -> None:
    """An empty resolved policy ("no constraint") must NOT collapse to
    an empty codec_menu — that would tell the Rust dispatcher to error
    on every plane. The expected behaviour is to leave `codec_menu` as
    the base config's value (None by default = full menu)."""
    rp = PolicyFile.default_empty().resolve([])
    config = CompressionConfig.from_resolved_policy(rp)
    assert config.codec_menu is None


def test_policy_allowing_only_huffman_restricts_menu(tmp_path: Path) -> None:
    huffman_id_hex = builtin_canonical_id("huffman").hex()
    pol = tmp_path / "p.toml"
    pol.write_text(
        '[allow]\nextra = ["blake3:' + huffman_id_hex + '"]\n',
        encoding="utf-8",
    )
    rp = PolicyFile.load(str(pol)).resolve([])
    config = CompressionConfig.from_resolved_policy(rp)
    assert config.codec_menu == [CodecId.Huffman]


def test_policy_allowing_multiple_codecs_includes_all(tmp_path: Path) -> None:
    rans_id = builtin_canonical_id("rans").hex()
    zstd_id = builtin_canonical_id("zstd").hex()
    huffman_id = builtin_canonical_id("huffman").hex()
    pol = tmp_path / "p.toml"
    pol.write_text(
        "[allow]\nextra = ["
        f'"blake3:{rans_id}",'
        f'"blake3:{zstd_id}",'
        f'"blake3:{huffman_id}"'
        "]\n",
        encoding="utf-8",
    )
    rp = PolicyFile.load(str(pol)).resolve([])
    config = CompressionConfig.from_resolved_policy(rp)
    # Order in the menu is dictated by the dict iteration in
    # `_BUILTIN_CODEC_NAMES` (Identity, Huffman, Rans, Zstd, ...), not by
    # the order in the policy file.
    assert set(config.codec_menu or []) == {
        CodecId.Huffman,
        CodecId.Rans,
        CodecId.Zstd,
    }


def test_unknown_canonical_ids_in_policy_are_ignored(tmp_path: Path) -> None:
    fake_id = "ff" * 32
    pol = tmp_path / "p.toml"
    pol.write_text(
        '[allow]\nextra = ["blake3:' + fake_id + '"]\n',
        encoding="utf-8",
    )
    rp = PolicyFile.load(str(pol)).resolve([])
    config = CompressionConfig.from_resolved_policy(rp)
    # Third-party / unknown ids do not appear in the built-in codec_menu;
    # they're routed via PlaneCodecRouter at decode time.
    assert config.codec_menu == []


def test_base_config_fields_are_preserved(tmp_path: Path) -> None:
    base = CompressionConfig(
        compression_threshold=0.5,
        zstd_level=11,
        threads=4,
    )
    # Use a policy that constrains the menu (so the helper actually
    # rewrites `codec_menu`); the unrelated base fields must survive.
    huffman_id_hex = builtin_canonical_id("huffman").hex()
    pol = tmp_path / "p.toml"
    pol.write_text(
        '[allow]\nextra = ["blake3:' + huffman_id_hex + '"]\n',
        encoding="utf-8",
    )
    rp = PolicyFile.load(str(pol)).resolve([])
    config = CompressionConfig.from_resolved_policy(rp, base=base)
    assert config.compression_threshold == 0.5
    assert config.zstd_level == 11
    assert config.threads == 4
    assert config.codec_menu == [CodecId.Huffman]


def test_constrained_menu_actually_filters_trial_encode(tmp_path: Path) -> None:
    """End-to-end: a policy that excludes the better codec must produce a
    different (typically larger) compressed blob than the unconstrained
    encoder. Asserting size inequality proves the menu actually reaches
    the Rust dispatcher, not just sits unused on the Python config."""
    import os

    from ptwm import Compressor, Format

    # Input dominated by a single byte value — Huffman / Zstd compress it
    # to a fraction of identity; identity passes it through verbatim.
    rng = os.urandom(128) + b"\x00" * 8192
    base = CompressionConfig(input_format=Format.BYTE)

    # Unconstrained encoder picks whichever codec wins trial-encode.
    blob_unconstrained = Compressor(base).compress(rng)

    # Force the menu to Identity only — the trial-encode loop must skip
    # Huffman/Zstd/etc. and emit a strictly larger payload.
    identity_only = CompressionConfig(
        input_format=Format.BYTE,
        codec_menu=[CodecId.Identity],
    )
    blob_identity_only = Compressor(identity_only).compress(rng)

    assert len(blob_identity_only) > len(blob_unconstrained), (
        f"Identity-only output ({len(blob_identity_only)} bytes) should be "
        f"larger than unconstrained ({len(blob_unconstrained)} bytes). If "
        f"these are equal, codec_menu isn't being honoured by the dispatcher."
    )


def test_per_group_codebook_and_order1_recognized(tmp_path: Path) -> None:
    pgc_id = builtin_canonical_id("per_group_codebook").hex()
    o1sac_id = builtin_canonical_id("order1_scale_ac").hex()
    pol = tmp_path / "p.toml"
    pol.write_text(
        f'[allow]\nextra = ["blake3:{pgc_id}","blake3:{o1sac_id}"]\n',
        encoding="utf-8",
    )
    rp = PolicyFile.load(str(pol)).resolve([])
    config = CompressionConfig.from_resolved_policy(rp)
    assert set(config.codec_menu or []) == {
        CodecId.PerGroupCodebook,
        CodecId.Order1ScaleAC,
    }
