"""Tests for the ``ptwm chains promote`` CLI command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ptwm.classify import AuditLog
from ptwm.integrations import compress_safetensors_file
from ptwm.preprocessing._explorer import ExploreOptions
from safetensors.torch import save_file


def _make_audit_log(tmp_path: Path) -> Path:
    """Produce a real audit log with discovered chains via compress_safetensors_file."""
    from ptwm.classify import HfQuantConfigClassifier

    _GROUP_SIZE = 16
    rng = np.random.default_rng(99)
    out_features, in_features = 64, 64
    n_groups = in_features // _GROUP_SIZE
    torch_state = {
        "model.weight": torch.from_numpy(
            rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
        ),
        "model.weight_scale": torch.from_numpy(
            rng.integers(64, 192, size=(out_features, n_groups), dtype=np.uint8)
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(torch_state, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src,
        out_dir,
        classifier=HfQuantConfigClassifier(quant_algo="NVFP4", exclude_modules=()),
        explore_options=opts,
    )

    audit_path = tmp_path / "audit.cbor"
    audit_path.write_bytes(audit.to_cbor())
    return audit_path


def test_chains_promote_help():
    """``ptwm chains promote --help`` exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ptwm.cli.main", "chains", "promote", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "AUDIT_LOG" in result.stdout


def test_chains_subcommand_help():
    """``weights chains --help`` exits 0 and lists subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "ptwm.cli.main", "chains", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "promote" in result.stdout


def test_promote_missing_audit_exits_nonzero(tmp_path):
    """Pointing promote at a nonexistent file exits non-zero."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "chains",
            "promote",
            str(tmp_path / "nonexistent.cbor"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_promote_with_real_audit_exits_zero_or_one(tmp_path):
    """promote on a real audit log exits with 0 or 1 (OK or validation skip)."""
    audit_path = _make_audit_log(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "chains",
            "promote",
            str(audit_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # Exit 0 (all valid) or 1 (some skipped) are both acceptable;
    # only a crash (e.g. exit 2 from argparse) is a failure.
    assert result.returncode in {0, 1}, (
        f"Unexpected exit code {result.returncode}:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_parse_chain_handles_integer_byte_terminal():
    """_parse_chain must handle Role::IntegerByte (tag 0x05, 2 extra bytes).

    Regression for `_role_extra` table that was missing 0x05 / 0x06 / 0x08 /
    0x09 / 0xFF — chains using `ByteSplit` (which emits `IntegerByte`
    terminals) were silently misaligned during deserialisation.
    """
    from ptwm.cli.chains import _parse_chain
    from ptwm.preprocessing._chains import _bf16_split_chain

    chain = _bf16_split_chain([4, 8])
    blob = chain.to_bytes()
    # _parse_chain returns a Chain on success; raises _Truncated on failure.
    decoded = _parse_chain(blob)
    assert decoded is not None
    # Two terminals (one IntegerByte, one ExponentByte) per the BF16 chain.
    assert len(decoded.terminals) == 2


def test_parse_chain_handles_residual_terminal():
    """_parse_chain must handle Role::Residual::Xor (tag 0x06, 1 extra byte)."""
    from ptwm.cli.chains import _parse_chain

    # Build a minimal chain manually: header + Source node + Terminal with Residual::Xor role.
    # We round-trip through Chain.to_bytes via the Python builder when possible;
    # here we go straight to bytes since we just want _parse_chain to walk past
    # a Residual terminal without _Truncated.
    from ptwm.preprocessing._chains import (
        Chain,
        ChainEdge,
        ChainNode,
        TerminalRef,
        _Op,
        _source_params,
    )

    chain = Chain(
        nodes=[
            ChainNode(op_id=_Op.Source, params=_source_params([16], 17)),  # int8
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[
            ChainEdge(src_node=0, src_output_idx=0, dst_node=1, dst_input_idx=0),
        ],
        terminals=[
            # Residual::Xor → wire bytes [0x06, 0x00]
            TerminalRef(node_idx=1, output_idx=0, role=bytes([0x06, 0x00])),
        ],
    )
    blob = chain.to_bytes()
    decoded = _parse_chain(blob)
    assert decoded is not None
    assert len(decoded.terminals) == 1


def test_promote_prints_structural_report(tmp_path):
    """promote on a real audit log prints op-name structure for each chain."""
    audit_path = _make_audit_log(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "chains",
            "promote",
            str(audit_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode in {0, 1}
    # Structural report should resolve op IDs to names — at minimum every
    # real chain starts with a Source node.
    assert "Source" in result.stdout, f"missing op-name resolution: {result.stdout!r}"
    assert "wire bytes" in result.stdout
    assert "structure" in result.stdout
    # No stale path references, no fake auto-apply messaging.
    assert "python/weights/preprocessing" not in result.stdout
    assert "not yet implemented" not in result.stderr


def test_promote_apply_flag_removed():
    """The defunct --apply flag is gone; argparse rejects it."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "chains",
            "promote",
            "--apply",
            "/nonexistent",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    # argparse rejects unknown args with exit 2.
    assert result.returncode == 2
    assert "--apply" in result.stderr


def test_promote_empty_audit_exits_zero(tmp_path):
    """promote on an audit log with no discovered chains exits 0."""
    audit = AuditLog()
    audit_path = tmp_path / "empty.cbor"
    audit_path.write_bytes(audit.to_cbor())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "chains",
            "promote",
            str(audit_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
