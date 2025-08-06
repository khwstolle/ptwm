"""Tests for Python chain definitions and the production lookup table."""

from __future__ import annotations

import pytest
from ptwm.preprocessing._chains import (
    CHAIN_BF16_ALPHA_STABLE_NORMALIZE,
    CHAIN_BF16_HUFF_LLM_RAW,
    CHAIN_BF16_MICROSCALE_REPACK,
    CHAIN_BF16_SPHERICAL_NORMALIZE,
    CHAIN_BF16_SPLIT,
    CHAIN_BF16_SPLIT_INTDELTA,
    CHAIN_BF16_SPLIT_MZS_K1,
    CHAIN_BF16_SPLIT_MZS_K2,
    CHAIN_BF16_SPLIT_MZS_K3,
    CHAIN_BF16_SPLIT_MZS_K4,
    CHAIN_BF16_SPLIT_PREDICTOR_XOR,
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    CHAIN_F8E4M3_NIBBLE,
    CHAIN_F8E4M3_SCALE_PASSTHROUGH,
    CHAIN_F32_GLOBAL_SCALE,
    CHAIN_FP4_NIBBLE_PASSTHROUGH,
    CHAIN_FP8E4M3_NIBBLE,
    CHAIN_FP8E5M2_NIBBLE,
    CHAIN_FP16_ALPHA_STABLE_NORMALIZE,
    CHAIN_FP16_HUFF_LLM_RAW,
    CHAIN_FP16_SPHERICAL_NORMALIZE,
    CHAIN_FP32_ALPHA_STABLE_NORMALIZE,
    CHAIN_FP32_SPHERICAL_NORMALIZE,
    CHAIN_FP32_SPLIT,
    CHAIN_FP32_SPLIT_INTDELTA,
    CHAIN_FP32_SPLIT_MZS_K1,
    CHAIN_FP32_SPLIT_MZS_K2,
    CHAIN_FP32_SPLIT_MZS_K3,
    CHAIN_FP32_SPLIT_MZS_K4,
    CHAIN_FP32_SPLIT_PREDICTOR_XOR,
    CHAIN_U8_SCALE_PASSTHROUGH,
    Chain,
    ClassifierRole,
    chains_for,
)
from ptwm.utils import DType as DT

# ---------------------------------------------------------------------------
# Chain dataclasses + wire serialisation
# ---------------------------------------------------------------------------


class TestChainWireFormat:
    def test_to_bytes_nonempty(self):
        chain = CHAIN_BF16_SPLIT([4, 128])
        data = chain.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) >= 9  # at least the 9-byte header

    def test_header_counts(self):
        chain = CHAIN_BF16_SPLIT([256])
        # `to_bytes()` produces a self-contained blob:
        #   local_et_count(u32) + local_et_entries + chain_header + sections
        # The chain structure is still accessible via the legacy writer for
        # direct header introspection.
        import struct

        from ptwm.preprocessing._chains import _write_chain

        chain_only = _write_chain(chain)
        num_nodes, num_edges, num_terminals = struct.unpack_from("<BBB", chain_only)
        assert num_nodes == len(chain.nodes)
        assert num_edges == len(chain.edges)
        assert num_terminals == len(chain.terminals)

    def test_bf16_structure(self):
        chain = CHAIN_BF16_SPLIT([1024])
        assert len(chain.nodes) == 3  # Source, BitReorderIeee16, ByteSplit
        assert len(chain.edges) == 2
        assert len(chain.terminals) == 2

    def test_fp32_structure(self):
        chain = CHAIN_FP32_SPLIT([256])
        assert len(chain.nodes) == 3  # Source, BitReorderIeee32, ByteSplit
        assert len(chain.edges) == 2
        assert len(chain.terminals) == 4

    def test_fp8e4m3_nibble_structure(self):
        chain = CHAIN_FP8E4M3_NIBBLE([512])
        assert len(chain.nodes) == 3  # Source, BitReorderFp8E4M3, NibbleSplit
        assert len(chain.edges) == 2
        assert len(chain.terminals) == 2

    def test_scale_passthrough_structure(self):
        chain = CHAIN_F8E4M3_SCALE_PASSTHROUGH([32, 16])
        assert len(chain.nodes) == 2  # Source, BytePassthrough
        assert len(chain.edges) == 1
        assert len(chain.terminals) == 1

    def test_minimal_chain_roundtrip_via_pyo3(self):
        """Write chain bytes → compress with PyO3 → decode → bytes match."""
        try:
            from ptwm._rust import compress_model, decode_tensor
        except ImportError:
            pytest.skip("ptwm._core extension not built")

        raw = bytes(range(64))
        shape = [64]
        chain = CHAIN_BYTE_PASSTHROUGH_VALUE(shape)
        chain_bytes = chain.to_bytes()

        blob = compress_model(
            records=[("t", DT.UINT8.code, 0, raw, [64], "uint8", None)],
            chains_per_tensor=[[chain_bytes]],
            emit_payload_hash=False,
            emit_plane_crc=True,
        )
        recovered = decode_tensor(blob, "t")
        assert recovered == raw

    def test_bf16_chain_roundtrip_via_pyo3(self):
        """BF16 chain (fused BitReorder+ByteSplit) round-trips via PyO3."""
        try:
            from ptwm._rust import compress_model, decode_tensor
        except ImportError:
            pytest.skip("ptwm._core extension not built")

        import struct

        n_elements = 64
        raw = struct.pack(
            f"<{n_elements}e", *[float(i) / 100 for i in range(n_elements)]
        )
        shape = [n_elements]
        chain = CHAIN_BF16_SPLIT(shape)
        chain_bytes = chain.to_bytes()

        blob = compress_model(
            records=[("w", DT.BFLOAT16.code, 0, raw, [n_elements], "bfloat16", None)],
            chains_per_tensor=[[chain_bytes]],
            emit_payload_hash=True,
            emit_plane_crc=True,
        )
        recovered = decode_tensor(blob, "w")
        assert recovered == raw

    def test_fp8e4m3_nibble_roundtrip_via_pyo3(self):
        """FP8 E4M3FN nibble chain round-trips via PyO3."""
        try:
            from ptwm._rust import compress_model, decode_tensor
        except ImportError:
            pytest.skip("ptwm._core extension not built")

        n_bytes = 128
        raw = bytes(range(n_bytes))
        shape = [n_bytes]
        chain = CHAIN_FP8E4M3_NIBBLE(shape)
        chain_bytes = chain.to_bytes()

        blob = compress_model(
            records=[
                ("fp8", DT.FLOAT8_E4M3FN.code, 0, raw, [n_bytes], "float8_e4m3fn", None)
            ],
            chains_per_tensor=[[chain_bytes]],
            emit_payload_hash=False,
            emit_plane_crc=True,
        )
        recovered = decode_tensor(blob, "fp8")
        assert recovered == raw


# ---------------------------------------------------------------------------
# chains_for lookup + PRODUCTION_CHAINS
# ---------------------------------------------------------------------------


class TestChainsFor:
    def test_fp8e4m3_scale_block(self):
        result = chains_for(DT.FLOAT8_E4M3FN.code, ClassifierRole.SCALE_BLOCK)
        assert result == [CHAIN_F8E4M3_SCALE_PASSTHROUGH, CHAIN_F8E4M3_NIBBLE]

    def test_uint8_scale_block(self):
        result = chains_for(DT.UINT8.code, ClassifierRole.SCALE_BLOCK)
        assert result == [CHAIN_U8_SCALE_PASSTHROUGH]

    def test_fp32_scale_global(self):
        result = chains_for(DT.FLOAT32.code, ClassifierRole.SCALE_GLOBAL)
        assert result == [CHAIN_F32_GLOBAL_SCALE]

    def test_fp4_packed_values(self):
        result = chains_for(DT.FLOAT4_E2M1FN_X2.code, ClassifierRole.PACKED_VALUES)
        assert result == [CHAIN_FP4_NIBBLE_PASSTHROUGH]

    def test_bfloat16_standard(self):
        result = chains_for(DT.BFLOAT16.code, ClassifierRole.STANDARD)
        assert result == [
            CHAIN_BF16_SPLIT,
            CHAIN_BF16_SPLIT_INTDELTA,
            CHAIN_BF16_SPLIT_PREDICTOR_XOR,
            CHAIN_BF16_MICROSCALE_REPACK,
            CHAIN_BF16_SPLIT_MZS_K1,
            CHAIN_BF16_SPLIT_MZS_K2,
            CHAIN_BF16_SPLIT_MZS_K3,
            CHAIN_BF16_SPLIT_MZS_K4,
            CHAIN_BF16_SPHERICAL_NORMALIZE,
            CHAIN_BF16_ALPHA_STABLE_NORMALIZE,
            CHAIN_BF16_HUFF_LLM_RAW,
        ]

    def test_float16_standard_same_chain_as_bf16(self):
        result = chains_for(DT.FLOAT16.code, ClassifierRole.STANDARD)
        assert result == [
            CHAIN_BF16_SPLIT,
            CHAIN_BF16_SPLIT_INTDELTA,
            CHAIN_BF16_SPLIT_PREDICTOR_XOR,
            CHAIN_BF16_SPLIT_MZS_K1,
            CHAIN_BF16_SPLIT_MZS_K2,
            CHAIN_BF16_SPLIT_MZS_K3,
            CHAIN_BF16_SPLIT_MZS_K4,
            CHAIN_FP16_SPHERICAL_NORMALIZE,
            CHAIN_FP16_ALPHA_STABLE_NORMALIZE,
            CHAIN_FP16_HUFF_LLM_RAW,
        ]

    def test_float32_standard(self):
        result = chains_for(DT.FLOAT32.code, ClassifierRole.STANDARD)
        assert result == [
            CHAIN_FP32_SPLIT,
            CHAIN_FP32_SPLIT_INTDELTA,
            CHAIN_FP32_SPLIT_PREDICTOR_XOR,
            CHAIN_FP32_SPLIT_MZS_K1,
            CHAIN_FP32_SPLIT_MZS_K2,
            CHAIN_FP32_SPLIT_MZS_K3,
            CHAIN_FP32_SPLIT_MZS_K4,
            CHAIN_FP32_SPHERICAL_NORMALIZE,
            CHAIN_FP32_ALPHA_STABLE_NORMALIZE,
        ]

    def test_fp8e4m3fn_standard(self):
        result = chains_for(DT.FLOAT8_E4M3FN.code, ClassifierRole.STANDARD)
        assert result == [CHAIN_FP8E4M3_NIBBLE]

    def test_fp8e5m2_standard(self):
        result = chains_for(DT.FLOAT8_E5M2.code, ClassifierRole.STANDARD)
        assert result == [CHAIN_FP8E5M2_NIBBLE]

    def test_uint8_standard(self):
        result = chains_for(DT.UINT8.code, ClassifierRole.STANDARD)
        assert result == [CHAIN_BYTE_PASSTHROUGH_VALUE]

    def test_unknown_combination_returns_empty(self):
        result = chains_for(DT.INT32.code, ClassifierRole.SCALE_BLOCK)
        assert result == []


# ---------------------------------------------------------------------------
# Canonical chain definitions
# ---------------------------------------------------------------------------


class TestCanonicalChains:
    """Each production chain builder can be called and produces a valid Chain."""

    @pytest.mark.parametrize(
        ("builder", "shape"),
        [
            (CHAIN_BF16_SPLIT, [4, 128]),
            (CHAIN_FP32_SPLIT, [256]),
            (CHAIN_FP8E4M3_NIBBLE, [512]),
            (CHAIN_FP8E5M2_NIBBLE, [512]),
            (CHAIN_F8E4M3_SCALE_PASSTHROUGH, [32, 16]),
            (CHAIN_U8_SCALE_PASSTHROUGH, [128]),
            (CHAIN_F32_GLOBAL_SCALE, [1]),
            (CHAIN_FP4_NIBBLE_PASSTHROUGH, [64]),
            (CHAIN_BYTE_PASSTHROUGH_VALUE, [256]),
        ],
    )
    def test_builder_produces_chain(self, builder, shape):
        chain = builder(shape)
        assert isinstance(chain, Chain)
        assert len(chain.nodes) >= 2
        assert len(chain.terminals) >= 1

    @pytest.mark.parametrize(
        ("builder", "shape"),
        [
            (CHAIN_BF16_SPLIT, [4, 128]),
            (CHAIN_FP32_SPLIT, [256]),
            (CHAIN_FP8E4M3_NIBBLE, [512]),
            (CHAIN_F8E4M3_SCALE_PASSTHROUGH, [32, 16]),
            (CHAIN_BYTE_PASSTHROUGH_VALUE, [256]),
        ],
    )
    def test_builder_to_bytes_nonempty(self, builder, shape):
        data = builder(shape).to_bytes()
        assert isinstance(data, bytes)
        assert len(data) >= 9

    def test_f8e4m3_nibble_alias_is_same_function(self):
        assert CHAIN_F8E4M3_NIBBLE is CHAIN_FP8E4M3_NIBBLE

    def test_source_node_is_first(self):
        for builder, shape in [
            (CHAIN_BF16_SPLIT, [128]),
            (CHAIN_FP32_SPLIT, [64]),
            (CHAIN_FP8E4M3_NIBBLE, [256]),
        ]:
            chain = builder(shape)
            assert chain.nodes[0].op_id == 0x0000  # OpId::Source

    def test_shape_embedded_in_source_params(self):
        chain_small = CHAIN_BF16_SPLIT([64])
        chain_large = CHAIN_BF16_SPLIT([4096])
        # Source params differ (different shape embedded)
        assert chain_small.nodes[0].params != chain_large.nodes[0].params

    def test_different_shape_produces_different_bytes(self):
        b1 = CHAIN_FP32_SPLIT([128]).to_bytes()
        b2 = CHAIN_FP32_SPLIT([256]).to_bytes()
        assert b1 != b2

    def test_same_shape_produces_same_bytes(self):
        b1 = CHAIN_BF16_SPLIT([512]).to_bytes()
        b2 = CHAIN_BF16_SPLIT([512]).to_bytes()
        assert b1 == b2


# ---------------------------------------------------------------------------
# MantissaZeroStrip chain variants
# ---------------------------------------------------------------------------


class TestMantissaZeroStripChains:
    """Structural + dispatcher behaviour for the MZS chain variants."""

    @pytest.mark.parametrize(
        ("builder", "expected_k"),
        [
            (CHAIN_BF16_SPLIT_MZS_K1, 1),
            (CHAIN_BF16_SPLIT_MZS_K2, 2),
            (CHAIN_BF16_SPLIT_MZS_K3, 3),
            (CHAIN_BF16_SPLIT_MZS_K4, 4),
        ],
    )
    def test_bf16_mzs_chain_structure(self, builder, expected_k):
        chain = builder([128])
        # Source, BitReorderIeee16, ByteSplit, MantissaZeroStrip
        assert len(chain.nodes) == 4
        # MantissaZeroStrip is the last node and carries k as its single
        # parameter byte.
        mzs = chain.nodes[3]
        assert mzs.op_id == 0x0064  # OpId::MantissaZeroStrip
        assert mzs.params == bytes([expected_k])
        # Mantissa terminal points at the MantissaZeroStrip output; the
        # exponent terminal still points at ByteSplit's second output.
        assert chain.terminals[0].node_idx == 3
        assert chain.terminals[1].node_idx == 2

    @pytest.mark.parametrize(
        ("builder", "expected_k"),
        [
            (CHAIN_FP32_SPLIT_MZS_K1, 1),
            (CHAIN_FP32_SPLIT_MZS_K2, 2),
            (CHAIN_FP32_SPLIT_MZS_K3, 3),
            (CHAIN_FP32_SPLIT_MZS_K4, 4),
        ],
    )
    def test_fp32_mzs_chain_structure(self, builder, expected_k):
        chain = builder([128])
        # Source, BitReorderIeee32, ByteSplit(4), MantissaZeroStrip
        assert len(chain.nodes) == 4
        mzs = chain.nodes[3]
        assert mzs.op_id == 0x0064
        assert mzs.params == bytes([expected_k])
        # Four terminals: MZS output + three pass-through ByteSplit outputs.
        assert len(chain.terminals) == 4
        # The first terminal (low mantissa byte) is the MZS output.
        assert chain.terminals[0].node_idx == 3
        # The other three terminals still come straight from ByteSplit.
        for t in chain.terminals[1:]:
            assert t.node_idx == 2

    @pytest.mark.parametrize(
        "builder",
        [
            CHAIN_BF16_SPLIT_MZS_K1,
            CHAIN_BF16_SPLIT_MZS_K2,
            CHAIN_BF16_SPLIT_MZS_K3,
            CHAIN_BF16_SPLIT_MZS_K4,
            CHAIN_FP32_SPLIT_MZS_K1,
            CHAIN_FP32_SPLIT_MZS_K2,
            CHAIN_FP32_SPLIT_MZS_K3,
            CHAIN_FP32_SPLIT_MZS_K4,
        ],
    )
    def test_mzs_chain_to_bytes_nonempty(self, builder):
        data = builder([64]).to_bytes()
        assert isinstance(data, bytes)
        assert len(data) >= 9

    def test_bf16_mzs_chain_factory_produces_distinct_chains_per_k(self):
        # Each k value must produce a wire-distinct chain so the
        # dispatcher can trial-encode them separately.
        seen = {
            CHAIN_BF16_SPLIT_MZS_K1([64]).to_bytes(),
            CHAIN_BF16_SPLIT_MZS_K2([64]).to_bytes(),
            CHAIN_BF16_SPLIT_MZS_K3([64]).to_bytes(),
            CHAIN_BF16_SPLIT_MZS_K4([64]).to_bytes(),
        }
        assert len(seen) == 4

    def test_mzs_chain_round_trips_on_all_ones_bf16(self):
        """All-1.0 BF16 input has all-zero mantissa bits, so every k
        is valid. The MZS chain must round-trip exactly when forced as
        the sole candidate."""
        try:
            from ptwm._rust import compress_model, decode_tensor
        except ImportError:
            pytest.skip("ptwm._core extension not built")

        n = 256
        # BF16(1.0) = 0x3F80; LE bytes = [0x80, 0x3F]. The mantissa
        # bits are all zero, so the post-reorder mantissa byte plane
        # is all-zero, which has 8 trailing zeros — every k value
        # passes the strip invariant.
        bf16 = b"\x80\x3f" * n

        for k, builder in [
            (1, CHAIN_BF16_SPLIT_MZS_K1),
            (2, CHAIN_BF16_SPLIT_MZS_K2),
            (3, CHAIN_BF16_SPLIT_MZS_K3),
            (4, CHAIN_BF16_SPLIT_MZS_K4),
        ]:
            blob = compress_model(
                records=[("w", DT.BFLOAT16.code, 0, bf16, [n], "bfloat16", None)],
                chains_per_tensor=[[builder([n]).to_bytes()]],
                emit_payload_hash=True,
                emit_plane_crc=True,
            )
            recovered = decode_tensor(blob, "w")
            assert recovered == bf16, f"MZS round-trip failed at k={k}"

    def test_mzs_chain_fails_gracefully_on_random_bf16(self):
        """MZS chain forward fails on non-quantized data; dispatcher
        must still produce a valid blob via a non-MZS candidate."""
        try:
            from ptwm._rust import compress_model, decode_tensor
        except ImportError:
            pytest.skip("ptwm._core extension not built")

        # Random-looking BF16 bytes — most mantissa bytes will have
        # non-zero low bits, so MZS(k=4) will fail in forward and the
        # dispatcher must fall back to CHAIN_BF16_SPLIT.
        n = 256
        bf16 = bytes((i * 37 + 11) & 0xFF for i in range(n * 2))
        chains = [
            CHAIN_BF16_SPLIT([n]),
            CHAIN_BF16_SPLIT_MZS_K4([n]),
        ]
        blob = compress_model(
            records=[("w", DT.BFLOAT16.code, 0, bf16, [n], "bfloat16", None)],
            chains_per_tensor=[[c.to_bytes() for c in chains]],
            emit_payload_hash=True,
            emit_plane_crc=True,
        )
        recovered = decode_tensor(blob, "w")
        assert recovered == bf16
