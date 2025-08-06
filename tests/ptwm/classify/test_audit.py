import cbor2
import pytest
from ptwm.classify import (
    AuditLog,
    TensorClassification,
    TensorRole,
)


def test_records_in_order_and_serialises_to_cbor():
    log = AuditLog()
    log.record("a.weight", TensorClassification(TensorRole.PACKED_VALUES, "flag", "*"))
    log.record(
        "a.weight_scale",
        TensorClassification(TensorRole.SCALE_BLOCK, "heuristic", "*_scale"),
    )
    cbor = log.to_cbor()
    # CBOR map with key "classifications" → list of entries; round-trip.
    decoded = AuditLog.from_cbor(cbor)
    entries = decoded.entries()
    assert len(entries) == 2
    assert entries[0].name == "a.weight"
    assert entries[0].role is TensorRole.PACKED_VALUES
    assert entries[1].source == "heuristic"


def test_from_cbor_rejects_corrupt_blob():
    # Truncated/garbage bytes that fail to decode as CBOR.
    with pytest.raises(ValueError, match="(failed to decode CBOR|top-level CBOR)"):
        AuditLog.from_cbor(b"\x9f\x9f")


def test_from_cbor_rejects_missing_classifications_key():
    blob = cbor2.dumps({"discovered_chains": []})
    with pytest.raises(ValueError, match="missing required 'classifications'"):
        AuditLog.from_cbor(blob)


def test_from_cbor_rejects_unknown_role():
    blob = cbor2.dumps(
        {
            "classifications": [
                {
                    "name": "a.weight",
                    "role": "not_a_real_role",
                    "source": "heuristic",
                    "pattern": "*",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="malformed classification entry at index 0"):
        AuditLog.from_cbor(blob)


def test_from_cbor_rejects_missing_entry_field():
    blob = cbor2.dumps(
        {
            "classifications": [
                {
                    "name": "a.weight",
                    "role": "standard",
                    # 'source' missing.
                    "pattern": "*",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="malformed classification entry at index 0"):
        AuditLog.from_cbor(blob)


def test_from_cbor_rejects_top_level_non_map():
    blob = cbor2.dumps([1, 2, 3])
    with pytest.raises(ValueError, match="top-level CBOR value must be a map"):
        AuditLog.from_cbor(blob)
