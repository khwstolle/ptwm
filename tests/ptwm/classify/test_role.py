import pytest
from ptwm.classify import TensorClassification, TensorClassifier, TensorRole


def test_roles_have_stable_string_values():
    assert TensorRole.SCALE_BLOCK.value == "scale_block"
    assert TensorRole.SCALE_GLOBAL.value == "scale_global"
    assert TensorRole.PACKED_VALUES.value == "packed_values"
    assert TensorRole.STANDARD.value == "standard"


def test_classification_is_immutable():
    c = TensorClassification(
        role=TensorRole.SCALE_BLOCK, source="flag", pattern="*_scales"
    )
    import dataclasses

    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.role = TensorRole.STANDARD


def test_protocol_callable_signature():
    # A minimal classifier conforming to the protocol.
    class Always(TensorClassifier):
        def classify(self, _name, _dtype, _shape, _archive_keys):
            return TensorClassification(
                role=TensorRole.STANDARD, source="test", pattern=""
            )

    c = Always()
    out = c.classify("foo", "F32", (1,), ("foo",))
    assert out.role == TensorRole.STANDARD
