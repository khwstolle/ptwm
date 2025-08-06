from ptwm.classify import (
    ClassifierChain,
    TensorClassification,
    TensorRole,
)


class _AlwaysStandard:
    def classify(self, _name, _dtype, _shape, _archive_keys):
        return TensorClassification(
            role=TensorRole.STANDARD, source="standard", pattern="*"
        )


class _AlwaysNone:
    def classify(self, _name, _dtype, _shape, _archive_keys):
        return None


def test_first_classifier_wins():
    chain = ClassifierChain([_AlwaysStandard(), _AlwaysNone()])
    out = chain.classify("x", "F32", (1,), ("x",))
    assert out.source == "standard"


def test_skips_to_next_on_none():
    chain = ClassifierChain([_AlwaysNone(), _AlwaysStandard()])
    out = chain.classify("x", "F32", (1,), ("x",))
    assert out.source == "standard"


def test_empty_chain_returns_default_standard():
    chain = ClassifierChain([])
    out = chain.classify("x", "F32", (1,), ("x",))
    assert out.role is TensorRole.STANDARD
    assert out.source == "default"
    assert out.pattern == ""


def test_all_none_returns_default_standard():
    chain = ClassifierChain([_AlwaysNone(), _AlwaysNone()])
    out = chain.classify("x", "F32", (1,), ("x",))
    assert out.role is TensorRole.STANDARD
    assert out.source == "default"
