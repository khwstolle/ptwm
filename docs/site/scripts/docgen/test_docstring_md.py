"""Unit tests for the reST → Markdown converter."""

from __future__ import annotations

import textwrap

import pytest
from docstring_md import (
    anchor_for,
    fence_example,
    slug_for,
    to_markdown,
    url_for,
)

SYMBOLS = {
    "ptwm.preprocessing.Chain": "/api/python/ptwm/preprocessing#python:ptwm.preprocessing.Chain",
    "ptwm.preprocessing": "/api/python/ptwm/preprocessing",
    "ptwm.random_access.TensorIndex": "/api/python/ptwm/random_access#python:ptwm.random_access.TensorIndex",
    "ptwm.delta.encode": "/api/python/ptwm/delta#python:ptwm.delta.encode",
}

EXTERNAL = {
    "torch.Tensor": "https://docs.pytorch.org/docs/stable/tensors.html",
    "transformers.PreTrainedModel.from_pretrained": "https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained",
    "transformers": "https://huggingface.co/docs/transformers",
    "pathlib.Path": "https://docs.python.org/3/library/pathlib.html#pathlib.Path",
}


def _render(text: str, *, scope: str = "") -> str:
    return to_markdown(
        text,
        symbol_table=SYMBOLS,
        external_inventory=EXTERNAL,
        scope=scope,
    )


# ---------- inline rules ---------------------------------------------------


def test_inline_literal_collapses_to_single_backtick():
    out = _render("Path to a ``.ptwm`` container.")
    assert out == "Path to a `.ptwm` container."


def test_class_role_resolves_internal_symbol():
    out = _render(
        ":class:`TensorIndex` opens a bundle.",
        scope="ptwm.random_access",
    )
    assert (
        "[`TensorIndex`](/api/python/ptwm/random_access#python:ptwm.random_access.TensorIndex)"
        in out
    )


def test_class_role_falls_back_to_inline_code_when_unknown():
    out = _render(":class:`Unresolved` not in the table.")
    assert "`Unresolved`" in out
    assert "[`Unresolved`]" not in out


def test_class_role_with_tilde_shows_last_segment_only():
    out = _render(
        ":meth:`~transformers.PreTrainedModel.from_pretrained` loads weights.",
    )
    assert (
        "[`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained)"
        in out
    )


def test_external_role_resolves_via_inventory_longest_prefix():
    out = _render(":class:`torch.Tensor`")
    assert "[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html)" in out


def test_inline_math_renders_dollar_delimited():
    out = _render("Bound by :math:`O(n \\log n)`.")
    assert "$O(n \\log n)$" in out


def test_external_hyperlink_reST_to_markdown():
    out = _render("See `the spec <https://example.org/spec>`_.")
    assert "[the spec](https://example.org/spec)" in out


def test_bare_url_autolinks():
    out = _render("See https://example.org for details.")
    assert "<https://example.org>" in out


# ---------- block rules ----------------------------------------------------


def test_literal_block_with_double_colon_marker():
    out = _render(
        textwrap.dedent("""\
        Use the helper like so::

            import ptwm
            ptwm.compress(...)
        """)
    )
    assert "```python" in out
    assert "import ptwm" in out
    assert "Use the helper like so" in out
    assert "::" not in out.splitlines()[0]


def test_bare_indented_code_block_no_marker():
    out = _render(
        textwrap.dedent("""\
        Random-access loader.

            idx = TensorIndex.open("model.ptwm")
            weight = idx.get_tensor("layer.weight")
        """)
    )
    assert "```python" in out
    assert 'idx = TensorIndex.open("model.ptwm")' in out


def test_doctest_block_becomes_python_fence():
    out = _render(
        textwrap.dedent("""\
        Example.

            >>> idx = TensorIndex.open("model.ptwm")
            >>> idx.names()
            ['layer0', 'layer1']
        """)
    )
    assert "```python" in out
    assert ">>> idx = TensorIndex.open" in out


def test_code_block_directive_with_language():
    out = _render(
        textwrap.dedent("""\
        Foo.

        .. code-block:: toml

            [tool.example]
            key = "value"
        """)
    )
    assert "```toml" in out
    assert "[tool.example]" in out


def test_note_admonition_becomes_callout():
    out = _render(
        textwrap.dedent("""\
        .. note::

            This is a note.
        """)
    )
    assert '::callout{type="note"' in out
    assert "This is a note." in out
    assert out.endswith("::")


def test_warning_admonition_maps_to_warning_callout():
    out = _render(
        textwrap.dedent("""\
        .. warning::

            Be careful.
        """)
    )
    assert 'type="warning"' in out


def test_versionadded_directive_becomes_callout():
    out = _render(
        textwrap.dedent("""\
        .. versionadded:: 1.2

            New parameter ``foo``.
        """)
    )
    assert "Added in 1.2" in out
    assert 'type="note"' in out


def test_deprecated_directive_becomes_warning_callout():
    out = _render(
        textwrap.dedent("""\
        .. deprecated:: 2.0

            Use :func:`new_func` instead.
        """)
    )
    assert "Deprecated since 2.0" in out
    assert 'type="warning"' in out


def test_math_block_directive_emits_math_fence():
    out = _render(
        textwrap.dedent("""\
        .. math::

            H(X) = -\\sum_i p_i \\log p_i
        """)
    )
    assert "```math" in out
    assert "H(X) =" in out


def test_bullet_list_passes_through():
    out = _render(
        textwrap.dedent("""\
        Surfaces:

        * :func:`patch_transformers` installs one patch.
        * :func:`materialize_hf_cache` walks the cache.
        """)
    )
    lines = out.splitlines()
    assert any(ln.startswith("- ") or ln.startswith("* ") for ln in lines)
    assert "`patch_transformers`" in out
    assert "`materialize_hf_cache`" in out


def test_numbered_list_renders_as_markdown():
    out = _render(
        textwrap.dedent("""\
        Steps:

        1. First step.
        2. Second step.
        """)
    )
    assert "1. First step." in out
    assert "2. Second step." in out


def test_simple_table_becomes_pipe_table():
    out = _render(
        textwrap.dedent("""\
        Codec table.

        ========  =====  =======
        Codec     Ratio  Notes
        ========  =====  =======
        huffman   0.85   integer
        rans      0.84   skewed
        ========  =====  =======
        """)
    )
    assert "| Codec" in out
    assert "| --- |" in out
    assert "| huffman" in out
    assert "| rans" in out


# ---------- end-to-end ----------------------------------------------------


def test_tensor_index_docstring_renders_with_code_block_and_xref():
    docstring = textwrap.dedent("""\
        Random-access loader for ``.ptwm`` containers.

        :class:`TensorIndex` opens a compressed bundle and decompresses individual
        tensors on demand. The index is tiny (a manifest of name → offset/length
        records), so opening a bundle and fetching one tensor touches only that
        tensor's bytes.

            idx = TensorIndex.open("model.safetensors.ptwm")
            weight = idx.get_tensor("layer_5.weight")
            for name, tensor in idx.stream_tensors(names_of_interest):
                ...
        """)
    out = _render(docstring, scope="ptwm.random_access")
    assert "`.ptwm`" in out
    assert "[`TensorIndex`](/api/python/ptwm/random_access" in out
    assert "```python" in out
    assert 'idx = TensorIndex.open("model.safetensors.ptwm")' in out


def test_empty_input_returns_empty_string():
    assert _render("") == ""
    assert _render("   \n\n   ") == ""


def test_summary_only_no_blocks():
    out = _render("A one-line summary.")
    assert out == "A one-line summary."


# ---------- helpers --------------------------------------------------------


def test_fence_example_wraps_python_default():
    out = fence_example("x = 1\ny = 2")
    assert out.startswith("```python")
    assert out.endswith("```")


def test_slug_for_escapes_index_leaf():
    assert slug_for("ptwm_core.index") == "ptwm_core/index_module"
    assert slug_for("ptwm.preprocessing.Chain") == "ptwm/preprocessing/Chain"


def test_anchor_for_strips_unsafe_chars():
    assert (
        anchor_for("python:ptwm.preprocessing.Chain")
        == "python-ptwm.preprocessing.Chain"
    )


def test_url_for_combines_slug_and_anchor():
    assert (
        url_for("python", "ptwm.preprocessing", anchor="x")
        == "/api/python/ptwm/preprocessing#x"
    )
    assert url_for("python", "ptwm.preprocessing") == "/api/python/ptwm/preprocessing"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
