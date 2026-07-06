"""reST → Markdown converter for PTWM docstrings.

The converter targets the docstring style actually used in the PTWM
package: NumPy section headers (Parameters / Returns / Raises / Examples
are extracted upstream by Griffe), plus reST inline roles, literal
blocks, doctest blocks, lists, simple tables, and directive admonitions.

The output is GitHub-flavoured Markdown with two MDC extensions:

* `::callout{type="…"}` blocks for admonitions
* a ``` ```math ``` ``` fence for math blocks (picked up by KaTeX in
  the Nuxt renderer)

All other constructs render as plain Markdown — fenced code blocks,
inline code, pipe tables, lists, links.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Recognised reST cross-reference roles. We keep the role-prefix open
# (``:class:``, ``:py:class:``, ``:meth:`~Foo.bar```) and resolve them
# all through the same lookup.
_ROLE_NAMES: Final = frozenset(
    {
        "class",
        "func",
        "meth",
        "mod",
        "data",
        "attr",
        "obj",
        "exc",
        "const",
    }
)

# `:role:`target`` or `:domain:role:`target``.
_ROLE_RE = re.compile(
    r":(?:(?P<domain>[a-zA-Z][a-zA-Z0-9]*):)?"
    r"(?P<role>[a-zA-Z]+):`(?P<target>[^`]+)`",
)

# `:math:`expr``.
_INLINE_MATH_RE = re.compile(r":math:`([^`]+)`")

# Double-backtick inline literal → single-backtick Markdown literal.
_INLINE_LITERAL_RE = re.compile(r"``([^`]+?)``")

# reST external hyperlink: `text <url>`_ or `text <url>`__.
_EXT_LINK_RE = re.compile(r"`([^`<]+?)\s+<([^>`]+)>`_{1,2}")

# Bare URL → Markdown autolink. Conservative: stop at whitespace, < or `.
_BARE_URL_RE = re.compile(r"(?<![`<\(])(https?://[^\s<`\)]+)")

# Substitution reference (unused today).
_SUBST_RE = re.compile(r"\|([^|]+)\|")

# Footnote / citation reference (unused today).
_FOOTNOTE_RE = re.compile(r"\[(#?[A-Za-z0-9_-]+)\]_")

# Directive: ``.. name:: argument``.
_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>\s*)\.\.[ \t]+(?P<name>[a-zA-Z][a-zA-Z0-9_-]*)::"
    r"(?:[ \t]+(?P<arg>.*))?$",
)

# Bullet list: ``* item``, ``- item``, ``+ item``.
_BULLET_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[*+\-])\s+(?P<rest>.*)$")

# Numbered list: ``1. item``, ``1) item``.
_NUMBERED_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>\d+[.)])\s+(?P<rest>.*)$")

# Field list (Sphinx-style): ``:name: value``. Not inside a paragraph.
_FIELD_RE = re.compile(
    r"^(?P<indent>\s*):(?P<name>[A-Za-z][\w\s.-]*?):\s*(?P<value>.*)$"
)

# Grid table border line.
_GRID_BORDER_RE = re.compile(r"^(?P<indent>\s*)\+[-+=]+\+\s*$")

# Simple table separator line.
_SIMPLE_SEP_RE = re.compile(r"^(?P<indent>\s*)=+(\s+=+)+\s*$")

# Doctest line prefix.
_DOCTEST_RE = re.compile(r"^(?P<indent>\s*)>>>\s")

# Admonition directive name → MDC callout `type` (mirrors @nuxt/content
# defaults).
_ADMONITION_MAP: Final = {
    "note": "note",
    "seealso": "note",
    "tip": "tip",
    "hint": "tip",
    "important": "info",
    "attention": "info",
    "warning": "warning",
    "caution": "warning",
    "danger": "danger",
    "error": "danger",
}


@dataclass
class Block:
    """A tokenised block ready for rendering."""

    kind: str
    lines: list[str] = field(default_factory=list)
    # Extra payload for directive / fence / table blocks.
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderContext:
    """Per-call resolution state, immutable for the duration of a render."""

    symbol_table: dict[str, str]
    external_inventory: dict[str, str]
    scope: str


_EXTERNAL_CACHE: dict[Path, dict[str, str]] = {}


def load_external_inventory(path: Path | str | None = None) -> dict[str, str]:
    """Load the static intersphinx-style inventory from JSON.

    Cached per resolved path; the file ships next to this module.
    """
    if path is None:
        path = Path(__file__).with_name("external_refs.json")
    resolved = Path(path).resolve()
    if resolved not in _EXTERNAL_CACHE:
        payload = json.loads(resolved.read_text())
        _EXTERNAL_CACHE[resolved] = dict(payload.get("entries", {}))
    return _EXTERNAL_CACHE[resolved]


def to_markdown(
    text: str,
    *,
    symbol_table: dict[str, str] | None = None,
    external_inventory: dict[str, str] | None = None,
    scope: str = "",
) -> str:
    """Convert a reST docstring fragment to GitHub-flavoured Markdown."""
    if not text or not text.strip():
        return ""

    ctx = RenderContext(
        symbol_table=dict(symbol_table or {}),
        external_inventory=dict(external_inventory or {}),
        scope=scope,
    )

    cleaned = textwrap.dedent(text).strip("\n")
    blocks = _tokenize(cleaned.splitlines())
    rendered = [_render_block(b, ctx) for b in blocks]
    return "\n\n".join(b for b in rendered if b).strip()


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _tokenize(lines: list[str]) -> list[Block]:
    """Split a sequence of lines into block tokens."""
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # Directives (admonitions, math, code-block, version*).
        m = _DIRECTIVE_RE.match(line)
        if m:
            block, i = _consume_directive(lines, i, m)
            blocks.append(block)
            continue

        # Doctest blocks (``>>> ...``) — must precede generic paragraph
        # detection since the leading ``>>>`` triggers code semantics.
        if _DOCTEST_RE.match(line):
            block, i = _consume_doctest(lines, i)
            blocks.append(block)
            continue

        # Grid table.
        if _GRID_BORDER_RE.match(line):
            block, ni = _try_consume_grid_table(lines, i)
            if block is not None:
                blocks.append(block)
                i = ni
                continue

        # Simple table.
        if _SIMPLE_SEP_RE.match(line):
            block, ni = _try_consume_simple_table(lines, i)
            if block is not None:
                blocks.append(block)
                i = ni
                continue

        # Bullet / numbered list.
        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            block, i = _consume_list(lines, i)
            blocks.append(block)
            continue

        # Field list (only at the start of a block).
        if _FIELD_RE.match(line):
            block, i = _consume_field_list(lines, i)
            blocks.append(block)
            continue

        # Otherwise, a paragraph, possibly followed by an indented
        # literal / code block.
        block, i = _consume_paragraph(lines, i)
        blocks.append(block)

        if _is_indented_block_next(lines, i):
            literal_block, i = _consume_indented_block(lines, i)
            # If the paragraph terminated with ``::`` strip the marker.
            _strip_literal_marker(block)
            blocks.append(literal_block)

    return blocks


def _is_indented_block_next(lines: list[str], i: int) -> bool:
    """Peek ahead: is the next non-blank line indented by 4+ spaces?"""
    j = i
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return False
    next_line = lines[j]
    leading = len(next_line) - len(next_line.lstrip(" "))
    if leading < 4:
        return False
    # Don't swallow lines that start with another structural marker.
    stripped = next_line.lstrip()
    if _DIRECTIVE_RE.match(next_line):
        return False
    if _BULLET_RE.match(next_line) or _NUMBERED_RE.match(next_line):
        return False
    if _FIELD_RE.match(next_line):
        return False
    if stripped.startswith(">>> "):
        return False
    return True


def _strip_literal_marker(block: Block) -> None:
    """Strip a trailing ``::`` from the last text line of a paragraph."""
    if not block.lines:
        return
    last = block.lines[-1].rstrip()
    if last.endswith("::") and not last.endswith(":::"):
        stripped = last[:-2].rstrip()
        block.lines[-1] = stripped or ""
        # Drop a now-empty trailing line.
        if not block.lines[-1]:
            block.lines.pop()


def _consume_paragraph(lines: list[str], i: int) -> tuple[Block, int]:
    """Consume one paragraph until a blank line or a structural boundary."""
    out: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if _DIRECTIVE_RE.match(line):
            break
        if _GRID_BORDER_RE.match(line) or _SIMPLE_SEP_RE.match(line):
            break
        if i != _first_index(out) and (
            _BULLET_RE.match(line) or _NUMBERED_RE.match(line)
        ):
            break
        if i != _first_index(out) and _FIELD_RE.match(line) and not out:
            break
        out.append(line)
        i += 1
    return Block(kind="paragraph", lines=out), i


def _first_index(_out: list[str]) -> int:
    # Helper so the structural-boundary checks above can express
    # "after the first line, also bail on a bullet" without bookkeeping.
    return -1 if not _out else 0


def _consume_indented_block(lines: list[str], i: int) -> tuple[Block, int]:
    """Consume a sequence of 4+ space indented lines (with embedded blanks)."""
    # Find the common leading indent of the first non-blank line.
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return Block(kind="code", lines=[]), i

    base_indent = len(lines[i]) - len(lines[i].lstrip(" "))
    out: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # Blank lines are part of the block as long as the next
            # non-blank line is still indented.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            next_indent = len(lines[j]) - len(lines[j].lstrip(" "))
            if next_indent < base_indent:
                break
            out.append("")
            i += 1
            continue
        leading = len(line) - len(line.lstrip(" "))
        if leading < base_indent:
            break
        out.append(line[base_indent:])
        i += 1

    # Trim trailing blank lines from the block body.
    while out and not out[-1].strip():
        out.pop()
    return Block(
        kind="code", lines=out, meta={"language": _guess_code_language(out)}
    ), i


def _guess_code_language(lines: list[str]) -> str:
    """Heuristic: any block reaching this point comes from a Python docstring."""
    for ln in lines:
        if ln.lstrip().startswith(">>> "):
            return "python"
    return "python"


def _consume_doctest(lines: list[str], i: int) -> tuple[Block, int]:
    """Consume a contiguous run of ``>>> ...`` / ``... ...`` doctest lines."""
    base_indent = len(lines[i]) - len(lines[i].lstrip(" "))
    out: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        leading = len(line) - len(line.lstrip(" "))
        if leading < base_indent:
            break
        out.append(line[base_indent:])
        i += 1
    return Block(kind="code", lines=out, meta={"language": "python"}), i


def _consume_directive(
    lines: list[str],
    i: int,
    match: re.Match[str],
) -> tuple[Block, int]:
    """Consume ``.. name:: arg`` plus its indented body."""
    name = match.group("name").lower()
    arg = (match.group("arg") or "").strip()
    base_indent = len(match.group("indent") or "")
    i += 1

    body: list[str] = []
    body_indent: int | None = None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # Trailing blank line keeps the directive open as long as
            # the next non-blank line is still indented past the marker.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            next_indent = len(lines[j]) - len(lines[j].lstrip(" "))
            if next_indent <= base_indent:
                break
            body.append("")
            i += 1
            continue
        leading = len(line) - len(line.lstrip(" "))
        if leading <= base_indent:
            break
        if body_indent is None:
            body_indent = leading
        body.append(line[body_indent:])
        i += 1
    while body and not body[-1].strip():
        body.pop()
    return (
        Block(kind="directive", lines=body, meta={"name": name, "arg": arg}),
        i,
    )


def _consume_list(lines: list[str], i: int) -> tuple[Block, int]:
    """Consume a bullet / numbered list (one block, items separated by markers)."""
    out: list[str] = []
    base_indent: int | None = None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # Allow a single blank between list items.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                break
            next_line = lines[j]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if base_indent is not None and next_indent < base_indent:
                break
            if not (
                _BULLET_RE.match(next_line)
                or _NUMBERED_RE.match(next_line)
                or next_indent > (base_indent or 0)
            ):
                break
            i += 1
            out.append("")
            continue
        leading = len(line) - len(line.lstrip(" "))
        if base_indent is None:
            base_indent = leading
        if leading < base_indent:
            break
        out.append(line[base_indent:])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    return Block(kind="list", lines=out), i


def _consume_field_list(lines: list[str], i: int) -> tuple[Block, int]:
    """Consume one or more contiguous ``:name: value`` lines and their bodies."""
    out: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if not _FIELD_RE.match(line):
            # Continuation of a previous field's body.
            leading = len(line) - len(line.lstrip(" "))
            if leading == 0:
                break
        out.append(line)
        i += 1
    return Block(kind="field_list", lines=out), i


def _try_consume_grid_table(
    lines: list[str],
    i: int,
) -> tuple[Block | None, int]:
    """Try to consume a reST grid table. Returns ``None`` on a malformed run."""
    start = i
    block_lines: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if not (_GRID_BORDER_RE.match(line) or line.lstrip().startswith("|")):
            break
        block_lines.append(line)
        i += 1
    if len(block_lines) < 3 or not _GRID_BORDER_RE.match(block_lines[0]):
        return None, start
    return Block(kind="grid_table", lines=block_lines), i


def _try_consume_simple_table(
    lines: list[str],
    i: int,
) -> tuple[Block | None, int]:
    """Try to consume a reST simple table. Returns ``None`` on malformed run."""
    start = i
    sep_line = lines[i]
    block_lines: list[str] = [sep_line]
    i += 1
    saw_close = False
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        block_lines.append(line)
        if _SIMPLE_SEP_RE.match(line):
            saw_close = True
            i += 1
            # A simple table can have a header sep AND a footer sep.
            # Stop once we've matched the footer.
            break
        i += 1
    if not saw_close or len(block_lines) < 3:
        return None, start
    return Block(kind="simple_table", lines=block_lines), i


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_block(block: Block, ctx: RenderContext) -> str:
    kind = block.kind
    if kind == "paragraph":
        return _render_paragraph(block, ctx)
    if kind == "code":
        return _render_code(block)
    if kind == "directive":
        return _render_directive(block, ctx)
    if kind == "list":
        return _render_list(block, ctx)
    if kind == "field_list":
        return _render_field_list(block, ctx)
    if kind == "grid_table":
        return _render_grid_table(block, ctx)
    if kind == "simple_table":
        return _render_simple_table(block, ctx)
    return "\n".join(block.lines)


def _render_paragraph(block: Block, ctx: RenderContext) -> str:
    text = "\n".join(block.lines).strip()
    if not text:
        return ""
    return _apply_inline(text, ctx)


def _render_code(block: Block) -> str:
    body = "\n".join(block.lines)
    language = block.meta.get("language", "")
    return f"```{language}\n{body}\n```"


def _render_list(block: Block, ctx: RenderContext) -> str:
    items: list[list[str]] = []
    current: list[str] | None = None
    for raw in block.lines:
        m_bullet = _BULLET_RE.match(raw)
        m_num = _NUMBERED_RE.match(raw)
        if m_bullet or m_num or current is None:
            current = [raw]
            items.append(current)
        else:
            current.append(raw)

    rendered_items: list[str] = []
    for item_lines in items:
        first = item_lines[0]
        m = _BULLET_RE.match(first) or _NUMBERED_RE.match(first)
        if m is None:
            rendered_items.append(_apply_inline("\n".join(item_lines), ctx))
            continue
        rest = m.group("rest")
        cont = item_lines[1:]
        if cont:
            cont_text = textwrap.dedent("\n".join(cont))
            inner = _apply_inline((rest + "\n" + cont_text).strip(), ctx)
        else:
            inner = _apply_inline(rest, ctx)
        marker = (
            "-"
            if _BULLET_RE.match(first)
            else (m.group("marker") + " ")[:3].strip() + " "
        )
        if marker == "- ":
            marker = "- "
        elif not marker.endswith(" "):
            marker += " "
        # Indent continuation lines so Markdown groups them under the bullet.
        prefix = marker
        body_lines = inner.splitlines() or [""]
        out = prefix + body_lines[0]
        for ln in body_lines[1:]:
            out += "\n  " + ln
        rendered_items.append(out)
    return "\n".join(rendered_items)


def _render_field_list(block: Block, ctx: RenderContext) -> str:
    out: list[str] = []
    pending_name: str | None = None
    pending_value: list[str] = []
    for raw in block.lines:
        m = _FIELD_RE.match(raw)
        if m:
            if pending_name is not None:
                out.append(_fmt_field(pending_name, pending_value, ctx))
            pending_name = m.group("name").strip()
            value = m.group("value").strip()
            pending_value = [value] if value else []
        else:
            pending_value.append(raw)
    if pending_name is not None:
        out.append(_fmt_field(pending_name, pending_value, ctx))
    return "\n\n".join(out)


def _fmt_field(name: str, value_lines: list[str], ctx: RenderContext) -> str:
    value = textwrap.dedent("\n".join(value_lines)).strip()
    rendered = _apply_inline(value, ctx) if value else ""
    if not rendered:
        return f"**{name}**"
    return f"**{name}** — {rendered}"


def _render_directive(block: Block, ctx: RenderContext) -> str:
    name = block.meta.get("name", "")
    arg = block.meta.get("arg", "")
    body_text = "\n".join(block.lines).strip("\n")

    if name in _ADMONITION_MAP:
        callout_type = _ADMONITION_MAP[name]
        title = arg or _humanise(name)
        rendered_body = to_markdown(
            body_text,
            symbol_table=ctx.symbol_table,
            external_inventory=ctx.external_inventory,
            scope=ctx.scope,
        )
        head = f'::callout{{type="{callout_type}" title={json.dumps(title)}}}'
        return f"{head}\n{rendered_body}\n::"

    if name in ("versionadded", "versionchanged"):
        label = "Added" if name == "versionadded" else "Changed"
        title = f"{label} in {arg}" if arg else label
        rendered_body = to_markdown(
            body_text,
            symbol_table=ctx.symbol_table,
            external_inventory=ctx.external_inventory,
            scope=ctx.scope,
        )
        head = f'::callout{{type="note" title={json.dumps(title)}}}'
        return f"{head}\n{rendered_body}\n::"

    if name == "deprecated":
        title = f"Deprecated since {arg}" if arg else "Deprecated"
        rendered_body = to_markdown(
            body_text,
            symbol_table=ctx.symbol_table,
            external_inventory=ctx.external_inventory,
            scope=ctx.scope,
        )
        head = f'::callout{{type="warning" title={json.dumps(title)}}}'
        return f"{head}\n{rendered_body}\n::"

    if name == "math":
        return f"```math\n{body_text}\n```"

    if name in ("code", "code-block", "sourcecode"):
        language = arg or "python"
        return f"```{language}\n{body_text}\n```"

    # Unknown directives surface visibly so future contributors notice.
    logger.warning(
        "docstring_md: unknown directive %r in scope %r; rendering as fenced block",
        name,
        ctx.scope,
    )
    fence = name or "text"
    return f"```{fence}\n{body_text}\n```"


def _humanise(name: str) -> str:
    return name.replace("seealso", "See also").replace("-", " ").title()


def _render_grid_table(block: Block, ctx: RenderContext) -> str:
    rows: list[list[str]] = []
    current: list[str] = []
    cells: list[list[str]] = []
    header_row_idx: int | None = None
    for line in block.lines:
        if _GRID_BORDER_RE.match(line):
            if current:
                rows.append(current)
                current = []
            if "=" in line and header_row_idx is None and rows:
                header_row_idx = len(rows) - 1
            continue
        # Split on | preserving cells.
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            stripped = stripped[1:-1]
        parts = [p.strip() for p in stripped.split("|")]
        if not current:
            current = parts
        else:
            current = [
                a + " " + b if b else a for a, b in zip(current, parts, strict=False)
            ]
    if current:
        rows.append(current)
    if not rows:
        return ""
    if header_row_idx is None:
        header_row_idx = 0
    return _emit_pipe_table(rows, header_row_idx, ctx)


def _render_simple_table(block: Block, ctx: RenderContext) -> str:
    if len(block.lines) < 3:
        return "\n".join(block.lines)
    sep = block.lines[0]
    col_spans = _column_spans(sep)
    rows: list[list[str]] = []
    header_row_idx: int | None = None
    for line in block.lines[1:]:
        if _SIMPLE_SEP_RE.match(line):
            if header_row_idx is None and rows:
                header_row_idx = len(rows) - 1
            continue
        cells = _slice_by_spans(line, col_spans)
        rows.append([c.strip() for c in cells])
    if not rows:
        return ""
    if header_row_idx is None:
        header_row_idx = 0
    return _emit_pipe_table(rows, header_row_idx, ctx)


def _column_spans(sep: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_col = False
    start = 0
    for idx, ch in enumerate(sep):
        if ch == "=":
            if not in_col:
                start = idx
                in_col = True
        elif in_col:
            spans.append((start, idx))
            in_col = False
    if in_col:
        spans.append((start, len(sep)))
    return spans


def _slice_by_spans(line: str, spans: list[tuple[int, int]]) -> list[str]:
    out: list[str] = []
    for idx, (start, end) in enumerate(spans):
        cell = line[start:end] if start < len(line) else ""
        # Allow the final column to overflow.
        if idx == len(spans) - 1 and end <= len(line):
            cell = line[start:]
        out.append(cell)
    return out


def _emit_pipe_table(
    rows: list[list[str]],
    header_row_idx: int,
    ctx: RenderContext,
) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[header_row_idx]
    body = rows[header_row_idx + 1 :]
    if not body:
        body = []
    rendered_header = "| " + " | ".join(_apply_inline(c, ctx) for c in header) + " |"
    rendered_sep = "| " + " | ".join("---" for _ in header) + " |"
    rendered_body = [
        "| " + " | ".join(_apply_inline(c, ctx) for c in r) + " |" for r in body
    ]
    return "\n".join([rendered_header, rendered_sep, *rendered_body])


# ---------------------------------------------------------------------------
# Inline pass
# ---------------------------------------------------------------------------


def _apply_inline(text: str, ctx: RenderContext) -> str:
    if not text:
        return text

    # 1. Double-backtick literals (before any inline pass touches backticks).
    text = _INLINE_LITERAL_RE.sub(lambda m: f"`{m.group(1)}`", text)
    # 2. Inline math.
    text = _INLINE_MATH_RE.sub(lambda m: f"${m.group(1)}$", text)
    # 3. Cross-reference roles.
    text = _ROLE_RE.sub(lambda m: _resolve_role(m, ctx), text)
    # 4. External hyperlinks `text <url>`_.
    text = _EXT_LINK_RE.sub(
        lambda m: f"[{m.group(1).strip()}]({m.group(2).strip()})", text
    )
    # 5. Bare URLs.
    text = _BARE_URL_RE.sub(lambda m: f"<{m.group(1)}>", text)
    # 6. Footnotes / citations — render literally with a console warning.
    text = _FOOTNOTE_RE.sub(_warn_footnote, text)
    # 7. Substitutions.
    text = _SUBST_RE.sub(lambda m: f"`|{m.group(1)}|`", text)
    return text


def _warn_footnote(match: re.Match[str]) -> str:
    logger.warning(
        "docstring_md: footnote-style reference %r not supported", match.group(0)
    )
    return f"`{match.group(0)}`"


def _resolve_role(match: re.Match[str], ctx: RenderContext) -> str:
    role = match.group("role").lower()
    if role not in _ROLE_NAMES and role != "ref":
        return match.group(0)
    target = match.group("target").strip()
    display = target.lstrip("~")
    if target.startswith("~"):
        display = display.split(".")[-1]

    resolved = _lookup_symbol(target.lstrip("~"), ctx)
    if resolved:
        return f"[`{display}`]({resolved})"
    return f"`{display}`"


def _lookup_symbol(target: str, ctx: RenderContext) -> str | None:
    """Resolve a cross-reference target to an absolute URL."""
    candidates = _candidate_names(target, ctx.scope)
    for c in candidates:
        if c in ctx.symbol_table:
            return ctx.symbol_table[c]
    return _lookup_external(target, ctx.external_inventory)


def _candidate_names(target: str, scope: str) -> list[str]:
    """Enumerate fallback names: literal, scope.target, walking up scope, ptwm.target."""
    seen: list[str] = []

    def add(name: str) -> None:
        if name and name not in seen:
            seen.append(name)

    add(target)
    if scope:
        parts = scope.split(".")
        for i in range(len(parts), 0, -1):
            add(".".join(parts[:i]) + "." + target)
    add("ptwm." + target)
    return seen


def _lookup_external(target: str, inventory: dict[str, str]) -> str | None:
    """Longest-prefix match on the dotted target."""
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in inventory:
            return inventory[candidate]
        parts.pop()
    return None


# ---------------------------------------------------------------------------
# Examples helper
# ---------------------------------------------------------------------------


def fence_example(code: str, *, language: str = "python") -> str:
    """Wrap an Examples-section code body in a fenced block."""
    body = textwrap.dedent(code).strip("\n")
    return f"```{language}\n{body}\n```"


def slug_for(qualified_name: str) -> str:
    """Mirror ``normalize.mjs slugFor()`` for Python identifiers."""
    sep = "."
    segments = [
        re.sub(r"[^a-zA-Z0-9._-]", "-", seg) for seg in qualified_name.split(sep)
    ]
    if segments and segments[-1] == "index":
        segments[-1] = "index_module"
    return "/".join(segments)


def anchor_for(item_id: str) -> str:
    """Mirror ``ApiItem.vue`` slug computation for in-page anchors."""
    return re.sub(r"[^a-zA-Z0-9._-]", "-", item_id)


def url_for(language: str, qualified_name: str, *, anchor: str | None = None) -> str:
    """Build the canonical API URL for a symbol."""
    base = f"/api/{language}/{slug_for(qualified_name)}"
    if anchor:
        return f"{base}#{anchor}"
    return base


if __name__ == "__main__":  # pragma: no cover
    sample = sys.stdin.read()
    print(to_markdown(sample))
