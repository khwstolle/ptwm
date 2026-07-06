#!/usr/bin/env python3
"""Extract the PTWM Python API into the normalized ApiItem schema.

Run with the docs dependency group active:

    uv run --group docs python docs/site/scripts/docgen/extract-python.py

Writes JSON to ``docs/site/.docgen/python.json``. The schema is consumed by
``docs/site/scripts/docgen/normalize.mjs`` and is intentionally kept simple —
the JS normalizer is the single source of truth for the on-page rendering
schema.

Targets griffe 2.x.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import griffe  # type: ignore[reportMissingTypeStubs]

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = REPO_ROOT / "python"
OUT_PATH = Path(__file__).resolve().parents[2] / ".docgen" / "python.json"

PACKAGE = "ptwm"

# Members starting with a single underscore are treated as private and skipped,
# unless they are dunder methods that are documented (rare for this library).
PUBLIC_DUNDERS = {"__init__", "__call__", "__enter__", "__exit__", "__iter__"}


def griffe_version() -> str:
    try:
        return version("griffe")
    except PackageNotFoundError:
        return "unknown"


def is_public(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return name in PUBLIC_DUNDERS
    return not name.startswith("_")


def is_public_path(qualified_name: str) -> bool:
    """`True` when every segment of *qualified_name* is public.

    Filters out items whose canonical path threads through a private
    submodule (e.g. ``ptwm._config.CompressionConfig``). Public re-exports
    of the same item (``ptwm.CompressionConfig``) survive because the
    emitter rewrites their qualified name to the alias path before this
    check runs.
    """
    return all(is_public(seg) for seg in qualified_name.split("."))


def relpath(file: str | None) -> str:
    if not file:
        return ""
    p = Path(file)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def parse_docstring(
    obj: griffe.Object,
) -> tuple[
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[str],
]:
    """Return (summary, description, params, returns, raises, examples)."""
    ds = obj.docstring
    if ds is None:
        return "", "", [], None, [], []

    try:
        parsed = ds.parse("numpy")
    except Exception:
        return ds.value.strip().split("\n\n", 1)[0], "", [], None, [], []

    summary = ""
    description = ""
    params: list[dict[str, Any]] = []
    returns: dict[str, Any] | None = None
    raises: list[dict[str, Any]] = []
    examples: list[str] = []

    body_chunks: list[str] = []
    for section in parsed:
        kind = (
            section.kind.value if hasattr(section.kind, "value") else str(section.kind)
        )
        if kind == "text":
            text = section.value if isinstance(section.value, str) else ""
            if not summary:
                parts = text.strip().split("\n\n", 1)
                summary = parts[0].replace("\n", " ").strip()
                if len(parts) > 1:
                    body_chunks.append(parts[1].strip())
            else:
                body_chunks.append(text.strip())
        elif kind == "parameters":
            for p in section.value or []:
                params.append(
                    {
                        "name": p.name,
                        "type": str(p.annotation) if p.annotation else None,
                        "default": str(p.default) if p.default else None,
                        "description": (p.description or "").strip() or None,
                        "kind": _param_kind(getattr(p, "kind", None)),
                    }
                )
        elif kind == "returns":
            items = section.value or []
            if items:
                head = items[0]
                returns = {
                    "type": str(head.annotation) if head.annotation else None,
                    "description": (head.description or "").strip() or None,
                }
        elif kind == "raises":
            for r in section.value or []:
                raises.append(
                    {
                        "type": str(r.annotation) if r.annotation else "Exception",
                        "description": (r.description or "").strip() or None,
                    }
                )
        elif kind == "examples":
            value = section.value
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        examples.append(entry[1])
                    elif isinstance(entry, str):
                        examples.append(entry)
                    elif hasattr(entry, "value"):
                        examples.append(getattr(entry, "value", ""))
            elif isinstance(value, str):
                examples.append(value)

    description = "\n\n".join(c for c in body_chunks if c)
    return summary, description, params, returns, raises, examples


def _param_kind(k: Any) -> str | None:
    if k is None:
        return None
    name = getattr(k, "name", str(k))
    mapping = {
        "var_positional": "var-positional",
        "var_keyword": "var-keyword",
        "positional_or_keyword": "positional",
        "positional_only": "positional",
        "keyword_only": "keyword",
        # griffe 1.x compatibility
        "VAR_POSITIONAL": "var-positional",
        "VAR_KEYWORD": "var-keyword",
        "POSITIONAL_OR_KEYWORD": "positional",
        "POSITIONAL_ONLY": "positional",
        "KEYWORD_ONLY": "keyword",
    }
    return mapping.get(name, "positional")


def normalize_signature(obj: griffe.Object) -> str:
    if obj.kind is griffe.Kind.FUNCTION:
        fn = cast(griffe.Function, obj)
        rendered_params = []
        for p in fn.parameters:
            piece = p.name
            kind_name = getattr(getattr(p, "kind", None), "name", "")
            if kind_name in ("var_positional", "VAR_POSITIONAL"):
                piece = f"*{p.name}"
            elif kind_name in ("var_keyword", "VAR_KEYWORD"):
                piece = f"**{p.name}"
            if p.annotation is not None:
                piece += f": {p.annotation}"
            if p.default is not None:
                piece += f" = {p.default}"
            rendered_params.append(piece)
        ret = f" -> {fn.returns}" if fn.returns is not None else ""

        # In griffe v2, methods are FUNCTION kind with parent.kind == CLASS.
        parent_is_class = (
            obj.parent is not None and obj.parent.kind is griffe.Kind.CLASS
        )
        prefix = "def "
        if parent_is_class and getattr(obj, "is_classmethod", False):
            prefix = "@classmethod\ndef "
        elif parent_is_class and getattr(obj, "is_staticmethod", False):
            prefix = "@staticmethod\ndef "
        return f"{prefix}{obj.name}({', '.join(rendered_params)}){ret}"

    if obj.kind is griffe.Kind.CLASS:
        cls = cast(griffe.Class, obj)
        bases_list = getattr(cls, "bases", None) or []
        bases = ", ".join(str(b) for b in bases_list)
        head = f"class {obj.name}"
        if bases:
            head += f"({bases})"
        return head + ":"

    if obj.kind is griffe.Kind.ATTRIBUTE:
        attr = cast(griffe.Attribute, obj)
        ann = f": {attr.annotation}" if attr.annotation else ""
        val = f" = {attr.value}" if attr.value else ""
        return f"{obj.name}{ann}{val}"

    if obj.kind is griffe.Kind.MODULE:
        return f"module {obj.canonical_path}"

    return obj.name


def classify(obj: griffe.Object) -> str | None:
    if obj.kind is griffe.Kind.MODULE:
        return "module"
    if obj.kind is griffe.Kind.CLASS:
        return "type"
    if obj.kind is griffe.Kind.FUNCTION:
        parent_is_class = (
            obj.parent is not None and obj.parent.kind is griffe.Kind.CLASS
        )
        return "method" if parent_is_class else "function"
    if obj.kind is griffe.Kind.ATTRIBUTE:
        attr = cast(griffe.Attribute, obj)
        if attr.name.isupper():
            return "constant"
        return "property"
    if obj.kind is griffe.Kind.TYPE_ALIAS:
        return "type"
    return None


def resolve_alias(obj: griffe.Object) -> griffe.Object | None:
    """Walk an alias chain to its concrete target.

    Skips chains that point outside the loaded package — those raise
    ``AliasResolutionError`` on ``.target`` access in griffe v2.
    """
    seen: set[str] = set()
    current = obj
    while getattr(current, "is_alias", False):
        path = getattr(current, "path", None) or getattr(
            current, "canonical_path", None
        )
        if path in seen:
            return None
        if path is not None:
            seen.add(path)
        try:
            current = current.target  # may raise AliasResolutionError
        except Exception:
            return None
        if current is None:
            return None
    return current


def emit(
    obj: griffe.Object,
    items: list[dict[str, Any]],
    parent_path: str,
    *,
    forced_qualified_name: str | None = None,
    seen: set[str] | None = None,
    seen_targets: set[str] | None = None,
) -> None:
    if seen is None:
        seen = set()
    if seen_targets is None:
        seen_targets = set()

    if not is_public(obj.name):
        return

    if obj.is_alias:
        target = resolve_alias(obj)
        if target is None:
            return
        # Skip aliases whose target lives outside our package — covers stdlib
        # re-exports like `ptwm.PackageNotFoundError` from importlib.metadata.
        target_path = getattr(target, "canonical_path", "") or ""
        if not target_path.startswith(f"{PACKAGE}.") and target_path != PACKAGE:
            return
        # Skip aliases that point upward / sideways within the package — e.g.
        # `import ptwm` inside `ptwm.cli.main` would otherwise embed every
        # public symbol under `ptwm.cli.main.ptwm.*`. A legitimate re-export
        # always lives in the alias's own namespace subtree.
        public_path = obj.path
        alias_parent = public_path.rsplit(".", 1)[0] if "." in public_path else ""
        if alias_parent and not target_path.startswith(f"{alias_parent}."):
            return
        # Public re-exports document at the alias path (`ptwm.X`), never at
        # the target's canonical (`ptwm._config.X`). Use `obj.path` — Griffe's
        # `canonical_path` on an Alias resolves through to the target, which
        # for our purposes is the private location we want to hide.
        if not is_public_path(public_path):
            return
        emit(
            target,
            items,
            parent_path,
            forced_qualified_name=public_path,
            seen=seen,
            seen_targets=seen_targets,
        )
        return

    qualified_name = forced_qualified_name or obj.canonical_path
    if not is_public_path(qualified_name):
        return
    if qualified_name in seen:
        return
    seen.add(qualified_name)

    # Dedup non-module symbols re-exported under several public aliases — a
    # class re-exported in both `ptwm` and `ptwm.core` would otherwise document
    # itself twice. Modules are not deduped because each module page is the
    # navigation target for that module.
    if obj.kind is not griffe.Kind.MODULE:
        canonical_target = getattr(obj, "canonical_path", "") or ""
        if canonical_target:
            if canonical_target in seen_targets:
                return
            seen_targets.add(canonical_target)

    kind = classify(obj)
    if kind is None:
        return

    summary, description, params, returns, raises, examples = parse_docstring(obj)

    file_path = obj.filepath
    source = None
    if file_path is not None:
        source = {
            "file": relpath(
                file_path.as_posix()
                if hasattr(file_path, "as_posix")
                else str(file_path)
            ),
            "line": obj.lineno or 1,
        }

    item = {
        "id": f"python:{qualified_name}",
        "language": "python",
        "kind": kind,
        "name": qualified_name.rsplit(".", 1)[-1],
        "qualifiedName": qualified_name,
        "signature": normalize_signature(obj),
        "summary": summary,
        "description": description,
        "parameters": params or None,
        "returns": returns,
        "raises": raises or None,
        "examples": examples or None,
        "source": source,
        "parent": parent_path or None,
    }
    item = {k: v for k, v in item.items() if v not in (None, "", [])}
    items.append(item)

    if obj.kind in (griffe.Kind.MODULE, griffe.Kind.CLASS):
        for child_name, child in obj.members.items():
            if not is_public(child_name):
                continue
            # When the parent was emitted at an alias path (forced_qualified_name
            # was set), each child inherits that prefix so the whole subtree
            # lives under the public path.
            child_qn = (
                f"{qualified_name}.{child.name}"
                if forced_qualified_name is not None
                else None
            )
            emit(
                child,
                items,
                qualified_name,
                forced_qualified_name=child_qn,
                seen=seen,
                seen_targets=seen_targets,
            )


def main() -> int:
    module = griffe.load(
        PACKAGE,
        search_paths=[str(SOURCE_DIR)],
        docstring_parser="numpy",
    )

    items: list[dict[str, Any]] = []
    emit(module, items, "")

    payload = {
        "generator": "griffe",
        "generatorVersion": griffe_version(),
        "package": PACKAGE,
        "items": items,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"extracted {len(items)} python api items → {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
