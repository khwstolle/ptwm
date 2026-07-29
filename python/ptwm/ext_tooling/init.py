"""ptwm ext init — scaffold a new extension project from a template."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "_templates"

SUPPORTED_LANGS = ("rust", "c", "zig", "assemblyscript")
SUPPORTED_FLAVORS = ("wasm", "native")


def init_extension(
    target_dir: Path,
    name: str,
    lang: str,
    kind: str = "plane_codec",
    flavor: str = "wasm",
    description: str | None = None,
    author_pubkey: str | None = None,
    label: str | None = None,
    canonical_id: str | None = None,
    version: str = "0.1.0",
) -> Path:
    if lang not in SUPPORTED_LANGS:
        msg = f"unsupported --lang {lang!r}; supported: {SUPPORTED_LANGS}"
        raise ValueError(msg)
    if flavor not in SUPPORTED_FLAVORS:
        msg = f"unsupported --flavor {flavor!r}; supported: {SUPPORTED_FLAVORS}"
        raise ValueError(msg)

    # Most kinds share one generic single-buffer scaffold per language. A
    # kind can override it with a `_kind_<kind>/` subdirectory when its ABI
    # doesn't fit that shape (e.g. delta_scheme's two-buffer base+target /
    # base+delta signature) — see `_templates/rust/_kind_delta_scheme/`.
    kind_template_dir = _TEMPLATES_DIR / lang / f"_kind_{kind}"
    template_dir = (
        kind_template_dir if kind_template_dir.is_dir() else _TEMPLATES_DIR / lang
    )
    if not template_dir.is_dir():
        msg = f"template not found: {template_dir}"
        raise FileNotFoundError(msg)

    context = {
        "name": name,
        "version": version,
        "description": description or f"{name} extension",
        "author_pubkey": author_pubkey or ("ed25519:" + "0" * 64),
        "canonical_id": canonical_id or ("blake3:" + "0" * 64),
        "label": label or f"local.dev.{name}",
        "kind": kind,
        "flavor": flavor,
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    for src in _iter_template_files(template_dir):
        rel = src.relative_to(template_dir)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            content = src.read_text(encoding="utf-8")
            for key, value in context.items():
                content = content.replace("{{ " + key + " }}", value)
            dst.write_text(content, encoding="utf-8")
    return target_dir


def _iter_template_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # `_kind_*/` subdirectories are per-kind overrides selected
        # explicitly in `init_extension` — never pulled in by the generic
        # (parent) template's own iteration.
        if any(part.startswith("_kind_") for part in p.relative_to(root).parts):
            continue
        yield p
