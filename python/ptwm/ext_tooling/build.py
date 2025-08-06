"""ptwm ext build — detect language, run the right toolchain, place the .wasm."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path


class BuildError(RuntimeError):
    """Raised when an `ext build` invocation fails."""


def detect_language(cwd: Path) -> str:
    """Detect the build language of an extension project.

    Parameters
    ----------
    cwd:
        Root directory of the extension project.

    Returns
    -------
    str
        One of ``"rust"``, ``"c"``, ``"zig"``, or ``"assemblyscript"``.

    Raises
    ------
    BuildError
        When no recognised project layout is found.
    """
    if (cwd / "Cargo.toml").exists():
        return "rust"
    if (cwd / "build.zig").exists():
        return "zig"
    if (cwd / "Makefile").exists() and (cwd / "src" / "lib.c").exists():
        return "c"
    if (cwd / "package.json").exists():
        return "assemblyscript"
    msg = f"could not detect a supported language layout under {cwd}"
    raise BuildError(msg)


def build_extension(cwd: Path, release: bool = True) -> Path:
    """Build the extension in *cwd* and return the path to the produced .wasm.

    Parameters
    ----------
    cwd:
        Root directory of the extension project (must contain ``manifest.toml``).
    release:
        When ``True`` (default), build with optimizations enabled.

    Returns
    -------
    Path
        Path to the ``.wasm`` file placed next to ``manifest.toml``.

    Raises
    ------
    BuildError
        When language detection or the underlying toolchain invocation fails.
    """
    lang = detect_language(cwd)
    if lang == "rust":
        return _build_rust(cwd, release)
    if lang == "c":
        return _build_c(cwd)
    if lang == "zig":
        return _build_zig(cwd, release)
    if lang == "assemblyscript":
        return _build_assemblyscript(cwd)
    msg = f"unsupported language: {lang}"
    raise BuildError(msg)


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        raise BuildError(msg)


def _build_rust(cwd: Path, release: bool) -> Path:
    cmd = ["cargo", "build", "--target", "wasm32-wasip1"]
    if release:
        cmd.append("--release")
    _run(cmd, cwd)
    sub = "release" if release else "debug"
    src = next((cwd / "target" / "wasm32-wasip1" / sub).glob("*.wasm"))
    name = _manifest_name(cwd)
    dst = cwd / f"{name}.wasm"
    shutil.copyfile(src, dst)
    return dst


def _build_c(cwd: Path) -> Path:
    _run(["make"], cwd)
    name = _manifest_name(cwd)
    src = cwd / "build" / f"{name}.wasm"
    if not src.exists():
        # Some Makefiles emit a fixed filename; pick the first .wasm.
        try:
            src = next((cwd / "build").glob("*.wasm"))
        except StopIteration as e:
            msg = f"no .wasm under {cwd / 'build'}"
            raise BuildError(msg) from e
    dst = cwd / f"{name}.wasm"
    shutil.copyfile(src, dst)
    return dst


def _build_zig(cwd: Path, release: bool) -> Path:
    cmd = ["zig", "build"]
    if release:
        cmd.append("-Doptimize=ReleaseFast")
    _run(cmd, cwd)
    name = _manifest_name(cwd)
    try:
        src = next((cwd / "zig-out" / "bin").glob("*.wasm"))
    except StopIteration as e:
        msg = "zig build produced no .wasm under zig-out/bin"
        raise BuildError(msg) from e
    dst = cwd / f"{name}.wasm"
    shutil.copyfile(src, dst)
    return dst


def _build_assemblyscript(cwd: Path) -> Path:
    _run(
        ["npx", "asc", "src/index.ts", "--outFile", "build/lib.wasm", "--optimize"],
        cwd,
    )
    name = _manifest_name(cwd)
    dst = cwd / f"{name}.wasm"
    shutil.copyfile(cwd / "build" / "lib.wasm", dst)
    return dst


def _manifest_name(cwd: Path) -> str:
    """Read ``[bundle].name`` from ``manifest.toml`` in *cwd*."""
    src = (cwd / "manifest.toml").read_text(encoding="utf-8")
    data = tomllib.loads(src)
    return data["bundle"]["name"]  # type: ignore[no-any-return]
