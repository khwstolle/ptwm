"""Model-level ``model.ptwm.index.json`` reader/writer (mirrors safetensors)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PtwmIndex"]


@dataclass(frozen=True, slots=True)
class PtwmIndex:
    total_size: int
    compressed_total_size: int
    weight_map: dict[str, str] = field(default_factory=dict)

    def write(self, path: str | Path) -> None:
        payload = {
            "metadata": {
                "total_size": self.total_size,
                "format": "ptwm",
                "format_version": 1,
                "compressed_total_size": self.compressed_total_size,
            },
            "weight_map": dict(self.weight_map),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def read(cls, path: str | Path) -> PtwmIndex:
        payload = json.loads(Path(path).read_text())
        meta = payload.get("metadata", {})
        fmt = meta.get("format")
        if fmt != "ptwm":
            msg = (
                f"{path}: metadata.format={fmt!r}, expected 'ptwm'. "
                "This is not a .ptwm index file."
            )
            raise ValueError(msg)
        return cls(
            total_size=int(meta["total_size"]),
            compressed_total_size=int(meta.get("compressed_total_size", 0)),
            weight_map=dict(payload.get("weight_map", {})),
        )
