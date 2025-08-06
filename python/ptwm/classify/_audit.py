"""Per-tensor classification audit log; serialises to CBOR."""

from __future__ import annotations

from dataclasses import dataclass

import cbor2

from ._role import TensorClassification, TensorRole

__all__ = ["AuditEntry", "AuditLog", "DiscoveredChain"]


@dataclass(frozen=True, slots=True)
class AuditEntry:
    name: str
    role: TensorRole
    source: str
    pattern: str


@dataclass(frozen=True, slots=True)
class DiscoveredChain:
    """A chain candidate that was explored for a given dtype + role pair."""

    chain: bytes  # serialised Chain wire bytes
    dtype_code: int
    role: str  # TensorRole value string
    n_candidates_tried: int
    sample_tensor_name: str


class AuditLog:
    """Ordered list of (name, classification) pairs, with optional discovery section."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._discovered: list[DiscoveredChain] = []

    def record(self, name: str, c: TensorClassification) -> None:
        self._entries.append(
            AuditEntry(name=name, role=c.role, source=c.source, pattern=c.pattern)
        )

    def record_discovered(
        self,
        chain_bytes: bytes,
        dtype_code: int,
        role: TensorRole,
        n_candidates_tried: int,
        sample_tensor_name: str,
    ) -> None:
        """Record a chain candidate explored during generative search."""
        self._discovered.append(
            DiscoveredChain(
                chain=chain_bytes,
                dtype_code=dtype_code,
                role=role.value,
                n_candidates_tried=n_candidates_tried,
                sample_tensor_name=sample_tensor_name,
            )
        )

    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def discovered_chains(self) -> tuple[DiscoveredChain, ...]:
        return tuple(self._discovered)

    def to_cbor(self) -> bytes:
        payload = {
            "classifications": [
                {
                    "name": e.name,
                    "role": e.role.value,
                    "source": e.source,
                    "pattern": e.pattern,
                }
                for e in self._entries
            ],
            "discovered_chains": [
                {
                    "chain": d.chain,
                    "dtype_code": d.dtype_code,
                    "role": d.role,
                    "n_candidates_tried": d.n_candidates_tried,
                    "sample_tensor_name": d.sample_tensor_name,
                }
                for d in self._discovered
            ],
        }
        return cbor2.dumps(payload)

    @classmethod
    def from_cbor(cls, blob: bytes) -> AuditLog:
        try:
            payload = cbor2.loads(blob)
        except cbor2.CBORDecodeError as exc:
            msg = f"AuditLog: failed to decode CBOR blob: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(payload, dict):
            msg = (
                f"AuditLog: top-level CBOR value must be a map, "
                f"got {type(payload).__name__}"
            )
            raise ValueError(msg)
        if "classifications" not in payload:
            msg = "AuditLog: missing required 'classifications' key"
            raise ValueError(msg)

        log = cls()
        for i, raw in enumerate(payload["classifications"]):
            try:
                log._entries.append(
                    AuditEntry(
                        name=raw["name"],
                        role=TensorRole(raw["role"]),
                        source=raw["source"],
                        pattern=raw["pattern"],
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                msg = f"AuditLog: malformed classification entry at index {i}: {exc}"
                raise ValueError(msg) from exc

        for i, raw in enumerate(payload.get("discovered_chains", [])):
            try:
                # Round-trip role through TensorRole to catch stale or
                # invalid role strings at load time rather than at use time.
                role_str = TensorRole(raw["role"]).value
                log._discovered.append(
                    DiscoveredChain(
                        chain=bytes(raw["chain"]),
                        dtype_code=int(raw["dtype_code"]),
                        role=role_str,
                        n_candidates_tried=int(raw["n_candidates_tried"]),
                        sample_tensor_name=raw["sample_tensor_name"],
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                msg = f"AuditLog: malformed discovered_chains entry at index {i}: {exc}"
                raise ValueError(msg) from exc
        return log
