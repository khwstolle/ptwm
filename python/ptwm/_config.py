from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptwm._rust.policy import ResolvedPolicy

    from .codecs import CodecId


class Method(Enum):
    AUTO = 0
    HUFFMAN = 1
    ZSTD = 2
    IDENTITY = 3
    RANS = 4
    MICROSCALE = 5

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]
        return None


class Format(Enum):
    BYTE = 1
    TORCH = 2
    NUMPY = 3
    FILE = 4

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]
        return None


class Framing(Enum):
    r"""Byte layout of a single-tensor :meth:`Compressor.compress` output.

    ``COMPACT`` (default) writes a compact single-tensor frame (magic
    ``\x89PTWS``) with a fixed cost of at most about 70 bytes, and stores the
    raw bytes when no codec makes them smaller. It falls back to a full
    container when the tensor cannot use the frame (streaming, or a plane that
    selects shared codec state). ``CONTAINER`` always writes a full ``.ptwm``
    container, which the random-access, store and transcode tools open.

    A full container (from ``CONTAINER`` or the ``COMPACT`` fallback) has the
    layout :attr:`CompressionConfig.container_layout` selects: ``\x89PTWC``
    for :attr:`ContainerLayout.COMPACT` (the default) and ``\x89PTWM`` for
    :attr:`ContainerLayout.LEGACY`. So ``Framing.CONTAINER`` with the default
    layout gives a ``\x89PTWC`` container.

    Readers before v1.2 read only ``CONTAINER`` output in the
    :attr:`ContainerLayout.LEGACY` layout (magic ``\x89PTWM``); they reject
    the ``\x89PTWS`` frame and the ``\x89PTWC`` compact container. Set
    ``framing=Framing.CONTAINER`` and
    ``container_layout=ContainerLayout.LEGACY`` for output that older releases
    read.
    """

    COMPACT = 1
    CONTAINER = 2

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]
        return None


class ContainerLayout(Enum):
    r"""On-disk layout of a multi-tensor ``.ptwm`` container.

    ``COMPACT`` (magic ``\x89PTWC``, the default for new writes) stores
    built-in extensions by id, integers as varints and tensor names only in
    the index; it needs v1.2 or later to read. ``LEGACY`` (magic
    ``\x89PTWM``) is the fixed-width layout that v1.1 and later read. Its
    Extension Table interns all 16 built-in codecs in built-in order, as the
    v1.1.0 writer does, so a v1.1 transcoder rebuilds exploded LMDB and
    WebDataset stores of it correctly.

    Strings coerce case-insensitively: ``ContainerLayout("compact")`` and
    ``ContainerLayout("LEGACY")`` work, and the bindings take the lower-case
    :attr:`value`.
    """

    COMPACT = "compact"
    LEGACY = "legacy"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]
        return None


def container_layout_name(layout: ContainerLayout | str) -> str:
    """Return the binding name (``"compact"`` or ``"legacy"``) of ``layout``.

    ``layout`` is a :class:`ContainerLayout` or a string that names one (any
    case). Anything else raises :class:`ValueError`.
    """
    if isinstance(layout, ContainerLayout):
        return layout.value
    try:
        return ContainerLayout(layout).value
    except ValueError:
        msg = f"unknown container_layout {layout!r} (expected 'compact' or 'legacy')"
        raise ValueError(msg) from None


class Lossy(Enum):
    NONE = 0
    INTEGER = 1
    UNSIGN = 2

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class CompressionConfig:
    method: Method = Method.AUTO
    input_format: Format = Format.BYTE
    bytearray_dtype: str = "bfloat16"
    is_monotonic: int = 0
    threads: int | None = None
    compression_threshold: float = 0.95
    check_th_after_percent: int = 10
    reorder_signbit: int = 0
    delta_compressed_type: str | None = None
    lossy_compressed_type: Lossy = Lossy.NONE
    lossy_compressed_factor: int = 27
    compression_chunk: int = 256 * 1024
    is_streaming: bool = False
    streaming_chunk: int = 1024 * 1024
    input_file: str | None = None
    compressed_file: str | None = None
    decompressed_file: str | None = None
    zstd_level: int = 3
    lz4_compression_level: int = 0
    codec_menu: list[CodecId] | None = None
    """Restrict per-role trial-encode menus to this set of codecs.

    None → use the full role-applicable menu. An empty intersection on a
    role raises ``InvalidContainer`` — an Identity fallback would silently
    falsify ablation measurements. Ignored when ``method`` selects a forced-
    codec path (ZSTD / RANS / IDENTITY).
    """
    codec: str | None = None
    device: int | None = None
    framing: Framing = Framing.COMPACT
    """Output layout; see :class:`Framing`."""
    container_layout: ContainerLayout = ContainerLayout.COMPACT
    """Layout of a full container written with ``Framing.CONTAINER`` (or as
    the ``Framing.COMPACT`` fallback): ``COMPACT`` (default, ``\x89PTWC``) or
    ``LEGACY`` (``\x89PTWM``, readable before v1.2); see
    :class:`ContainerLayout`. A string such as ``"legacy"`` is coerced on
    construction."""
    emit_payload_hash: bool = True
    """Store an xxh64 of the raw bytes (8 B per tensor), checked on decode."""
    emit_plane_crc: bool = True
    """Store a CRC32 of every coded plane (4 B per plane), checked on decode.

    A compact frame stores plane CRCs only when ``emit_payload_hash`` is off:
    the payload hash already covers every plane of its one tensor.
    """

    def __post_init__(self) -> None:
        # Accept "compact" / "legacy" (any case); unknown names raise
        # ValueError here rather than when the container is written.
        if not isinstance(self.container_layout, ContainerLayout):
            object.__setattr__(
                self, "container_layout", ContainerLayout(self.container_layout)
            )
        # The same for framing ("compact" / "container", or the int value).
        # Without it, a string never equals Framing.COMPACT and the output is
        # silently a full container.
        if not isinstance(self.framing, Framing):
            object.__setattr__(self, "framing", Framing(self.framing))

    @classmethod
    def from_resolved_policy(
        cls,
        rp: ResolvedPolicy,
        *,
        base: CompressionConfig | None = None,
    ) -> CompressionConfig:
        """Build a constrained ``CompressionConfig`` from a ``ResolvedPolicy``.

        The returned config's ``codec_menu`` is the intersection of the
        resolved policy's ``effective_global`` set with the wire-stable
        built-in ``CodecId`` variants.

        The resolved policy lists every contribution (built-in or installed
        third-party) that is currently trusted *and* allowed by the policy
        file. Restricting the trial-encode menu to that intersection is how
        an ablation run actually exercises a variant policy: tensors that
        previously round-tripped through a now-excluded codec will fall
        through to the next-best allowed coder instead.

        ``base`` is the starting config (defaults to a default-constructed
        ``CompressionConfig``); every field other than ``codec_menu`` is
        preserved.
        """
        # Imported locally — `ptwm.codecs` reaches back through
        # `ptwm.preprocessing → ptwm.utils → ptwm._config` at top level,
        # so a module-level import here is a real circular-import bug
        # (not just a ruff style preference).
        from ptwm._rust import builtin_canonical_id  # noqa: PLC0415

        from .codecs import CodecId  # noqa: PLC0415

        # Map every wire-stable built-in codec to the short name used in
        # `crates/ptwm-core/src/extension/builtins.rs`.
        _BUILTIN_CODEC_NAMES: dict[CodecId, str] = {
            CodecId.Identity: "identity",
            CodecId.Huffman: "huffman",
            CodecId.Rans: "rans",
            CodecId.Zstd: "zstd",
            CodecId.PerGroupCodebook: "per_group_codebook",
            CodecId.Order1ScaleAC: "order1_scale_ac",
        }

        allowed_hex = set(rp.effective_global_ids())

        # Empty `effective_global` means "no policy applied" (the default
        # `PolicyFile.default_empty()` case): preserve the base's
        # codec_menu rather than collapsing to an empty list, which the
        # Rust dispatcher would treat as "no codecs allowed → error".
        if not allowed_hex:
            return base if base is not None else cls()

        menu: list[CodecId] = []
        for codec_id, short_name in _BUILTIN_CODEC_NAMES.items():
            if builtin_canonical_id(short_name).hex() in allowed_hex:
                menu.append(codec_id)

        starting = base if base is not None else cls()
        return replace(starting, codec_menu=menu)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecompressionConfig:
    # Most properties are inferred from the header; this struct only carries
    # runtime specifics like thread count and the delta reference.
    threads: int | None = None
    delta_second_data: bytes | None = None
    skip_missing: bool = False
    """If ``True``, open containers that reference non-builtin extensions
    that are not installed on this host.  Tensors needing those extensions
    will fail when decoded, but the container can still be opened and
    interrogated (e.g. listing tensor names).  When ``False`` (the default),
    opening such a container raises ``ValueError``.
    """
    codec: str | None = None
    device: int | None = None
