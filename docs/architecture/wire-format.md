# The `.ptwm` wire formats

The on-disk formats are the most stable surface in PTWM. A reader must
keep decoding every file that an older PTWM release wrote, and a format
change must make older readers fail cleanly instead of misreading the
new bytes. This document describes the bytes of each format and the
reasoning behind the choices that the bytes do not explain.

PTWM writes three formats. All of them start with the bytes `\x89PTW`,
and the fifth byte tells them apart:

| Magic | Format | Writer | Readers |
|---|---|---|---|
| `\x89PTWM\r\n\x1a\n` | legacy multi-tensor container | `ContainerLayout::Legacy` / `container_layout="legacy"` | v1.2 and later; older releases with a caveat (see [Compatibility](#compatibility)) |
| `\x89PTWX\r\n\x1a\n` | legacy container with embedded WASM extensions | extension bundling | as `\x89PTWM` |
| `\x89PTWC\r\n\x1a\n` | compact multi-tensor container | `ContainerLayout::Compact`, the default | v1.2 and later |
| `\x89PTWS` | compact single-tensor frame | `Compressor.compress`, the default | v1.2 and later |

The three container magics are nine bytes long. The frame magic is five
bytes long. Compare the whole magic, not a prefix: readers before v1.2
check only for `\x89PTWM`, and that check is what makes them reject the
newer formats.

## Why a custom container at all

The natural alternative is to wrap each tensor's compressed bytes in
the existing `.safetensors` format. PTWM still supports this through
the integration adapter (`compress_safetensors_file` with
`mode="b"`), which produces a `.safetensors` file whose data section
is a `.ptwm` blob. The native `.ptwm` container exists alongside the
integration because three properties are hard to achieve in
`.safetensors` directly.

First, per-plane codec dispatch. A trained-weight tensor compresses
best when its exponent byte and its mantissa bytes flow through
different codecs. The plane records in PTWM tell the decoder which
codec produced each plane, and the preprocessing chain fixes the plane
boundaries, so the decoder does not guess them. `.safetensors` has no
notion of per-tensor metadata beyond shape and dtype.

Second, shared state. Some codecs (`Order1ScaleAC`, for example) use a
probability table that several tensors can share. The prelude stores
such state once per file instead of once per tensor.

Third, the Extension Table. Third-party codecs need a stable way to
say "the bytes in this plane were produced by the codec with this
canonical id". The Extension Table makes that explicit and lets the
verifier check the contribution before any third-party code runs.

## Conventions

These rules hold for every format unless a section says otherwise.

- Fixed-width integers are little-endian.
- A *varint* is unsigned LEB128 (`crate::varint`). It is at most 10
  bytes long and must be minimal: a reader rejects a varint with
  redundant trailing `0x80` groups, so every value has one encoding.
  Every varint field also has a range, given in its section, which the
  reader checks.
- A reader rejects bytes it does not understand (unknown flag bits,
  unknown tags, unknown enum values) in the compact formats. The legacy
  container is older and less strict; [What the legacy reader
  checks](#what-the-legacy-reader-checks) lists the gaps.
- Every container ends with the 8-byte EOF sentinel
  `89 54 57 4D 45 4F 46 1A` (`\x89TWMEOF\x1a`, `index::EOF_SENTINEL`).
  The reader checks it first, so a truncated file fails in constant
  time. The first byte has the high bit set, like the magic, so a
  transfer that strips the eighth bit also breaks the sentinel.

## Versioning

No format stores a numeric version. A version field tends to encourage
a v1-then-v2 mindset: every reader becomes a switch statement over the
version, and every breaking change adds another arm. PTWM uses two
other mechanisms instead.

1. **A new magic for an incompatible layout.** `\x89PTWX`,
   `\x89PTWC` and `\x89PTWS` are examples. A reader that predates a
   magic rejects the file at the magic check, so an old reader never
   applies its rules to new bytes. The rule runs in that direction: a
   new reader keeps reading the old magics.
2. **Reserved flag bits for compatible additions.** In the compact
   formats every flag bit that is not defined must be zero, and the
   reader rejects the file otherwise. A future feature takes a free
   bit, and readers that predate it reject files that use it with a
   clear error. The PTWC header reserves all 16 bits of `hflags` for
   this.

Built-in operations and codecs are identified on the wire by the
`OpId` and `CodecId` wire ids in PTWC and PTWS, and by the built-in
name in the legacy Extension Table, whose ids are derived from the name
and the release version (see [Built-in entries](#built-in-entries)).
The legacy plane record and prelude also store `CodecId` wire ids. A
reader decodes a built-in with the algorithm of its own release. **The meaning of a built-in wire id must
never change.** A change to what a built-in op or codec does to the
bytes, even a bug fix, needs a new `OpId` or `CodecId` and a new
built-in name, with the old ones kept for decoding existing files.
The same applies to a change to the layout of a codec's state or
params: a new state version or a new id, never a reinterpretation.

## Legacy container (`\x89PTWM`, `\x89PTWX`)

`ContainerLayout::Legacy`. Every PTWM release reads this layout;
readers before v1.2 also need the file's built-in ids to be those of
their own version (see [Compatibility](#compatibility)).

```
[ header                 ]  91 bytes
[ Extension Table        ]  header: extension_table_offset/length
[ prelude                ]  header: shared_prelude_offset; the
[ chain registry         ]    shared-prelude section covers both
[ tensor record 1        ]
[ tensor record 2        ]
...
[ tensor index           ]  header: tensor_index_offset/length
[ EOF sentinel           ]  8 bytes
```

The writer streams the records and then seeks back to fill in the
header, so it needs a seekable output.

### Header

```
magic                   [u8; 9]   \x89PTWM\r\n\x1a\n or \x89PTWX\r\n\x1a\n
flags                   u16       bit 0: PTWX (embedded WASM)
extension_table_offset  u64       91 in files current writers produce
extension_table_length  u64
extension_table_hash    [u8; 32]  blake3 of the Extension Table bytes
tensor_index_offset     u64
tensor_index_length     u64
shared_prelude_offset   u64
shared_prelude_length   u64       prelude and chain registry together
```

The header is fixed-size, so a reader can map the file, read 91 bytes
and go straight to the section it needs. There is no field for the
chain registry: it starts where the prelude ends, inside the
shared-prelude section, and the reader parses both from that section
alone. Tensor records start at the end of the shared-prelude section.

The Extension Table hash sits in the header rather than next to the
table, so a reader checks the table against bytes it has already
read. Bit 0 of `flags` must be set exactly when the magic is
`\x89PTWX`.

### Extension Table

```
count                  u32        at most 65,536
entry × count:
    canonical_id         [u8; 32]
    human_label          u16 length + UTF-8
    kind                 u16      Transform, PlaneCodec, ...
    abi_version          u16
    lifecycle            u8
    flavor_hints         u8       bit 0 WASM, bit 1 native, bit 2 host
    capabilities         u32 length (at most 1 MiB) + canonical CBOR
    attestation_kind     u8       1 = PGP signature
    attestation          u16 length + bytes
    install_hint         u16 length + UTF-8 (length 0: none)
    embedded_wasm_offset u64      0: none
    embedded_wasm_length u64      0: none
```

The table lists every contribution the file needs at decode time:
first the ops of every registry chain, then the plane codecs, in
built-in registry order. Chain nodes and plane records refer to entries
by position. A built-in entry costs about 130 bytes, which is why the
compact container stores built-ins by wire id instead.

The legacy writers intern all 16 built-in plane codecs, about 2 KB per
file, whether or not a plane uses them. This is the v1.1.0 layout, and
the v1.1 transcoder depends on it (see [Compatibility](#compatibility)).
`ContainerWriter::set_chain_registry_with_codecs` writes a table with
only the referenced codecs. Readers resolve `codec_table_idx` by
position and check the table hash against the bytes present, so they
read both forms. The transcoder rebuilds the table from the chain
registry and the codec list and keeps whichever form reproduces the
recorded hash.

The flavor hints are a bitfield rather than an enum because a
contribution can ship in several flavours and the reader picks one by
host capability. The capabilities are CBOR so the reader needs no TOML
parser.

#### Built-in entries

A built-in entry has the canonical id
`blake3(BUILTIN_PUBKEY ‖ 0x00 ‖ name ‖ 0x00 ‖ version)` and the label
`io.ptwm.builtin.<name>@<version>`. Releases up to v1.1 wrote their own
crate version there, so the id changed with every release although the
entry means the same thing. From v1.2 the legacy writers write
`LEGACY_BUILTIN_VERSION` (`1.1.0`) whatever their crate version, so
v1.1 readers recognise their built-ins. The transcoder can write other
versions with `ContainerWriter::set_builtin_version` to reproduce files
of other releases.

From v1.2 the reader handles this with `normalize_builtin_ids`: after
the hash check, an entry whose label names a built-in and whose id
re-derives from that label is mapped to the current built-in. An entry
whose id does not re-derive keeps its id and is treated as third-party,
so a forged label gains no trust. Readers before v1.2 recognise only
the ids of their own version.

### Prelude

```
count                   u32
entry × count:
    shared_state_id       u16
    codec_id              u16      CodecId wire id
    state_format_version  u8
    applies_to_mask       u8       bit (r − 1) for PlaneRole r in 1..=8
    reserved              u16
    state_xxhash64        u64      xxh64 (seed 0) of state_bytes
    name                  u16 length + UTF-8, zero-padded to a multiple of 4
    state_bytes           u32 length + bytes
```

A plane with state source Shared names an entry by `shared_state_id`
(its `state_info`). The reader checks the version, the role mask and
the hash before it hands the state to the codec.

### Chain registry

```
num_chains              u16      at most 0x8000
chain × num_chains:
    chain_id              u16      at most 0x7FFF; writers use the position
    chain                 legacy chain wire
```

The legacy chain wire (`chain::wire`) is:

```
num_nodes, num_edges, num_terminals   u8 each
nodes_len, edges_len, terminals_len   u16 each, section byte lengths
node:      op u16 (Extension Table index), param_len u8, params
edge:      src u8, src_out u8, dst u8, dst_in u8, has_role u8,
           [role], vendor_len u16, vendor bytes
terminal:  node u8, out u8, role
```

A tensor record addresses a registry chain with 15 bits, so a writer
fails on a registry with more than 0x8000 chains or a chain id above
0x7FFF.

### Tensor record

```
chain_ref        u16     bit 15: inline chain; else bits 0-14 are the chain_id
dtype_code       u16
input_format     u8
num_planes       u8
flags            u8      0x01 dependencies, 0x02 metadata, 0x04 payload hash
reserved         u8
orig_size        u64     raw tensor bytes
name             u16 length + UTF-8, zero-padded to a multiple of 4
[payload_hash]   u64     xxh64 (seed 0) of the raw tensor        (0x04)
[inline chain]   Extension Table (u32 count + entries as above), then a
                 legacy chain whose ops index that table      (bit 15)
[dependencies]   u16 count; per dependency: ref_kind u8, ref_flags u8
                 (bit 0: hash), u16 length + ref bytes, [blake3 32 B]
                                                                  (0x01)
[metadata]       u32 length + CBOR shape map (shape, dtype name)  (0x02)
plane × num_planes
```

The fixed part is 18 bytes. With the name, the shape metadata and four
planes, an fp32 tensor record carries 166 bytes besides its payload, or
190 bytes with the payload hash and plane CRCs (name of 28 bytes).

### Plane record

```
role              u8      PlaneRole (diagnostic label)
plane_flags       u8      0x01 chunk table, 0x02 CRC32, 0x04 codec_table_idx
                          present (always set)
codec_id          u16     CodecId wire id
codec_table_idx   u16     Extension Table index of the plane codec
state_source      u8      0 none, 1 inline, 2 shared, 3 external
state_version     u8
state_info        u16     shared: shared_state_id
inline_state_len  u32
payload_len       u64
[crc32]           u32     CRC32 of the decoded plane                (0x02)
[chunk table]     u32 count; per chunk: offset_in_payload u32,
                  decoded_size u32                                  (0x01)
[external state]  ref_kind u8, ref_flags u8 (bit 0: hash), u16 length +
                  ref bytes, state_format_version u8, [blake3 32 B]
                                                       (state_source 3)
layout_kind       u8      0 flat, 1 rows
[row_len]         u32     non-zero                            (rows)
inline state      inline_state_len bytes
payload           payload_len bytes
```

A flat plane without CRC costs 23 bytes plus its payload. The decoder
finds the codec through `codec_table_idx` alone. `codec_id` duplicates
it; every writer fills both, and the reader requires `codec_id` to be a
known `CodecId` but does not dispatch on it. The reader reads
`codec_table_idx` whether or not flag 0x04 is set.

The plane record stores no shape and no decoded length. The decoder
derives each plane's length from the chain (see [Reader
limits](#reader-limits)).

### Tensor index

```
count            u64
entry × count:
    name_hash      u64    xxh64 (seed 0) of the UTF-8 name
    tensor_offset  u64    absolute offset of the record
    tensor_len     u64    record length
```

Entries are sorted by `name_hash`. A lookup binary-searches the hash,
then compares the name in each record of the run of equal hashes, so
two names with the same hash resolve correctly. Writers reject
duplicate tensor names. Tools that list or decode every tensor return
them in record (write) order, not index order.

### What the legacy reader checks

The reader checks the EOF sentinel, the magic, the PTWX flag against
the magic, the bounds of every section, the Extension Table hash, the
prelude and registry within their section, every index entry's range
before it reads a record, the decoded plane lengths against the plane
bound (see [Reader limits](#reader-limits)), the plane CRCs and the
payload hash.

It does not check:

- that the index is sorted (an unsorted index makes lookups miss);
- the reserved bytes and unknown flag bits of records and the prelude;
- that records are contiguous or that every byte belongs to a section.

None of these can make the reader read out of bounds; they are why the
legacy layout has more than one encoding for the same content, and why
the compact layout rejects them.

Payloads are not aligned. Only the names in the tensor record and the
prelude are padded to a multiple of 4 bytes, and the rest of the file
has no alignment. Codecs read their input as bytes and do not assume
alignment.

## Compact container (`\x89PTWC`)

`ContainerLayout::Compact`, the default for multi-tensor output since
v1.2. It holds the same information as the legacy container, with
varints, built-ins stored by wire id, names moved into the index, and
fields the reader can derive left out. The reader parses it into the
same in-memory records as a legacy file, so decoding, trust checks and
the transcoder share one code path.

```
FILE = HEADER | EXTENSION TABLE | PRELUDE | REGISTRY | BODY | INDEX | FOOTER
```

All sections are contiguous. The writer streams them and never seeks:
the header comes first because the table, prelude and registry are
known before the first record, and the index length goes in the
footer.

Every section parser gets exactly its slice and must consume all of
it. Together with minimal varints and zero reserved bits, this gives
each file one encoding for its content, so the transcoder's
`implode(explode(x))` reproduces `x` byte for byte.

### Header and footer

```
magic         [u8; 9]   \x89PTWC\r\n\x1a\n
hflags        u16       must be 0 (all bits reserved)
et_len        varint    Extension Table bytes
prelude_len   varint    prelude bytes
registry_len  varint    registry bytes
meta_xxh64    u64       xxh64 (seed 0) of the header bytes before this
                        field, then the Extension Table, prelude and
                        registry bytes
...
index_len     u64       (footer)
sentinel      [u8; 8]   \x89TWMEOF\x1a (footer)
```

The header is 22 bytes for files whose sections are each shorter than
128 bytes, and grows by one byte per seven bits of section length. The
body runs from the end of the registry to the start of the index,
which is `file_len − 16 − index_len`. The reader checks all offsets
with checked arithmetic and rejects a file whose body would have a
negative length.

`meta_xxh64` protects the sections a reader parses before it touches
a record. xxh64 is enough here: it detects corruption, and the trust
decision does not rest on it, because a built-in is stored by wire id
and a third-party entry carries its own canonical id and attestation.

### Extension Table

```
count       varint   at most 65,536 and at most et_len / 2
entry × count:
    tag     u8
    0x01    OpId wire id (varint)       built-in transform
    0x02    CodecId wire id (varint)    built-in plane codec
    0x03    name length varint (1..=255), then the built-in name
                                        built-in without a wire id
    0x7F    length varint, then one legacy entry of exactly that length
                                        third-party entry
```

The reader expands tags 0x01 to 0x03 into the built-in entry of its own
version, so a PTWC file does not depend on the release that wrote it.
Six built-in entries take 13 bytes, against 779 in the legacy table.

The reader rejects unknown tags, ids and names, duplicate entries, a
built-in stored with tag 0x7F (of this release or, by its label, of
another one), a tag 0x03 entry for a built-in that has a wire id, and
any entry with embedded WASM. Embedded WASM exists only in `\x89PTWX`
files; the PTWC writer refuses such entries. The writer orders entries
like the legacy writer: chain ops first, then the referenced plane
codecs.

### Prelude

The legacy prelude bytes, parsed within `prelude_len`. Shared state is
rare and the format is already compact enough.

### Registry

```
count            varint   at most 0x8000 (chain ids 0..=0x7FFF)
chain × count:   chain_len varint, then a CW2 chain
```

The chain id is the position. A node's `op_ref` is an Extension Table
index that must name a built-in transform. A registry chain may have a
bound Source (see [CW2](#compact-chain-wire-cw2)), which takes its
params from the record that uses the chain. Tensors that differ only in
shape then share one registry chain: 64 fp32 tensors with 64 distinct
shapes need two registry chains (48 bytes) in PTWC, against 64 chains
(3,872 bytes) in the legacy layout.

### Tensor record

Records follow each other in write order. A record does not store its
name: the index does.

```
tflags          u8      bit 0 PAYLOAD_HASH, bit 1 INLINE_CHAIN, bit 2 DEPS,
                        bit 3 SHAPE, bit 4 DTYPE_NAME (needs SHAPE),
                        bit 5 RAW_META (excludes SHAPE); bits 6-7 zero
[chain_id]      varint  < registry count               (not INLINE_CHAIN)
dtype_code      varint  at most 65,535
input_format    u8
orig_size       varint
[shape]         varint ndim (at most 64), varint per dimension     (SHAPE)
[dtype_name]    varint length + UTF-8                         (DTYPE_NAME)
[raw_meta]      varint length + bytes                           (RAW_META)
[payload_hash]  u64     xxh64 (seed 0) of the raw tensor    (PAYLOAD_HASH)
[inline chain]  varint length + compact Extension Table of the chain's
                ops in first-use order, then varint length + CW2 chain
                                                            (INLINE_CHAIN)
[dependencies]  varint count (at least 1); per dependency: ref_kind u8,
                ref_flags u8 (bit 0: hash), varint length + ref bytes,
                [blake3 32 B]                                       (DEPS)
n_planes        varint  at most 255
plane × n_planes
```

SHAPE stands for the CBOR shape map of the legacy record, built from
the shape and the canonical dtype name of `dtype_code`, or from the
stored dtype name under DTYPE_NAME. The reader rejects a DTYPE_NAME
equal to the canonical name. The writer uses SHAPE only when it
reproduces the record's metadata bytes exactly; other metadata goes in
RAW_META.

The record must end exactly at the length the index gives. The
in-memory `chain_ref`, `flags` and `name` are rebuilt from the record
and the index, as the legacy parser would produce them.

### Plane record

```
pflags          u8      bit 0 CRC32, bit 1 CHUNKS, bits 2-3 state source
                        (0 none, 1 inline, 2 shared, 3 external),
                        bit 4 LAYOUT_ROWS, bit 5 STATE_EXTRA; bits 6-7 zero
role            u8      PlaneRole
codec_ref       varint  Extension Table index of a plane codec
[state_version] u8                                         (STATE_EXTRA)
[state_info]    varint  at most 65,535                     (STATE_EXTRA)
[inline_len]    varint                                   (state inline)
payload_len     varint
[row_len]       varint  1..=u32::MAX                      (LAYOUT_ROWS)
[crc32]         u32     CRC32 of the decoded plane              (CRC32)
[chunks]        varint n (at most payload_len); n × (varint offset,
                varint decoded_size), each at most u32::MAX   (CHUNKS)
[external]      ref_kind u8, ref_flags u8 (bit 0: hash), varint length +
                ref bytes, state_format_version u8, [blake3 32 B]
                                                       (state external)
[inline state]  inline_len bytes
payload         payload_len bytes
```

STATE_EXTRA is set exactly when `state_version` or `state_info` is
non-zero. The plane does not store `codec_id`: the reader derives it
from the Extension Table entry, and the writer rejects a record whose
`codec_id` disagrees. Role and layout are stored so that parsing a
record never needs the chain.

A typical plane takes 4 bytes plus its payload, or 8 with a CRC.

### Tensor index

```
count        varint   at most index_len / 3
entry × count, in body order:
    name       varint length (1..=65,535) + UTF-8
    rec_len    varint, at least 3
```

Record offsets are the running sum of `rec_len` from the start of the
body, and the sum must equal the body length exactly, so records cannot
overlap or leave gaps. Names must be unique. The reader sorts the
entries by name hash in memory; a lookup compares names, so a hash
collision cannot return the wrong tensor. `tensor_names` reads the
index alone, without touching the records.

### Sizes

Measured with `benchmarks/container_overhead.py`, fp32 tensors of shape
`(64,)` named `model.layers.NNN.attn.q.bias` (28 bytes), written with
`compress_ptwm_blob`. "Non-payload" excludes plane payloads and inline
and shared codec state.

| Case | PTWC B/tensor | Legacy B/tensor |
|---|---|---|
| 64 tensors, same shape, hash and CRC on | 83 | 262 |
| 64 tensors, distinct shapes, hash and CRC on | 83 | 317 |
| 64 tensors, same shape, hash and CRC off | 59 | 238 |
| 1 tensor, hash and CRC on | 165 | 3,029 |

Per tensor, the PTWC index entry takes 31 bytes (mostly the name) and
the record about 50 bytes with the payload hash and four plane CRCs,
26 without. CI holds these to the bounds in
`tests/ptwm/benchmarks/test_container_overhead.py` and
`crates/ptwm-core/tests/ptwc.rs`.

## Compact chain wire (CW2)

`chain::wire_compact` stores a chain in about 15 to 30 bytes, against
50 to 70 for the legacy chain wire. PTWC registry and inline chains and
the PTWS frame use it.

```
n_nodes, n_edges, n_terms    varint each (n_nodes 1..=255, others <= 255)
node:      op_ref varint, param_len varint (<= 255), params
n_spine    varint (<= n_edges; only when n_edges > 0)
edge:      src, src_out, dst, dst_in varint each (<= 255); etag u8
           (bit 0 role, bit 1 vendor; other bits zero); [compact role];
           [varint length (1..=65,535) + vendor bytes]
terminal:  ref u8; compact role
```

- `op_ref` is an Extension Table index in PTWC and the `OpId` wire id
  in PTWS.
- The first `n_spine` edges are `(i, 0) → (i + 1, 0)` with no role
  override and no vendor bytes, and are not stored. The writer always
  stores the longest such run, and the reader rejects a stored first
  edge that would extend it.
- A Source node with `param_len = 0` is a *bound Source*: its params
  are those of the record's or frame's shape and dtype (`u8` ndim,
  `u32` dims, `u16` chain dtype). The writer binds a Source exactly
  when its params equal these, and the reader rejects explicit params
  that equal them. Real Source params are never empty, so the marker
  cannot collide with an explicit Source. A record without a shape
  cannot use a chain with a bound Source.
- A terminal `ref` is one byte, `node << 4 | out`, when `node <= 14`
  and `out <= 15`. Otherwise it is `0xFF` followed by `node` u8 and
  `out` u8. The reader rejects bytes `0xF0` to `0xFE` (reserved) and an
  escape for a pair that fits the one-byte form.
- A compact role is one byte for every role production chains use, and
  `0xFF` followed by the legacy role encoding (`Role::write`) for the
  rest. The reader rejects every other byte and an escaped role that
  has a one-byte form:

  | byte | role |
  |---|---|
  | `0x00` to `0x04` | `Scale { format }`, format tag 0 E8M0, 1 E4M3, 2 E5M2, 3 F32, 4 F16 |
  | `0x08` to `0x0C` | `GlobalScale { format }`, `0x08` + format tag |
  | `0x10` to `0x12` | `Nibble { kind }`, kind 0 Exponent, 1 SignMantissa, 2 Value |
  | `0x18` | `Value { Fp4E2m1 }` |
  | `0x19` | `ExponentByte` |
  | `0x1A` | `Raw` |
  | `0x1B` | `Index` |
  | `0x1C` | `Residual { Xor }` |
  | `0x40 + 8 × (of − 1) + index` | `MantissaByte { index, of }`, `index < of <= 8` |
  | `0x80 + 8 × (of − 1) + index` | `IntegerByte { index, of }`, `index < of <= 8` |
  | `0xC0 + bits − 1` | `Value { IntN { bits } }`, `1 <= bits <= 32` |
  | `0xFF` | escape, then the legacy role bytes |

## Compact single-tensor frame (`\x89PTWS`)

`Compressor.compress` writes one tensor. For one tensor, a container's
sections cost about 85 to 120 bytes for bf16 and 125 to 140 bytes for
fp32 (PTWC), or 2.5 to 2.9 KB (legacy, which interns every built-in
codec). That is more than the payload of a small tensor. By default
the compressor writes a compact frame instead:

```
magic          [u8; 5]   \x89PTWS
flags          u8        bit 0 payload hash, bit 1 plane CRC, bit 2 shape,
                         bit 3 dtype name, bit 4 delta reference,
                         bit 5 stored, bit 6 size; bit 7 zero, and
                         plane CRC with payload hash is rejected
dtype_code     varint    at most 65,535
input_format   u8
orig_size      varint                             (size)
shape          varint ndim (<= 64) + varint dims  (shape)
dtype_name     varint length + UTF-8              (dtype name)
delta_ref      [u8; 32]  blake3 of the reference  (delta reference)
payload_hash   u64       xxh64 of the raw bytes   (payload hash)
body
```

`orig_size` is stored only when the shape does not give it as
`product(shape) × element_size(dtype_code)`. The dtype name is stored
only when it differs from the canonical name of `dtype_code`. A reader
rejects any field written in a form the writer would not produce, so
each field has one encoding for the value it holds. The frame as a
whole is not unique for a tensor: the writer chooses the codecs, the
chain, a stored or a coded body, and plane state 0 or 2 (below), and
the reader accepts any valid choice. A reader rejects bytes after the
end of the body.

The body is either the raw bytes (the stored flag, written when no
codec makes the tensor smaller) or a coded tensor: a CW2 chain whose
`op_ref` is the `OpId` wire id and whose bound Source takes the
frame's shape and dtype, then one record per terminal plane:

```
head           varint    CodecId wire id << 2 | state
                         (0 none, 1 inline, 2 inline with version 0 and
                         no bytes; 3 rejected)
[state]        version u8, varint length, bytes        (state 1)
payload_len    varint
[crc32]        u32       CRC32 of the decoded plane    (plane CRC)
payload        payload_len bytes
```

States 0 and 2 decode alike. The payload hash already covers every
plane of the one tensor, so the writer stores plane CRCs only when it
does not store the hash. The frame does not store plane layouts, plane
lengths or the plane count, because the decoder derives them from the
chain.

The fixed cost is about 20 to 70 bytes, depending on the chain and on
which integrity fields are on. With the default settings (payload hash
on) it is at most 43 bytes for a bf16 tensor and 55 bytes for an fp32
tensor with the split chain, and 49 bytes for a bf16 tensor and 61
bytes for an fp32 tensor with the IntDelta or PredictorXor chain that
smooth data selects. Plane CRCs instead of the hash add up to 8 bytes
more for four planes.

A tensor whose planes need shared state (a prelude) or chunked encoding
cannot use the frame; the compressor writes a full container for it,
in the configured `container_layout`. The frame holds no Extension
Table, so it can reference built-in ops and codecs only.

`CompressionConfig(framing=Framing.CONTAINER)` writes a full container
instead of the frame, for the random-access, store and transcode tools.

## Compatibility

| File | Readers before v1.2 | v1.2 and later |
|---|---|---|
| `\x89PTWS` frame (default single-tensor output) | reject: Python `Decompressor` "not a .ptwm container"; Rust and `_rust` container functions "missing EOF sentinel (file may be truncated)" | read |
| `\x89PTWC` container (default multi-tensor output) | reject: Python `Decompressor` "not a .ptwm container"; Rust and `_rust` container functions "bad magic" | read |
| `\x89PTWM` container | read if its built-in ids are of the reader's version | read, built-in ids of any release |

Old readers fail cleanly on the new formats; none misreads them. The
Python `Decompressor` before v1.2 checks the 5-byte `\x89PTWM` prefix
and reports that the input is not a container. The Rust reader before
v1.2 (`ContainerReader::open`, and through it `_rust.decode_model`,
`_rust.decode_tensor`, `_rust.list_tensor_names` and the transcoder)
checks the EOF sentinel first. A PTWC container ends with the
sentinel, so the reader goes on to `Header::read` and fails with "bad
magic". A PTWS frame has no sentinel, so the reader fails with
"missing EOF sentinel (file may be truncated)", although the file is
intact. From v1.2 the container functions name a PTWS frame as such
and point to `decode_compact`. The transcoder before v1.2 rejects the
`"ptwm-manifest-c"` manifest of a PTWC container.

To write files that a v1.1 reader opens:

- Python: `CompressionConfig(framing=Framing.CONTAINER,
  container_layout=ContainerLayout.LEGACY)`, or
  `container_layout="legacy"` on `compress_ptwm_blob`,
  `write_ptwm_shard`, `compress_safetensors_file`, the LMDB and
  WebDataset writers and `_rust.compress_model`.
- CLI: `ptwm compress --container-layout legacy`. It applies to every
  output, and for a `.safetensors` file without `--mode` it writes each
  tensor as a full `\x89PTWM` container instead of a `\x89PTWS` frame.
- Rust: `CompressorOptions { container_layout: ContainerLayout::Legacy,
  .. }` or `ContainerWriter::new`.

A v1.1 reader recognises only the built-in ids of version 1.1.0. The
legacy writers stamp those ids whatever their own crate version (see
[Built-in entries](#built-in-entries)), so their output opens in v1.1.

The legacy writers also keep the v1.1.0 Extension Table layout: the
chain ops, then all 16 built-in plane codecs in built-in order. The
LMDB (`layout="exploded"`) and WebDataset stores hold exploded
`"ptwm-manifest"` version 1 manifests, and a v1.1 reader rebuilds each
tensor with the v1.1 transcoder. That transcoder always interns all 16
codecs and copies each plane's `codec_table_idx` from the manifest
unchanged, so the indices must refer to that table. A legacy container
whose table holds only the referenced codecs (written with
`ContainerWriter::set_chain_registry_with_codecs`) still opens in v1.1,
but its exploded stores decode there with the wrong codecs.

## Reader limits

The readers treat every file as untrusted input. In both layouts:

- a vector sized from a count or length read from the file is bounded
  by the bytes that remain or by a small constant: an 8-bit or 16-bit
  field, or 64 Extension Table entries, after which the table grows
  only as entries parse. Every offset and length is added with checked
  arithmetic;
- the chain's Source must declare exactly `orig_size` bytes in PTWC
  and PTWS. A legacy Source may declare up to 8 × `orig_size`, so that
  files written by the v1.1 `Format.BYTE` bug keep decoding;
- the plane lengths are derived symbolically from the chain before any
  plane is decoded, and no plane may exceed `16 × orig_size + 4096`
  bytes (`max_plane_len`), so a corrupt file cannot make a codec
  allocate far more than the tensor;
- `orig_size` and the shape come from the file too, so the declared
  plane length does not size an allocation by itself. The Zstd decoders
  reserve at most what the payload can decode to (32 KiB per compressed
  byte) and grow the buffer as data arrives; reservations use
  `try_reserve`, so a failed allocation is an error, not an abort;
- a decoded plane must have exactly its derived length in PTWC and
  PTWS. In the legacy layout it may differ (older files carry Sources
  that overstate or understate the raw size) but stays under the plane
  bound. In every layout the inverse chain must give exactly
  `orig_size` bytes;
- a file whose Extension Table holds only built-ins opens without
  extension discovery or keyring access.

## Why blake3 for the legacy table hash

Blake3 is faster on the workstation CPUs PTWM targets than SHA-256, by
about a factor of three on the inputs that matter. The legacy
Extension Table hash is computed on every legacy write and checked on
every legacy open, and a slower hash would dominate the open path for
small files. PTWC uses xxh64 for its metadata check (see
[Header and footer](#header-and-footer)) because its table is a few
bytes and trust does not rest on the checksum. Blake3 stays the hash
for canonical ids, dependency references and extension bundles.

Blake3 also avoids a hash-agility surface that PTWM has no use for.
Adding SHA-256 would mean every hash field needs a discriminator. The
choice may need revisiting if policy documents come to require another
hash, but the migration cost is bounded by the number of on-disk hash
fields, which is small.

## Sentinel public key

Built-in contributions derive their canonical ids from a sentinel
public key, `BUILTIN_PUBKEY = [0xBB; 32]`. The reader recognises a
built-in by looking its id up in the table of derived built-in ids
(`builtin_registry`), after mapping ids of other releases (see
[Built-in entries](#built-in-entries)), and trusts it without a
keyring check. The sentinel bytes are recognisable, which helps when
reading hex dumps of an Extension Table.

A real Ed25519 public key with all bytes equal to 0xBB would require a
chosen-private-key attack on Ed25519. The chance of a real key
matching the sentinel by accident is `2^-256`, the collision bound the
codebase relies on elsewhere.
