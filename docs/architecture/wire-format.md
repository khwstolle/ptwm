# The `.ptwm` wire format

The on-disk container is the most stable surface in PTWM. Every
change to it must be backwards compatible with every file that an
older PTWM version may have produced. This document explains the
overall shape and the reasoning behind the choices that are not
obvious from the bytes themselves.

## Why a custom container at all

The natural alternative is to wrap each tensor's compressed bytes in
the existing `.safetensors` format. PTWM still supports this through
the integration adapter (`compress_safetensors_file` with
`mode="b"`), which produces a `.safetensors` file whose data section
is a `.ptwm` blob. The native `.ptwm` container exists alongside the
integration because three properties are hard to achieve in
`.safetensors` directly:

First, per-plane codec dispatch. A trained-weight tensor compresses
best when its exponent byte and its mantissa bytes flow through
different codecs. The plane records in PTWM let the decoder know
which codec was used for which slice of bytes, with the slice
boundaries recorded by the preprocessing chain rather than guessed
at decode time. `.safetensors` has no notion of per-tensor metadata
beyond shape and dtype.

Second, shared state. The microscaling codecs (`Order1ScaleAC` and
`PerGroupCodebook`) need a small probability table or codebook that
multiple tensors share. The `.ptwm` prelude stores these once per
file rather than once per tensor.

Third, the extension table. Third-party codecs need a stable way to
say "the bytes in this plane were produced by the codec with this
canonical id". The extension table makes that explicit and lets the
verifier check the contribution chain before any third-party code
runs.

## Top-level layout

The container is a flat byte stream with the following sections in
order. All offsets and lengths are stored in little-endian.

```
[ 91-byte header                       ]
[ extension table                       ]   variable length
[ shared-state prelude                  ]   variable length
[ tensor 1 record                       ]
[ tensor 2 record                       ]
...
[ tensor index                          ]   sorted by name hash
[ 8-byte EOF sentinel: 0x89 P T W M EOF]
```

Three of the four "variable length" sections (extension table,
prelude, tensor index) carry their offset and length in the header.
The header is fixed-size, so a reader can mmap the container, parse
91 bytes, and jump to any section without scanning the rest.

The EOF sentinel exists so a reader can detect a truncated file in
constant time. The sentinel is `\x89PTWMEOF\x1a` plus padding; the
first four bytes are PNG-style so that a file misnamed as an image
type still trips an early decode error.

## Header (91 bytes)

The choice of 91 bytes is the result of two constraints meeting. A
header that fits in one or two cache lines on a modern x86 machine
keeps header parsing branch-predictable; the natural targets are 64
or 128 bytes. The integrity field needs a cryptographic-strength
hash so that a malformed extension table is detected before any of
its entries are read; 32 bytes is the natural width for that. The
remaining offset and length fields claim another 56 bytes between
them, which puts the total above the 64-byte cache line. The 91
bytes that result fit comfortably in two cache lines and leave room
for the offset fields without a numeric version field competing for
space.

The header carries:

```
magic              [u8;9]    PTWM or PTWX
flags              u16
extension_table_offset u64
extension_table_length u64
extension_table_hash   [u8;32]   blake3
tensor_index_offset    u64
tensor_index_length    u64
shared_prelude_offset  u64
shared_prelude_length  u64
```

The extension table hash is committed in the header rather than next
to the table itself. A reader that wants to check integrity does it
without trusting any byte that lives outside the header.

## No numeric version field

PTWM does not store a version `u16` in the header. A version field
tends to encourage a v1-then-v2 mindset: every reader becomes a
switch statement over the version, and every breaking change adds
another arm. The alternative used here is per-contribution version
negotiation through the extension table, with a fresh magic for any
wire-format change that crosses the contribution boundary. The
PTWX magic (for the embedded-WASM variant) is one example of the
pattern; a future format-incompatible variant would pick a new
magic. Files with the old magic are rejected by new readers, so
there is no path by which a reader silently applies new rules to
old bytes.

This is the kind of architecture choice that is hard to walk back.
Adding a version field later is possible but the files in the wild
at the time of the addition cannot benefit from it.

## Extension table

The extension table is the per-file manifest of every contribution
the container needs at decode time. Each entry records:

```
canonical_id       [u8;32]   blake3-derived
human_label        utf8      length-prefixed
kind               u16       Transform, PlaneCodec, ...
abi_version        u16
lifecycle          u8        none, thread, process
flavor_hints       u8        bitfield: wasm, native, host
capabilities       cbor
attestation        Attestation
install_hint       Option<utf8>
embedded_wasm_offset Option<u64>
embedded_wasm_length Option<u64>
```

The canonical id is the cryptographic identifier the verifier checks
against the keyring. The human label is informational. The kind and
abi_version together tell the host which ABI surface the
contribution expects.

Two design choices here are worth flagging. The flavor hints are a
bitfield rather than an enum because a contribution can ship in
multiple flavours and the reader chooses based on host capability.
The capabilities are CBOR rather than TOML because the format is
parsed in Rust without an extra dependency on a TOML parser path
that the reader would otherwise not need.

## Plane records

Each tensor's record holds a fixed header plus a sequence of plane
records, one per terminal of the preprocessing chain. The plane
record carries:

```
codec_id              u16       legacy CodecId enum
codec_table_idx       u16       new: index into extension table
role                  u8        role bytes from PPG terminal
state_source          u8        none, inline, shared, external
state_version         u8        per-codec version of the state
state_info            u16       interpretation depends on source
inline_state_bytes    bytes     present if source == inline
shape                 list<u64> per-plane shape
payload_hash          u64       optional
plane_flags           u8        bitfield
payload_len           u64
crc32                 u32       optional
chunk_table           Option<...>
external_state        Option<...>
payload_bytes         bytes
```

The `codec_id` field predates the `codec_table_idx`. New writers fill
both. Readers that see a valid `codec_table_idx` route through the
extension table; readers that see only the legacy `codec_id` fall
back to the in-tree codec table. This dual path will be removed once
no field-deployed writer emits the legacy form alone.

The state source is the most subtle field. PTWM's compression
strategy depends on whether a codec needs per-plane state, a state
shared across planes within a tensor, a state shared across tensors
(the prelude case), or no state at all. The four-valued enum lets
the writer record exactly one of those four, which the reader uses
to find the bytes to feed back into the codec.

## Wire-format invariants

The reader assumes the following without checking, since checks are
expensive on large files and a violated invariant would surface as a
codec error at decode time anyway:

The plane shape always agrees with the role declared in the chain
terminal. The PPG validator enforces this at chain construction;
malformed chains never reach the wire format.

Plane payloads are aligned to their codec's preferred alignment. The
writer pads with zeros if necessary; the reader trusts the alignment
and may panic on misaligned access.

The tensor index is sorted by name hash. The reader uses a binary
search and would return wrong results on an unsorted index.

The reader does check the magic, the extension-table hash, and the
EOF sentinel. The first two protect against accidental corruption;
the last protects against truncation.

## Why blake3 and not sha256

Two reasons. Blake3 is faster on the workstation CPUs PTWM targets
for compression work, by about a factor of three on the inputs that
matter. The speed matters because the extension-table hash is
computed on every file write and verified on every file read; the
manifest plus binary hash is computed when verifying a signed
extension bundle. A slower hash would dominate the open path on
small files.

Blake3 also avoids the cryptographic agility surface that PTWM has
no use for. The library is committed to a single hash function;
adding a second one to support sha256 callers would mean every plane
hash needs a discriminator. The choice may need to be revisited if
NIST publishes a new standard that supplants blake3 in policy
documents, but the migration cost is bounded by the number of
on-disk hash fields, which is small.

## Sentinel public key

Built-in contributions use a canonical id derived from a sentinel
public key, `BUILTIN_PUBKEY = [0xBB; 32]`. The sentinel is treated as
trusted by construction; the verifier short-circuits before checking
the keyring when it sees an id derived from this key. The sentinel
bytes are recognisable, which helps when reading hex dumps of an
extension table during debugging.

A real Ed25519 public key would never have all bytes equal to
0xBB, since the key derivation would require a chosen-private-key
attack on Ed25519 to construct one. The probability of accidentally
matching the sentinel from a real key is `2^-256`, which is the
collision bound the codebase relies on elsewhere.
