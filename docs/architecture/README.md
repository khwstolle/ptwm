# PTWM architecture

This directory contains narrative documentation of PTWM's internal
structure. It is aimed at readers who want to understand why the
codebase is organised the way it is, not what each function does.

The top-level [`README.md`](../../README.md) is for users who want to
compress weights with PTWM. The documentation here is for contributors
who want to extend, refactor, or audit it.

## Reading order

A first-time reader can take these in order. Each builds on the
previous one.

1. `wire-format.md` covers the on-disk container, since the wire
   format is the most stable surface in the codebase and every other
   subsystem either writes to it or reads from it.
2. `ppg.md` describes the preprocessing graph, which decides what
   bytes the entropy coder ever sees. The compression-ratio story
   lives almost entirely in this stage.
3. `codec-dispatch.md` covers the trial-encode loop, which picks an
   entropy coder per plane after the PPG runs.
4. `extension-system.md` describes the thirteen contribution kinds
   and the three flavours (WASM, native, host), and explains why the
   architecture had to absorb a substantial extension surface.
5. `trust-model.md` documents the signature and capability checks
   that gate third-party extensions. This is security-sensitive and
   worth reading even if you only plan to use built-in codecs, since
   it constrains what new codecs can do.

## Cross-references

When code comments refer back to these docs they cite the file by
name, for example "see `architecture/ppg.md`". The intent is that
inline comments stay short and the long-form reasoning lives here.

## Scope

These documents do not duplicate type signatures or function lists.
They cover the design questions that a reader cannot answer by
following `cargo doc` or by reading the test suite. If a fact is
already obvious from the code, it is not repeated here.
