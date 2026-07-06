# PTWM documentation site

A Nuxt 4 + Cloudflare Workers documentation site for PTWM. The site
unifies handwritten content (concepts, guides, benchmarks) with
auto-generated API reference pages for the Python package and the Rust
crate.

## Architecture in 30 seconds

```
docs/site/
  app/           Nuxt source — layouts, pages, components, design tokens
  content/       Handwritten Markdown + generated content/api/**
  scripts/       Python + Rust extractors and the normalizer
  .docgen/       (gitignored) JSON dumps from griffe and rustdoc
  .output/       (gitignored) Nuxt build output, deployed to Workers
  wrangler.toml  Cloudflare Workers + Static Assets deploy config
```

The cohesion guarantee: both Python (Griffe) and Rust (`cargo rustdoc
-Z unstable-options --output-format json`, on the nightly toolchain
that fenix provides) emit JSON; the normaliser collapses both into a
single MDC schema; the same Vue component renders every API entry —
Python and Rust items share the exact DOM structure.

## Local development

### Prerequisites

```bash
nix develop .#docs
```

The `docs` devShell provides Node 22, pnpm 9, wrangler, pagefind, uv,
prek, and a **nightly Rust toolchain** via the `fenix` flake input
(rustdoc JSON requires `-Z unstable-options`, which only nightly
accepts). No separate `rustup` install needed.

Bump the nightly with:

```bash
nix flake update fenix
```

Note: the docs shell ships nightly Rust *instead of* stable. For
regular `ptwm-core` development, use the default shell (`nix develop`),
which ships stable.

### Install JS deps

```bash
cd docs/site
pnpm install
```

### Regenerate the API reference

```bash
pnpm run docgen
```

This runs:

1. `scripts/docgen/extract-python.py` → `.docgen/python.json` via Griffe.
2. `scripts/docgen/extract-rust.sh` → `.docgen/rust.json` via
   `cargo rustdoc --output-format=json` (nightly toolchain from fenix).
3. `scripts/docgen/normalize.mjs` → `content/api/{python,rust}/**.md`.

To skip one side (handy when iterating on prose without re-running cargo):

```bash
PTWM_SKIP_RUST_DOCGEN=1 pnpm run docgen
PTWM_SKIP_PYTHON_DOCGEN=1 pnpm run docgen
```

### Run the dev server

```bash
pnpm run dev
```

Site at http://localhost:3000. Edits to `content/**/*.md` and `app/**`
hot-reload.

### Build a production bundle

```bash
pnpm run build
pnpm run search:index
```

The first command builds Nuxt with the `cloudflare_module` Nitro
preset, emitting `.output/server/index.mjs` and `.output/public/**`.
The second runs Pagefind over the static output to build the search
index under `/_pagefind/`.

### Local preview of the production bundle

```bash
pnpm run preview
```

### Deploy

`.github/workflows/docs.yml` automates deployment. A push to `master`
triggers a production deploy; a pull request triggers a preview deploy
and links the preview URL back into the PR.

Manual deploy from your machine (requires `wrangler login` and a
Workers project named `ptwm-docs`):

```bash
pnpm run deploy
```

## Authoring conventions

### Page ordering

Pages are ordered by filename prefix:

```
content/1.quickstart.md       → /quickstart
content/2.concepts/1.index.md → /concepts/
content/2.concepts/2.ppg.md   → /concepts/ppg
```

The router strips the numeric prefix from the URL. Keep prefixes
gap-free per section so reordering takes a single rename.

### Frontmatter

```yaml
---
title: Required — appears in <title> and navigation
description: Optional — meta description
---
```

API pages also carry `language`, `kind`, and `summary`. The
docgen normaliser populates these fields. Do not edit generated pages
under `content/api/python/**` or `content/api/rust/**` — every
`pnpm run docgen` re-emits them.

### Embedding components in Markdown

The site uses MDC (Markdown Components) from `@nuxt/content`. Any
component in `app/components/` can be embedded:

```mdc
::api-item
---
id: python:ptwm.Compressor
language: python
kind: type
name: Compressor
qualifiedName: ptwm.Compressor
signature: 'class Compressor(config: CompressionConfig)'
summary: Multi-tensor compressor for the .ptwm container format.
---
::
```

The normaliser emits one `::api-item` block per documented entry into
the generated tree.

## Cohesion check

If a Python item and a Rust item ever render with visibly different
typography, signature treatment, or layout, treat it as a bug. Both go
through `app/components/ApiItem.vue`; both inherit the same Tailwind
tokens; both invoke the same `<ApiSignature>` (which feeds
`<ProseCode>` for syntax highlighting). Only the kind label inside
`<ApiKind>` (`fn` for Rust vs. `function` for Python) and the
syntax-highlighting language hint vary by language.

If the schemas drift upstream (Griffe or rustdoc), absorb the change in
the normaliser. Do not introduce conditional rendering inside
`<ApiItem>`.

## Layout map

```
app/
  app.vue                  Entry — picks 'home' or 'default' layout per route
  layouts/
    default.vue            Header + sidebar + content + on-page TOC
    home.vue               Header + hero + footer
  pages/
    index.vue              The home page
    [...slug].vue          Catch-all for content collection
  components/
    Site*.vue              Chrome (header, footer, search, sidebar, TOC)
    Hero.vue               Home-page hero
    Api*.vue               Unified API renderers — touch with care
  types/
    api.ts                 The shared ApiItem schema
  assets/css/
    main.css               Tailwind v4 entry + design tokens
```

## Troubleshooting

**`extract-rust.sh: nightly Rust required`** — you are in the default
shell (which ships stable Rust), not the docs shell. Run `nix develop
.#docs` first, or set `PTWM_SKIP_RUST_DOCGEN=1` to skip Rust API
extraction.

**`Error: griffe is not installed`** — run `uv sync --group docs` at
the repository root.

**Pagefind index empty** — make sure `pnpm run build` ran before
`pnpm run search:index`. Pagefind crawls `.output/public/`.

**Cloudflare deploy fails on `wrangler versions upload`** — your
Cloudflare account needs Workers Static Assets enabled for the project,
and the `CF_API_TOKEN` secret must include "Workers Scripts: Edit" and
"Account → Workers KV Storage: Edit" permissions.
