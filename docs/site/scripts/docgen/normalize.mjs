#!/usr/bin/env node
// Normalize the Python (Griffe) and Rust (rustdoc) JSON dumps into a single
// content/api/{python,rust}/**.md tree of MDC pages.
//
// Each page contains one `::api-item` MDC block per documented entry, plus a
// frontmatter summary used by the navigation. The Vue `<ApiItem>` component
// renders the MDC block, guaranteeing that Python and Rust items share
// identical DOM treatment.
//
// Usage:
//   node scripts/docgen/normalize.mjs

import { readFile, writeFile, mkdir, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import YAML from 'js-yaml'

const __filename = fileURLToPath(import.meta.url)
const SCRIPT_DIR = dirname(__filename)
const SITE_DIR = resolve(SCRIPT_DIR, '..', '..')
const DOCGEN_DIR = join(SITE_DIR, '.docgen')
const CONTENT_API = join(SITE_DIR, 'content', 'api')

const PY_JSON = join(DOCGEN_DIR, 'python.json')
const RS_JSON = join(DOCGEN_DIR, 'rust.json')

function ensureItem(raw, language) {
  const item = {
    id: raw.id,
    language,
    kind: raw.kind,
    name: raw.name,
    qualifiedName: raw.qualifiedName,
    signature: raw.signature,
  }
  if (raw.summary) item.summary = raw.summary
  if (raw.description) item.description = raw.description
  if (raw.parameters?.length) item.parameters = raw.parameters
  if (raw.returns) item.returns = raw.returns
  if (raw.raises?.length) item.raises = raw.raises
  if (raw.examples?.length) item.examples = raw.examples
  if (raw.source) item.source = raw.source
  if (raw.parent) item.parent = raw.parent
  if (raw.deprecated) item.deprecated = raw.deprecated
  return item
}

function groupByModule(items, language) {
  const sep = language === 'rust' ? '::' : '.'
  const groups = new Map()

  // Real modules only — classes share the `parent` field shape but are not
  // modules and must not anchor their own page.
  const modules = new Set()
  for (const raw of items) {
    if (raw.kind === 'module') modules.add(raw.qualifiedName)
  }

  const climbToModule = (qualifiedName, parent) => {
    const segments = qualifiedName.split(sep)
    for (let i = segments.length - 1; i >= 1; i -= 1) {
      const candidate = segments.slice(0, i).join(sep)
      if (modules.has(candidate)) return candidate
    }
    return parent ?? ''
  }

  for (const raw of items) {
    const item = ensureItem(raw, language)
    const moduleKey = item.kind === 'module'
      ? item.qualifiedName
      : (climbToModule(item.qualifiedName, item.parent) || 'root')
    if (!groups.has(moduleKey)) {
      groups.set(moduleKey, { language, module: moduleKey, items: [] })
    }
    groups.get(moduleKey).items.push(item)
  }

  return groups
}

// File-stem slugs that collide with Nuxt Content's directory-index
// convention (`<dir>/index.md` is treated as the directory's own page).
// A real submodule named `index` (e.g. `ptwm_core::index`) would otherwise
// shadow its parent module's page on the same route. Underscores survive
// the prerender's URL normalisation (a trailing `-` does not), so the
// slug and the generated URL stay in sync.
const RESERVED_LEAF_NAMES = new Set(['index'])

function slugFor(qualifiedName, language) {
  const sep = language === 'rust' ? '::' : '.'
  const segments = qualifiedName.split(sep).map(s => s.replace(/[^a-zA-Z0-9._-]/g, '-'))
  const last = segments[segments.length - 1]
  if (RESERVED_LEAF_NAMES.has(last)) {
    segments[segments.length - 1] = `${last}_module`
  }
  return segments.join('/')
}

function renderItemBlock(item) {
  // MDC component invocation. The YAML body is parsed into props by @nuxt/content.
  const yamlBody = YAML.dump(item, { lineWidth: -1, noRefs: true }).trimEnd()
  return `::api-item\n---\n${yamlBody}\n---\n::\n`
}

async function writeModule(group) {
  const moduleName = group.module
  const slug = slugFor(moduleName, group.language)
  const file = join(CONTENT_API, group.language, `${slug}.md`)

  const moduleItem = group.items.find(i => i.kind === 'module' && i.qualifiedName === moduleName)
  const moduleSummary = moduleItem?.summary ?? ''

  // Sort: module first, then alphabetical by qualifiedName. This keeps
  // class methods adjacent to their class on the page (`Foo`, `Foo.__init__`,
  // `Foo.bar`, `FooBar`, …) without a separate level of grouping.
  const sortedItems = [...group.items].sort((a, b) => {
    if (a.kind === 'module' && b.kind !== 'module') return -1
    if (b.kind === 'module' && a.kind !== 'module') return 1
    return a.qualifiedName.localeCompare(b.qualifiedName)
  })

  const frontmatter = YAML.dump({
    title: moduleName,
    language: group.language,
    kind: 'module',
    summary: moduleSummary,
    navigation: {
      title: moduleName.split(group.language === 'rust' ? '::' : '.').pop(),
    },
  }, { lineWidth: -1 }).trimEnd()

  const body = sortedItems.map(renderItemBlock).join('\n')

  const content = `---\n${frontmatter}\n---\n\n# \`${moduleName}\`\n\n${moduleSummary ? `${moduleSummary}\n\n` : ''}${body}`

  await mkdir(dirname(file), { recursive: true })
  await writeFile(file, content)
  return file
}

async function readJson(path) {
  if (!existsSync(path)) {
    console.warn(`normalize.mjs: ${path} missing — skipping`)
    return null
  }
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (err) {
    console.error(`normalize.mjs: failed to parse ${path}:`, err.message)
    return null
  }
}

function convertRustdoc(rustdoc) {
  // rustdoc emits `index: { id: Item }` and `paths: { id: { path, kind } }`.
  // We walk the index and produce normalized items.
  if (!rustdoc || rustdoc.skipped) return []

  const items = []
  const { index, paths } = rustdoc
  if (!index) return []

  const allowedKinds = new Set([
    'module', 'struct', 'enum', 'function', 'trait', 'typedef', 'type_alias',
    'union', 'constant', 'static', 'structfield', 'enumvariant',
    // Methods don't appear in `paths` because they're reached via impl blocks.
    // We skip them at this layer; the parent type's signature documents the
    // surface, and full per-method docs can be linked out to docs.rs.
  ])

  for (const [id, item] of Object.entries(index)) {
    const itemKind = item.inner ? Object.keys(item.inner)[0] : null
    if (!itemKind || !allowedKinds.has(itemKind)) continue

    // Restrict to public API. `pub(crate)` and friends serialise as
    // `{ restricted: ... }`; only `"public"` reaches end users.
    if (!isPublicVisibility(item.visibility)) continue

    const pathInfo = paths?.[id]
    if (!pathInfo || !Array.isArray(pathInfo.path) || pathInfo.path.length === 0) continue
    const qualifiedName = pathInfo.path.join('::')

    // Hide any path threading through a private-by-convention module
    // (`mod _foo;` is unusual in Rust; here it catches stray underscore
    // segments and keeps the API tree consistent with Python's filter).
    if (qualifiedName.split('::').some(seg => seg.startsWith('_'))) continue

    const mappedKind = mapRustKind(itemKind)
    if (!mappedKind) continue

    const signature = formatRustSignature(item, itemKind)
    if (!signature) continue

    const summary = (item.docs ?? '').trim().split('\n\n')[0]?.replace(/\n/g, ' ').trim() ?? ''
    const description = (item.docs ?? '').trim().split('\n\n').slice(1).join('\n\n').trim()
    const examples = extractRustExamples(item.docs ?? '')

    const parentPath = pathInfo ? pathInfo.path.slice(0, -1).join('::') : ''

    const out = {
      id: `rust:${qualifiedName}`,
      language: 'rust',
      kind: mappedKind,
      name: item.name ?? qualifiedName.split('::').pop(),
      qualifiedName,
      signature,
      summary,
    }
    if (description) out.description = description
    if (examples.length) out.examples = examples
    if (item.span) {
      out.source = {
        file: item.span.filename.replace(/^\.?\//, ''),
        line: item.span.begin?.[0] ?? 1,
      }
    }
    if (parentPath) out.parent = parentPath
    if (item.deprecation) out.deprecated = { since: item.deprecation.since ?? 'unknown', message: item.deprecation.note ?? '' }

    items.push(out)
  }

  return items
}

function isPublicVisibility(vis) {
  // rustdoc emits `"public"` for `pub`, the string `"default"` for
  // re-exports / inherent items, and `{ restricted: ... }` for `pub(crate)`
  // or `pub(in path)`. Crate roots arrive with `null`. Treat `public`,
  // `default`, and `null` as visible; reject anything `restricted`.
  if (vis === undefined || vis === null) return true
  if (typeof vis === 'string') return vis === 'public' || vis === 'default'
  return false
}

function mapRustKind(kind) {
  switch (kind) {
    case 'module': return 'module'
    case 'struct':
    case 'enum':
    case 'trait':
    case 'typedef':
    case 'type_alias':
    case 'union':
      return 'type'
    case 'function':
      return 'function'
    case 'method':
      return 'method'
    case 'structfield':
    case 'enumvariant':
      return 'property'
    case 'constant':
    case 'static':
      return 'constant'
    default:
      return null
  }
}

function formatRustSignature(item, kind) {
  const name = item.name ?? '?'
  switch (kind) {
    case 'module':
      return `mod ${name}`
    case 'struct': {
      const struct = item.inner?.struct
      const structKind = struct?.kind
      if (typeof structKind === 'string') return `pub struct ${name};` // unit
      if (structKind && 'tuple' in structKind) return `pub struct ${name}(...);`
      return `pub struct ${name}`
    }
    case 'enum':
      return `pub enum ${name}`
    case 'trait':
      return `pub trait ${name}`
    case 'typedef':
    case 'type_alias':
      return `pub type ${name}`
    case 'union':
      return `pub union ${name}`
    case 'constant':
      return `pub const ${name}`
    case 'static':
      return `pub static ${name}`
    case 'function':
    case 'method':
      return formatRustFnSignature(name, item.inner?.[kind])
    default:
      return name
  }
}

function formatRustFnSignature(name, fn) {
  if (!fn) return `pub fn ${name}(...)`
  // rustdoc JSON schema (nightly 2026+): function/method.sig.{inputs,output}.
  // Older variants used .decl.{inputs,output}; accept either.
  const sig = fn.sig ?? fn.decl
  if (!sig) return `pub fn ${name}(...)`
  const inputs = (sig.inputs ?? []).map(([n, ty]) => {
    if (n === 'self' && ty && typeof ty === 'object' && ty.borrowed_ref) {
      const mut = ty.borrowed_ref.is_mutable || ty.borrowed_ref.mutable ? 'mut ' : ''
      return `&${mut}self`
    }
    if (n === 'self') return 'self'
    return `${n}: ${stringifyRustType(ty)}`
  }).join(', ')
  const output = sig.output ? ` -> ${stringifyRustType(sig.output)}` : ''
  return `pub fn ${name}(${inputs})${output}`
}

function stringifyRustType(ty) {
  if (!ty) return '?'
  if (typeof ty === 'string') return ty
  if (ty.resolved_path) {
    const rp = ty.resolved_path
    // 2026 schema: rp.path is a `::`-joined string. Older schemas had rp.name.
    if (typeof rp.path === 'string') {
      // Trim a leading "crate::" so on-page types match how users write them.
      const stripped = rp.path.replace(/^crate::/, '')
      // For long paths, prefer the last segment.
      const segments = stripped.split('::')
      const last = segments[segments.length - 1]
      const generics = formatGenericArgs(rp.args)
      return `${last}${generics}`
    }
    return rp.name ?? '?'
  }
  if (ty.primitive) return ty.primitive
  if (ty.generic) return ty.generic
  if (ty.borrowed_ref) {
    const br = ty.borrowed_ref
    const mut = br.is_mutable || br.mutable ? 'mut ' : ''
    return `&${mut}${stringifyRustType(br.type)}`
  }
  if (ty.tuple) return `(${ty.tuple.map(stringifyRustType).join(', ')})`
  if (ty.slice) return `[${stringifyRustType(ty.slice)}]`
  if (ty.array) return `[${stringifyRustType(ty.array.type)}; ${ty.array.len}]`
  if (ty.raw_pointer) {
    const rp = ty.raw_pointer
    const mut = rp.is_mutable || rp.mutable ? 'mut' : 'const'
    return `*${mut} ${stringifyRustType(rp.type)}`
  }
  if (ty.qualified_path) return ty.qualified_path.name ?? '?'
  if (ty.dyn_trait) return `dyn ${(ty.dyn_trait.traits ?? []).map(t => t.trait?.name ?? '?').join(' + ')}`
  if (ty.impl_trait) return `impl ${(ty.impl_trait ?? []).map(b => b.trait_bound?.trait?.name ?? '?').join(' + ')}`
  return '?'
}

function formatGenericArgs(args) {
  if (!args) return ''
  if (args.angle_bracketed && args.angle_bracketed.args?.length) {
    const parts = args.angle_bracketed.args.map((a) => {
      if (a.type) return stringifyRustType(a.type)
      if (a.lifetime) return a.lifetime
      return '?'
    })
    return `<${parts.join(', ')}>`
  }
  return ''
}

function extractRustExamples(docs) {
  // Accept fenced blocks tagged with `rust`, `no_run`, `ignore`, or untagged
  // (rustdoc treats untagged blocks as Rust by default).
  const matches = [...docs.matchAll(/```([^\n]*)\n([\s\S]*?)\n```/g)]
  return matches
    .filter(m => {
      const lang = m[1].trim().toLowerCase()
      return lang === '' || lang === 'rust' || lang.startsWith('no_run') || lang.startsWith('ignore') || lang.startsWith('compile_fail') || lang.startsWith('should_panic')
    })
    .map(m => m[2])
}

async function main() {
  // Reset generated tree.
  await rm(join(CONTENT_API, 'python'), { recursive: true, force: true })
  await rm(join(CONTENT_API, 'rust'), { recursive: true, force: true })
  await mkdir(join(CONTENT_API, 'python'), { recursive: true })
  await mkdir(join(CONTENT_API, 'rust'), { recursive: true })

  const py = await readJson(PY_JSON)
  const rs = await readJson(RS_JSON)

  const written = []
  const pythonGroups = []
  const rustGroups = []

  if (py?.items) {
    const groups = groupByModule(py.items, 'python')
    for (const group of groups.values()) {
      pythonGroups.push(group)
      written.push(await writeModule(group))
    }
  }
  if (rs) {
    const rsItems = convertRustdoc(rs)
    const groups = groupByModule(rsItems, 'rust')
    for (const group of groups.values()) {
      rustGroups.push(group)
      written.push(await writeModule(group))
    }
  }

  if (pythonGroups.length) {
    written.push(await writeLanguageIndex('python', pythonGroups))
  }
  if (rustGroups.length) {
    written.push(await writeLanguageIndex('rust', rustGroups))
  }

  await writeFile(join(CONTENT_API, 'python', '.gitkeep'), '')
  await writeFile(join(CONTENT_API, 'rust', '.gitkeep'), '')

  console.error(`normalize.mjs: wrote ${written.length} api pages`)
}

async function writeLanguageIndex(language, groups) {
  const sep = language === 'rust' ? '::' : '.'
  const label = language === 'python' ? 'Python API' : 'Rust API'
  const description = language === 'python'
    ? 'Reference for the `ptwm` Python package.'
    : 'Reference for the `ptwm-core` Rust crate.'

  // Group modules into top-level packages for the navigation.
  const sortedGroups = [...groups].sort((a, b) => a.module.localeCompare(b.module))

  const frontmatter = YAML.dump({
    title: label,
    language,
    kind: 'module',
    summary: description,
    navigation: { title: label },
  }, { lineWidth: -1 }).trimEnd()

  let body = `# ${label}\n\n${description}\n\n`
  body += sortedGroups.length === 1
    ? `${sortedGroups.length} module documented.\n\n`
    : `${sortedGroups.length} modules documented.\n\n`

  for (const group of sortedGroups) {
    const slug = slugFor(group.module, language)
    const url = `/api/${language}/${slug}`
    const moduleItem = group.items.find(i => i.kind === 'module' && i.qualifiedName === group.module)
    const summary = moduleItem?.summary ?? ''
    const symbolCount = group.items.filter(i => i.kind !== 'module').length
    body += `- [\`${group.module}\`](${url})`
    if (summary) body += ` — ${summary}`
    body += ` _(${symbolCount} symbol${symbolCount === 1 ? '' : 's'})_\n`
  }

  const file = join(CONTENT_API, language, 'index.md')
  await mkdir(dirname(file), { recursive: true })
  await writeFile(file, `---\n${frontmatter}\n---\n\n${body}`)
  return file
}

await main().catch((err) => {
  console.error('normalize.mjs failed:', err)
  process.exit(1)
})
