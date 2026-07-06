#!/usr/bin/env node
// Orchestrator for the docs-site docgen pipeline.
// Runs the Python and Rust extractors, then normalizes their JSON output into
// MDC pages under docs/site/content/api/{python,rust}/**.

import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const SCRIPT_DIR = dirname(__filename)
const SITE_DIR = resolve(SCRIPT_DIR, '..', '..')

function run(cmd, args, opts = {}) {
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, { stdio: 'inherit', cwd: SITE_DIR, ...opts })
    child.on('exit', (code) => {
      if (code === 0) res(undefined)
      else rej(new Error(`${cmd} ${args.join(' ')} exited with ${code}`))
    })
    child.on('error', rej)
  })
}

const skipPython = process.env.PTWM_SKIP_PYTHON_DOCGEN === '1'
const skipRust = process.env.PTWM_SKIP_RUST_DOCGEN === '1'

if (!skipPython) {
  console.log('docgen: extracting Python API via griffe')
  try {
    await run('python', ['scripts/docgen/extract-python.py'])
  } catch (err) {
    console.error('docgen: python extraction failed —', err.message)
    if (process.env.PTWM_DOCGEN_STRICT === '1') throw err
  }
} else {
  console.log('docgen: skipping python extraction (PTWM_SKIP_PYTHON_DOCGEN=1)')
}

if (!skipRust) {
  console.log('docgen: extracting Rust API via cargo rustdoc (nightly)')
  try {
    await run('bash', ['scripts/docgen/extract-rust.sh'])
  } catch (err) {
    console.error('docgen: rust extraction failed —', err.message)
    if (process.env.PTWM_DOCGEN_STRICT === '1') throw err
  }
} else {
  console.log('docgen: skipping rust extraction (PTWM_SKIP_RUST_DOCGEN=1)')
}

console.log('docgen: normalizing into content/api/**')
await run('node', ['scripts/docgen/normalize.mjs'])

console.log('docgen: done')
