const repoUrl = 'https://github.com/khwstolle/ptwm'
const siteUrl = process.env.NUXT_PUBLIC_SITE_URL ?? 'https://ptwm.khws.io'

export default defineNuxtConfig({
  compatibilityDate: '2025-01-01',
  future: { compatibilityVersion: 4 },

  srcDir: 'app/',

  modules: [
    '@nuxt/eslint',
    '@nuxt/fonts',
    '@nuxt/image',
    '@nuxt/ui',
    '@nuxt/content',
    'nuxt-llms',
  ],

  devtools: { enabled: true },

  css: ['~/assets/css/main.css'],

  app: {
    head: {
      htmlAttrs: { lang: 'en' },
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'theme-color', content: '#0b0b0f' },
        {
          name: 'description',
          content:
            'PTWM — lossless compression for PyTorch model weights. Exponent-plane separation, microscaling-aware codecs, random-access container.',
        },
      ],
      link: [{ rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    },
  },

  runtimeConfig: {
    public: {
      siteUrl,
      repoUrl,
      releaseRef: process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev',
    },
  },

  site: {
    url: siteUrl,
    name: 'PTWM',
  },

  content: {
    build: {
      markdown: {
        highlight: {
          theme: {
            default: 'github-light',
            dark: 'github-dark',
          },
          langs: ['python', 'rust', 'bash', 'shell', 'toml', 'yaml', 'json', 'diff', 'vue', 'ts'],
        },
        toc: { depth: 3, searchDepth: 3 },
      },
    },
    experimental: {
      sqliteConnector: 'native',
    },
  },

  experimental: {
    asyncContext: true,
  },

  // @nuxt/ui bundles @nuxt/icon and pre-installs lucide. Force local
  // bundling so the icons resolve offline at build time.
  icon: {
    serverBundle: 'local',
  },

  // LLM-friendly index. Generates /llms.txt + /llms-full.txt and /raw/<path>.md.
  llms: {
    domain: siteUrl,
    title: 'PTWM',
    description:
      'PTWM (PyTorch Weights Manifest) is a lossless compression library for PyTorch model weights. The .ptwm container ships a random-access tensor index, hash-verified payloads, and per-plane codec dispatch. Optimised for IEEE 754 floats via exponent-plane separation, and for microscaling formats (MXFP4 / NVFP4) via Order-1 ScaleAC on the scale plane.',
    full: {
      title: 'PTWM — full documentation',
      description:
        'Complete reference for PTWM: concepts, guides, Python API, Rust API, and benchmarks.',
    },
    sections: [
      {
        title: 'Concepts',
        contentCollection: 'docs',
        contentFilters: [{ field: 'path', operator: 'LIKE', value: '/concepts%' }],
      },
      {
        title: 'Guides',
        contentCollection: 'docs',
        contentFilters: [{ field: 'path', operator: 'LIKE', value: '/guides%' }],
      },
      {
        title: 'API',
        contentCollection: 'docs',
        contentFilters: [{ field: 'path', operator: 'LIKE', value: '/api%' }],
      },
    ],
  },

  fonts: {
    // Three-family stack tuned for a technical / academic audience:
    //
    //   Geist           — display / headings. Variable sans (wght 100-900)
    //                     designed by Vercel / Basement Studio for
    //                     engineering UIs. Sharp terminals, geometric
    //                     construction, distinct from Inter / Roboto.
    //   IBM Plex Sans   — body face. Engineered for technical reading,
    //                     visually distinct from Roboto / Inter.
    //   IBM Plex Mono   — code / kbd / samp / pre. Shares design DNA with
    //                     Plex Sans so inline code sits cleanly inside
    //                     body copy.
    families: [
      {
        name: 'Geist',
        provider: 'google',
        weights: ['100 900'],
        styles: ['normal'],
        subsets: ['latin', 'latin-ext'],
      },
      {
        name: 'IBM Plex Sans',
        provider: 'google',
        weights: [400, 500, 600, 700],
        styles: ['normal', 'italic'],
        subsets: ['latin', 'latin-ext'],
      },
      {
        name: 'IBM Plex Mono',
        provider: 'google',
        weights: [400, 500, 600],
        styles: ['normal', 'italic'],
        subsets: ['latin', 'latin-ext'],
      },
    ],
    defaults: {
      weights: [400, 500, 600, 700],
      styles: ['normal', 'italic'],
      subsets: ['latin', 'latin-ext'],
    },
  },

  eslint: {
    config: {
      stylistic: {
        commaDangle: 'always-multiline',
        braceStyle: '1tbs',
      },
    },
  },

  nitro: {
    // `static` produces plain prerendered HTML in `.output/public/`.
    // We deliberately avoid `cloudflare_*` presets — they wire @nuxt/content
    // v3 to D1 at runtime. With `static`, content is baked into HTML at build
    // time (via better-sqlite3) and the bundle deploys as-is.
    preset: 'static',
    prerender: {
      crawlLinks: true,
      routes: ['/', '/api', '/api/python', '/api/rust'],
      autoSubfolderIndex: false,
      failOnError: false,
    },
  },

  typescript: {
    strict: true,
    typeCheck: false,
  },
})
