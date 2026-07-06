export type ApiLanguage = 'python' | 'rust'

export type ApiKind =
  | 'module'
  | 'type'
  | 'function'
  | 'method'
  | 'property'
  | 'constant'

export interface ApiParam {
  name: string
  type?: string
  default?: string
  description?: string
  kind?: 'positional' | 'keyword' | 'var-positional' | 'var-keyword'
}

export interface ApiReturn {
  type?: string
  description?: string
}

export interface ApiRaise {
  type: string
  description?: string
}

export interface ApiSource {
  file: string
  line: number
}

export interface ApiDeprecation {
  since: string
  message?: string
}

export interface ApiItem {
  id: string
  language: ApiLanguage
  kind: ApiKind
  name: string
  qualifiedName: string
  signature: string
  summary?: string
  description?: string
  parameters?: ApiParam[]
  returns?: ApiReturn
  raises?: ApiRaise[]
  examples?: string[]
  source?: ApiSource
  parent?: string
  children?: string[]
  deprecated?: ApiDeprecation
}

export const KIND_LABEL: Record<ApiKind, string> = {
  module: 'module',
  type: 'type',
  function: 'function',
  method: 'method',
  property: 'property',
  constant: 'constant',
}

export const KIND_LANGUAGE_LABEL: Record<ApiLanguage, Record<ApiKind, string>> = {
  python: {
    module: 'module',
    type: 'class',
    function: 'function',
    method: 'method',
    property: 'attribute',
    constant: 'constant',
  },
  rust: {
    module: 'mod',
    type: 'type',
    function: 'fn',
    method: 'fn',
    property: 'field',
    constant: 'const',
  },
}
