<script setup lang="ts">
import type {
  ApiKind,
  ApiLanguage,
  ApiParam,
  ApiReturn,
  ApiRaise,
  ApiSource,
  ApiDeprecation,
} from '~/types/api'

const props = defineProps<{
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
  deprecated?: ApiDeprecation
}>()

const slug = props.id.replace(/[^a-zA-Z0-9._-]/g, '-')
</script>

<template>
  <section
    :id="slug"
    class="not-prose api-item my-6 overflow-hidden rounded-lg border border-(--ui-border) bg-(--ui-bg-elevated)"
    :data-language="language"
    :data-kind="kind"
  >
    <header class="flex flex-wrap items-baseline gap-3 px-4 pt-4 pb-2">
      <ApiKind :kind="kind" :language="language" />
      <h3 class="m-0 font-mono text-base font-semibold tracking-tight">
        <a :href="`#${slug}`" class="hover:text-(--ui-primary)">{{ qualifiedName }}</a>
      </h3>
      <ApiSourceLink v-if="source" :source="source" class="ml-auto" />
    </header>

    <UAlert
      v-if="deprecated"
      :title="`Deprecated since ${deprecated.since}`"
      :description="deprecated.message"
      color="warning"
      variant="subtle"
      icon="i-lucide-triangle-alert"
      class="mx-4 my-2"
    />

    <div class="px-4 pb-2">
      <ApiSignature :signature="signature" :language="language" />
    </div>

    <div class="space-y-3 px-4 pb-4 text-sm">
      <p v-if="summary" class="text-(--ui-text)">{{ summary }}</p>
      <div v-if="description" class="whitespace-pre-line text-(--ui-text-muted)">
        {{ description }}
      </div>

      <ApiParams v-if="parameters && parameters.length" :params="parameters" />
      <ApiReturns v-if="returns" :returns="returns" />
      <ApiRaises v-if="raises && raises.length" :raises="raises" />

      <div v-if="examples && examples.length" class="space-y-2">
        <div class="text-xs font-semibold tracking-wider uppercase text-(--ui-text-dimmed)">
          Example
        </div>
        <ApiExample v-for="(ex, i) in examples" :key="i" :code="ex" :language="language" />
      </div>
    </div>
  </section>
</template>
