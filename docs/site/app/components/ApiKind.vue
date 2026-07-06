<script setup lang="ts">
import type { ApiKind, ApiLanguage } from '~/types/api'
import { KIND_LANGUAGE_LABEL } from '~/types/api'
import { computed } from 'vue'

const props = defineProps<{
  kind: ApiKind
  language: ApiLanguage
}>()

const label = computed(() => KIND_LANGUAGE_LABEL[props.language][props.kind])
const languageLabel = computed(() => (props.language === 'rust' ? 'rs' : 'py'))
</script>

<template>
  <UBadge
    variant="subtle"
    size="xs"
    class="font-mono uppercase tracking-wider"
    :class="`kind-${kind}`"
  >
    <span class="opacity-70">{{ languageLabel }}</span>
    <span>{{ label }}</span>
  </UBadge>
</template>

<style scoped>
.kind-module    { background: color-mix(in oklch, var(--color-kind-module) 14%, transparent); color: var(--color-kind-module); }
.kind-type      { background: color-mix(in oklch, var(--color-kind-type) 14%, transparent); color: var(--color-kind-type); }
.kind-function  { background: color-mix(in oklch, var(--color-kind-function) 14%, transparent); color: var(--color-kind-function); }
.kind-method    { background: color-mix(in oklch, var(--color-kind-method) 14%, transparent); color: var(--color-kind-method); }
.kind-property  { background: color-mix(in oklch, var(--color-kind-property) 14%, transparent); color: var(--color-kind-property); }
.kind-constant  { background: color-mix(in oklch, var(--color-kind-constant) 14%, transparent); color: var(--color-kind-constant); }
</style>
