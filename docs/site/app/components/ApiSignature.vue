<script setup lang="ts">
import type { ApiLanguage } from '~/types/api'

defineProps<{
  signature: string
  language: ApiLanguage
}>()
</script>

<template>
  <pre
    class="my-0 overflow-x-auto rounded-md border border-(--ui-border) bg-(--ui-bg-muted) px-4 py-3 font-mono text-sm whitespace-pre-wrap break-words"
    :data-language="language"
  ><code>{{ signature }}</code></pre>
</template>
