<script setup lang="ts">
import type { ApiLanguage } from '~/types/api'

defineProps<{ code: string, language: ApiLanguage }>()
</script>

<template>
  <pre
    class="overflow-x-auto rounded-md border border-(--ui-border) bg-(--ui-bg-muted) p-4 font-mono text-sm whitespace-pre"
    :data-language="language"
  ><code>{{ code }}</code></pre>
</template>
