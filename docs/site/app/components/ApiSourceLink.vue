<script setup lang="ts">
import type { ApiSource } from '~/types/api'
import { computed } from 'vue'

const props = defineProps<{ source: ApiSource }>()
const runtimeConfig = useRuntimeConfig()

const url = computed(() => {
  const ref = runtimeConfig.public.releaseRef as string
  const refSeg = ref === 'dev' ? 'master' : ref
  return `${runtimeConfig.public.repoUrl}/blob/${refSeg}/${props.source.file}#L${props.source.line}`
})
</script>

<template>
  <UButton
    :to="url"
    target="_blank"
    rel="noopener noreferrer"
    variant="ghost"
    color="neutral"
    size="xs"
    icon="i-lucide-external-link"
    class="font-mono"
  >
    {{ source.file }}:{{ source.line }}
  </UButton>
</template>
