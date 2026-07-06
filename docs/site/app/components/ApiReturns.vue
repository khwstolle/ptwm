<script setup lang="ts">
import type { ApiReturn } from '~/types/api'

defineProps<{ returns: ApiReturn }>()
</script>

<template>
  <div class="my-3 border-l-2 border-(--ui-border-accented) pl-3">
    <div class="mb-1 text-xs font-semibold tracking-wider uppercase text-(--ui-text-dimmed)">
      Returns
    </div>
    <div class="text-sm">
      <code v-if="returns.type" class="font-mono">{{ returns.type }}</code>
      <span v-if="returns.type && returns.description"> — </span>
      <span v-if="returns.description">{{ returns.description }}</span>
    </div>
  </div>
</template>
