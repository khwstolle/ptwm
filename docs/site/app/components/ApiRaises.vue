<script setup lang="ts">
import type { ApiRaise } from '~/types/api'

defineProps<{ raises: ApiRaise[] }>()
</script>

<template>
  <div class="my-3 border-l-2 border-(--ui-border-accented) pl-3">
    <div class="mb-1 text-xs font-semibold tracking-wider uppercase text-(--ui-text-dimmed)">
      Raises
    </div>
    <ul class="space-y-1 text-sm">
      <li v-for="r in raises" :key="r.type">
        <code class="font-mono">{{ r.type }}</code>
        <span v-if="r.description"> — {{ r.description }}</span>
      </li>
    </ul>
  </div>
</template>
