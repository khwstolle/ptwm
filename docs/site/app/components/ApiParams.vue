<script setup lang="ts">
import type { ApiParam } from '~/types/api'

defineProps<{ params: ApiParam[] }>()
</script>

<template>
  <UTable
    :data="params.map(p => ({
      name: p.name + (p.kind === 'var-positional' ? '*' : p.kind === 'var-keyword' ? '**' : '') + (p.default ? ` = ${p.default}` : ''),
      type: p.type ?? '—',
      description: p.description ?? '',
    }))"
    :columns="[
      { accessorKey: 'name', header: 'Name', meta: { class: { td: 'font-mono whitespace-nowrap' } } },
      { accessorKey: 'type', header: 'Type', meta: { class: { td: 'font-mono text-(--ui-text-muted)' } } },
      { accessorKey: 'description', header: 'Description' },
    ]"
    class="my-3"
  />
</template>
