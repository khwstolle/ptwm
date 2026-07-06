<script setup lang="ts">
const { footer } = useAppConfig()

const columns = [{
  label: 'Documentation',
  children: [{
    label: 'Quickstart',
    to: '/10.getting-started/quickstart'
  }, {
    label: 'Configuration',
    to: '/10.getting-started/configuration'
  }, {
    label: 'Integrations',
    to: '/10.getting-started/integrations'
  }]
}, {
  label: 'Architecture',
  children: [{
    label: 'Codecs',
    to: '/20.concepts/codecs'
  }, {
    label: 'Transforms',
    to: '/20.concepts/transforms'
  }, {
    label: 'Header Format',
    to: '/20.concepts/header'
  }]
}, {
  label: 'Community & Project',
  children: [{
    label: 'GitHub Repository',
    to: 'https://github.com/khwstolle/ptwm',
    target: '_blank'
  }, {
    label: 'Issue Tracker',
    to: 'https://github.com/khwstolle/ptwm/issues',
    target: '_blank'
  }, {
    label: 'PyPI Package',
    to: 'https://pypi.org/project/ptwm/',
    target: '_blank'
  }]
}]
</script>

<template>
  <UFooter :ui="{ container: '!max-w-none' }">
    <UFooterColumns :links="columns" />

    <template #left>
      {{ footer.credits }}
    </template>

    <template #right>
      <UColorModeButton v-if="footer?.colorMode" />

      <template v-if="footer?.links">
        <UButton
          v-for="(link, index) of footer?.links"
          :key="index"
          v-bind="{ color: 'neutral', variant: 'ghost', ...link }"
        />
      </template>
    </template>
  </UFooter>
</template>
