<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import PageMeta from './components/PageMeta.vue'
import ReadingAssist from './components/ReadingAssist.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-top>
      <ReadingAssist />
    </template>
    <template #doc-before>
      <PageMeta />
    </template>
  </Layout>
</template>
