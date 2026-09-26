<script setup lang="ts">
/** 标题上方的眉题：所属板块 · 难度 · 预计阅读时间；以及可选的前置知识。 */
import { computed, onMounted, ref, watch } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'

const { frontmatter } = useData()
const route = useRoute()
const minutes = ref<number | null>(null)

const kicker = computed<string | undefined>(() => frontmatter.value.kicker)
const level = computed<string | undefined>(() => frontmatter.value.level)
const prereq = computed<{ text: string; link: string }[]>(() => frontmatter.value.prereq ?? [])

function measure(): void {
  const doc = document.querySelector('.vp-doc')
  if (!doc) return
  const text = doc.textContent ?? ''
  const cjk = (text.match(/[㐀-鿿]/g) ?? []).length
  const words = (text.replace(/[㐀-鿿]/g, ' ').match(/[A-Za-z0-9]+/g) ?? []).length
  // 中文约 400 字/分钟，英文约 200 词/分钟，公式与图另计一点余量
  const m = cjk / 400 + words / 200 + doc.querySelectorAll('.katex-display, .mermaid-figure, .lineage').length * 0.4
  minutes.value = Math.max(1, Math.round(m))
}

onMounted(() => setTimeout(measure, 50))
watch(() => route.path, () => setTimeout(measure, 120))
</script>

<template>
  <div v-if="kicker || level || prereq.length" class="page-meta">
    <div class="page-meta-row">
      <span v-if="kicker" class="page-kicker">{{ kicker }}</span>
      <span v-if="level" class="page-pill">{{ level }}</span>
      <span v-if="minutes" class="page-pill">约 {{ minutes }} 分钟</span>
    </div>
    <div v-if="prereq.length" class="page-prereq">
      <span class="page-prereq-label">建议先读</span>
      <a v-for="p in prereq" :key="p.link" :href="withBase(p.link)">{{ p.text }}</a>
    </div>
  </div>
</template>

<style scoped>
.page-meta {
  margin: 0 0 14px;
}
.page-meta-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.page-kicker {
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--vp-c-brand-1);
  margin-right: 4px;
}
.page-pill {
  font-size: 12px;
  padding: 3px 9px;
  border-radius: 999px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
}
.page-prereq {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-top: 10px;
  font-size: 13px;
}
.page-prereq-label {
  color: var(--vp-c-text-3);
}
.page-prereq a {
  color: var(--vp-c-text-2);
  text-decoration: none;
  border-bottom: 1px dashed var(--vp-c-border);
}
.page-prereq a:hover {
  color: var(--vp-c-brand-1);
  border-bottom-color: var(--vp-c-brand-1);
}
</style>
