<script setup lang="ts">
/**
 * 术语悬浮卡：<Term t="grpo">GRPO</Term>。桌面端悬停、移动端点按，展示「说人话」解释并链接到术语表。
 */
import { computed, nextTick, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'

const props = defineProps<{ t: string }>()
const term = computed(() => data.glossary.find((g) => g.id === props.t))
const open = ref(false)
const anchor = ref<HTMLElement | null>(null)
const pos = ref({ left: 0, top: 0, above: false })
let closeTimer: number | undefined

async function show(): Promise<void> {
  window.clearTimeout(closeTimer)
  open.value = true
  await nextTick()
  const r = anchor.value?.getBoundingClientRect()
  if (!r) return
  const width = Math.min(340, window.innerWidth - 24)
  const left = Math.min(Math.max(12, r.left + r.width / 2 - width / 2), window.innerWidth - width - 12)
  const above = r.bottom + 220 > window.innerHeight && r.top > 240
  pos.value = { left, top: above ? r.top - 10 : r.bottom + 10, above }
}
function hide(): void {
  closeTimer = window.setTimeout(() => (open.value = false), 140)
}
function toggle(): void {
  open.value ? (open.value = false) : show()
}
onBeforeUnmount(() => window.clearTimeout(closeTimer))
</script>

<template>
  <span
    ref="anchor"
    class="term"
    tabindex="0"
    role="button"
    :aria-expanded="open"
    @mouseenter="show"
    @mouseleave="hide"
    @focus="show"
    @blur="hide"
    @click="toggle"
  ><slot>{{ term?.term ?? t }}</slot></span>
  <Teleport v-if="open && term" to="body">
    <div
      class="term-card"
      :class="{ above: pos.above }"
      :style="{ left: `${pos.left}px`, top: `${pos.top}px` }"
      role="tooltip"
      @mouseenter="show"
      @mouseleave="hide"
    >
      <p class="term-name">
        <b>{{ term.term }}</b>
        <span v-if="term.zh && !term.zh.startsWith(term.term)">{{ term.zh }}</span>
        <span v-else-if="term.zh && term.zh !== term.term">{{ term.zh.slice(term.term.length).trim() }}</span>
      </p>
      <p v-if="term.full" class="term-full">{{ term.full }}</p>
      <p class="term-plain"><span class="term-plain-label">说人话</span><span v-if="term.html?.plain" v-html="term.html.plain" /><template v-else>{{ term.plain }}</template></p>
      <a class="term-more" :href="withBase(`/glossary#${term.id}`)">术语表中查看 →</a>
    </div>
  </Teleport>
</template>

<style scoped>
.term {
  cursor: help;
  border-bottom: 1.5px dotted color-mix(in srgb, var(--vp-c-brand-2) 70%, transparent);
  transition: color 0.2s, border-color 0.2s;
}
.term:hover,
.term:focus-visible {
  color: var(--vp-c-brand-1);
  border-bottom-color: var(--vp-c-brand-1);
  outline: none;
}
</style>

<style>
.term-card {
  position: fixed;
  z-index: 150;
  width: min(340px, calc(100vw - 24px));
  padding: 14px 16px 12px;
  border-radius: 14px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-elv);
  box-shadow: 0 18px 48px rgba(20, 20, 20, 0.16);
  font-size: 14px;
  line-height: 1.7;
  animation: term-in 0.16s var(--ease);
}
.term-card.above {
  transform: translateY(-100%);
}
.dark .term-card {
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.55);
}
@keyframes term-in {
  from {
    opacity: 0;
  }
}
.term-card p {
  margin: 0;
}
.term-name {
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.term-name b {
  font-size: 15px;
}
.term-name span {
  color: var(--vp-c-text-2);
}
.term-full {
  font-size: 12.5px;
  color: var(--vp-c-text-3);
}
.term-plain {
  margin-top: 8px !important;
  color: var(--vp-c-text-1);
}
.term-plain-label {
  margin-right: 8px;
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.06em;
  color: var(--box-human-ink);
}
.term-more {
  display: inline-block;
  margin-top: 8px;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
</style>
