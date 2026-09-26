<script setup lang="ts">
/** 浏览器端渲染 Mermaid，跟随明暗主题重绘；点击「放大」全屏查看大图。 */
import { onMounted, ref, useId, watch } from 'vue'
import { useData } from 'vitepress'

const props = defineProps<{ code: string; caption?: string }>()
const { isDark } = useData()
const svg = ref('')
const error = ref('')
const zoomed = ref(false)
const uid = (useId() ?? 'm').replace(/[^a-zA-Z0-9]/g, '')
let seq = 0

function themeVariables(dark: boolean): Record<string, string> {
  return dark
    ? {
        background: 'transparent',
        primaryColor: '#15272a',
        primaryBorderColor: '#2f6f69',
        primaryTextColor: '#e9e8e4',
        secondaryColor: '#221c33',
        secondaryBorderColor: '#5b4b91',
        tertiaryColor: '#2a1f14',
        tertiaryBorderColor: '#8a5a24',
        lineColor: '#7c8088',
        textColor: '#d6d6d2',
        mainBkg: '#15272a',
        nodeBorder: '#2f6f69',
        clusterBkg: 'rgba(255,255,255,0.03)',
        clusterBorder: '#2d323a',
        edgeLabelBackground: '#0d0f12',
        actorBkg: '#15272a',
        actorBorder: '#2f6f69',
        actorTextColor: '#e9e8e4',
        signalColor: '#a8aab0',
        signalTextColor: '#d6d6d2',
        noteBkgColor: '#2a2415',
        noteBorderColor: '#6b5520',
        noteTextColor: '#e9e8e4',
        labelBoxBkgColor: '#15272a',
        labelBoxBorderColor: '#2f6f69',
        fontSize: '14px',
      }
    : {
        background: 'transparent',
        primaryColor: '#eef8f6',
        primaryBorderColor: '#8fcac1',
        primaryTextColor: '#1a1b1e',
        secondaryColor: '#f4f1ff',
        secondaryBorderColor: '#c4b5fd',
        tertiaryColor: '#fff6ea',
        tertiaryBorderColor: '#f1c68a',
        lineColor: '#8c8e94',
        textColor: '#33353a',
        mainBkg: '#eef8f6',
        nodeBorder: '#8fcac1',
        clusterBkg: '#faf9f6',
        clusterBorder: '#e2e0d8',
        edgeLabelBackground: '#fcfcfa',
        actorBkg: '#eef8f6',
        actorBorder: '#8fcac1',
        actorTextColor: '#1a1b1e',
        signalColor: '#55575d',
        signalTextColor: '#33353a',
        noteBkgColor: '#fff8ec',
        noteBorderColor: '#f1d9ab',
        noteTextColor: '#1a1b1e',
        labelBoxBkgColor: '#eef8f6',
        labelBoxBorderColor: '#8fcac1',
        fontSize: '14px',
      }
}

async function render(): Promise<void> {
  try {
    const mermaid = (await import('mermaid')).default
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'base',
      themeVariables: themeVariables(isDark.value),
      fontFamily: getComputedStyle(document.body).fontFamily,
      flowchart: { curve: 'basis', padding: 14, nodeSpacing: 36, rankSpacing: 46, htmlLabels: true },
      sequence: { mirrorActors: false, messageAlign: 'center', actorMargin: 60 },
    })
    const out = await mermaid.render(`mmd-${uid}-${seq++}`, decodeURIComponent(props.code))
    svg.value = out.svg
    error.value = ''
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

onMounted(render)
watch(isDark, render)
</script>

<template>
  <figure class="mermaid-figure">
    <div v-if="error" class="mermaid-error">图表渲染失败：{{ error }}</div>
    <div v-else-if="!svg" class="mermaid-loading" aria-hidden="true" />
    <div v-else class="mermaid-canvas" v-html="svg" />
    <figcaption v-if="caption || svg">
      <span>{{ caption }}</span>
      <button v-if="svg" type="button" class="mermaid-zoom" @click="zoomed = true">放大</button>
    </figcaption>
    <Teleport to="body">
      <div v-if="zoomed" class="mermaid-overlay" role="dialog" aria-modal="true" @click.self="zoomed = false" @keydown.esc="zoomed = false">
        <div class="mermaid-overlay-inner">
          <button type="button" class="mermaid-close" aria-label="关闭" @click="zoomed = false">×</button>
          <div class="mermaid-overlay-canvas" v-html="svg" />
          <p v-if="caption" class="mermaid-overlay-caption">{{ caption }}</p>
        </div>
      </div>
    </Teleport>
  </figure>
</template>

<style scoped>
.mermaid-figure {
  margin: 26px 0;
  padding: 18px 16px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--radius-lg);
  background: var(--vp-c-bg);
}
.mermaid-canvas {
  display: flex;
  justify-content: center;
  overflow-x: auto;
}
.mermaid-canvas :deep(svg) {
  max-width: 100%;
  height: auto;
}
.mermaid-loading {
  height: 160px;
  border-radius: var(--radius-md);
  background: linear-gradient(90deg, var(--vp-c-bg-soft), var(--vp-c-bg-alt), var(--vp-c-bg-soft));
  background-size: 200% 100%;
  animation: shimmer 1.4s linear infinite;
}
@keyframes shimmer {
  to {
    background-position: -200% 0;
  }
}
.mermaid-error {
  font-size: 13px;
  color: var(--box-pitfall-ink);
}
figcaption {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.mermaid-zoom {
  flex: none;
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
}
.mermaid-zoom:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-2);
}
.mermaid-overlay {
  position: fixed;
  inset: 0;
  z-index: 200;
  display: grid;
  place-items: center;
  padding: 24px;
  background: rgba(10, 12, 14, 0.55);
  backdrop-filter: blur(4px);
}
.mermaid-overlay-inner {
  position: relative;
  max-width: 96vw;
  max-height: 92vh;
  overflow: auto;
  padding: 36px 28px 20px;
  border-radius: var(--radius-lg);
  background: var(--vp-c-bg);
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
}
.mermaid-overlay-canvas :deep(svg) {
  width: min(1400px, 90vw) !important;
  max-width: none !important;
  height: auto;
}
.mermaid-overlay-caption {
  margin-top: 10px;
  text-align: center;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.mermaid-close {
  position: absolute;
  top: 8px;
  right: 10px;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  font-size: 20px;
  color: var(--vp-c-text-2);
}
.mermaid-close:hover {
  background: var(--vp-c-bg-soft);
}
</style>
