<script setup lang="ts">
/**
 * 阅读辅助：
 * 1. 顶部阅读进度条；
 * 2. 「返回阅读点」：在正文、目录、侧边栏里点链接跳走前记下当前位置，右下角一键跳回（可连跳多级）；
 * 3. 「继续阅读」：记住每页读到哪，再次打开时提示续读。
 * 状态只存在浏览器本地（session/localStorage），读写都容错。
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData, useRoute, useRouter } from 'vitepress'

interface Mark {
  path: string
  y: number
  label: string
}

const STACK_KEY = 'atlas:return-stack'
const POS_KEY = 'atlas:reading-positions'
const MAX_STACK = 12

const route = useRoute()
const router = useRouter()
const { frontmatter, page } = useData()

const progress = ref(0)
const scrollY = ref(0)
const stack = ref<Mark[]>([])
const resume = ref<Mark | null>(null)
const scrolled = ref(false)

const isDoc = computed(() => !page.value.isNotFound && (frontmatter.value.layout ?? 'doc') === 'doc')
const top = computed(() => stack.value[stack.value.length - 1] ?? null)

function load<T>(storage: Storage | undefined, key: string, fallback: T): T {
  try {
    const raw = storage?.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}
function save(storage: Storage | undefined, key: string, value: unknown): void {
  try {
    storage?.setItem(key, JSON.stringify(value))
  } catch {
    /* 隐私模式等情况下存储不可用，功能静默降级 */
  }
}

function clean(text: string | null | undefined): string {
  return (text ?? '').replace(/[​#]/g, '').replace(/\s+/g, ' ').trim()
}

/** 当前视口顶部所在的小节标题。 */
function currentHeading(): string {
  let label = clean(document.querySelector('.vp-doc h1')?.textContent) || document.title
  for (const h of document.querySelectorAll<HTMLElement>('.vp-doc h2, .vp-doc h3')) {
    if (h.getBoundingClientRect().top > 140) break
    label = clean(h.textContent)
  }
  return label
}

function normalize(path: string): string {
  return decodeURI(path).replace(/\.html$/, '').replace(/\/index$/, '/')
}

function onScroll(): void {
  scrollY.value = window.scrollY
  scrolled.value = window.scrollY > 600
  const doc = document.querySelector<HTMLElement>('.vp-doc')
  if (!doc) {
    progress.value = 0
    return
  }
  const start = doc.getBoundingClientRect().top + window.scrollY - 80
  const end = start + doc.offsetHeight - window.innerHeight + 160
  progress.value = Math.min(1, Math.max(0, (window.scrollY - start) / Math.max(1, end - start)))
  if (resume.value && window.scrollY > 400) resume.value = null
  rememberPosition()
}

let saveTimer: number | undefined
function rememberPosition(): void {
  window.clearTimeout(saveTimer)
  saveTimer = window.setTimeout(() => {
    if (!isDoc.value) return
    const all = load<Record<string, Mark & { t: number }>>(globalThis.localStorage, POS_KEY, {})
    all[normalize(route.path)] = { path: route.path, y: Math.round(window.scrollY), label: currentHeading(), t: Date.now() }
    // 只保留最近 60 页
    const keys = Object.keys(all).sort((a, b) => all[b].t - all[a].t)
    for (const k of keys.slice(60)) delete all[k]
    save(globalThis.localStorage, POS_KEY, all)
  }, 400)
}

function isInternal(a: HTMLAnchorElement): boolean {
  if (a.target === '_blank' || a.hasAttribute('download')) return false
  const href = a.getAttribute('href') ?? ''
  if (href.startsWith('#')) return true
  try {
    const url = new URL(a.href, location.href)
    return url.origin === location.origin
  } catch {
    return false
  }
}

/** 在正文、页内目录、侧边栏、组件里点击站内链接时记录返回点。 */
function onClick(event: MouseEvent): void {
  // 注意：VitePress 路由会在 window 捕获阶段对站内链接 preventDefault，所以这里不能以 defaultPrevented 为条件。
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  const a = (event.target as HTMLElement | null)?.closest?.('a') as HTMLAnchorElement | null
  if (!a || !isInternal(a) || a.closest('.atlas-return, .VPNavBar, .VPLocalNav, .atlas-resume')) return
  if (!a.closest('.vp-doc, .VPDocAside, .VPSidebar, .atlas-keep-return')) return
  if (!isDoc.value || window.scrollY < 240) return
  const mark: Mark = { path: route.path, y: Math.round(window.scrollY), label: currentHeading() }
  const last = stack.value[stack.value.length - 1]
  if (last && last.path === mark.path && Math.abs(last.y - mark.y) < 120) return
  stack.value = [...stack.value, mark].slice(-MAX_STACK)
  save(globalThis.sessionStorage, STACK_KEY, stack.value)
}

async function scrollToY(y: number): Promise<void> {
  await nextTick()
  requestAnimationFrame(() => window.scrollTo({ top: y, behavior: 'smooth' }))
}

async function goBack(): Promise<void> {
  const mark = stack.value[stack.value.length - 1]
  if (!mark) return
  stack.value = stack.value.slice(0, -1)
  save(globalThis.sessionStorage, STACK_KEY, stack.value)
  if (normalize(mark.path) !== normalize(route.path)) {
    await router.go(mark.path)
    window.setTimeout(() => scrollToY(mark.y), 60)
  } else {
    scrollToY(mark.y)
  }
}

function dismissStack(): void {
  stack.value = []
  save(globalThis.sessionStorage, STACK_KEY, [])
}

function checkResume(): void {
  resume.value = null
  if (!isDoc.value || location.hash) return
  const saved = load<Record<string, Mark & { t: number }>>(globalThis.localStorage, POS_KEY, {})[normalize(route.path)]
  const fresh = saved && Date.now() - saved.t < 1000 * 60 * 60 * 24 * 45
  if (fresh && saved.y > 900 && window.scrollY < 200) resume.value = saved
}

function doResume(): void {
  if (resume.value) scrollToY(resume.value.y)
  resume.value = null
}

function toTop(): void {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

/** 返回点就在眼前（同页且相距不到半屏）时不显示按钮。 */
const showReturn = computed(() => {
  const m = top.value
  if (!m) return false
  const samePage = normalize(m.path) === normalize(route.path)
  return !samePage || Math.abs(scrollY.value - m.y) > 360
})

onMounted(() => {
  stack.value = load<Mark[]>(globalThis.sessionStorage, STACK_KEY, [])
  window.addEventListener('scroll', onScroll, { passive: true })
  document.addEventListener('click', onClick, true)
  onScroll()
  window.setTimeout(checkResume, 300)
})
onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
  document.removeEventListener('click', onClick, true)
})
watch(
  () => route.path,
  () => {
    window.setTimeout(() => {
      onScroll()
      checkResume()
    }, 300)
  },
)
</script>

<template>
  <div class="atlas-progress" :class="{ hidden: !isDoc }" :style="{ transform: `scaleX(${progress})` }" aria-hidden="true" />

  <Transition name="atlas-fade">
    <div v-if="resume" class="atlas-resume" role="status">
      <span class="atlas-resume-text">上次读到 <b>{{ resume.label }}</b></span>
      <button class="atlas-resume-go" type="button" @click="doResume">继续阅读</button>
      <button class="atlas-icon-btn" type="button" aria-label="关闭" @click="resume = null">×</button>
    </div>
  </Transition>

  <div class="atlas-float">
    <Transition name="atlas-fade">
      <div v-if="showReturn && top" class="atlas-return">
        <button class="atlas-return-btn" type="button" :title="`返回：${top.label}`" @click="goBack">
          <svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true">
            <path d="M8 4 3 9l5 5M3.5 9H12a5 5 0 0 1 0 10h-1.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <span class="atlas-return-text">
            <span class="atlas-return-kicker">返回阅读点{{ stack.length > 1 ? ` · ${stack.length}` : '' }}</span>
            <span class="atlas-return-label">{{ top.label }}</span>
          </span>
        </button>
        <button class="atlas-icon-btn" type="button" aria-label="清除返回点" title="清除返回点" @click="dismissStack">×</button>
      </div>
    </Transition>
    <Transition name="atlas-fade">
      <button v-if="scrolled && isDoc" class="atlas-top-btn" type="button" aria-label="回到顶部" title="回到顶部" @click="toTop">
        <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
          <path d="M10 16V4M4.5 9.5 10 4l5.5 5.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>
    </Transition>
  </div>
</template>

<style scoped>
.atlas-progress {
  position: fixed;
  inset: 0 0 auto 0;
  height: 2.5px;
  z-index: 100;
  transform-origin: 0 50%;
  background: linear-gradient(90deg, var(--vp-c-brand-3), var(--vp-c-brand-1));
  transition: transform 0.12s linear, opacity 0.3s;
  pointer-events: none;
}
.atlas-progress.hidden {
  opacity: 0;
}

.atlas-float {
  position: fixed;
  right: max(20px, env(safe-area-inset-right));
  bottom: max(22px, env(safe-area-inset-bottom));
  z-index: 60;
  display: flex;
  align-items: flex-end;
  gap: 10px;
}

.atlas-return {
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 4px 4px 4px 4px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--vp-c-bg-elv) 92%, transparent);
  border: 1px solid var(--vp-c-divider);
  box-shadow: var(--card-shadow-hover);
  backdrop-filter: blur(10px);
}
.atlas-return-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: min(320px, 70vw);
  padding: 6px 10px;
  border-radius: 10px;
  color: var(--vp-c-text-1);
  text-align: left;
  transition: background 0.2s;
}
.atlas-return-btn:hover {
  background: var(--vp-c-bg-soft);
}
.atlas-return-btn svg {
  flex: none;
  color: var(--vp-c-brand-1);
}
.atlas-return-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.atlas-return-kicker {
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--vp-c-text-3);
}
.atlas-return-label {
  font-size: 13.5px;
  font-weight: 600;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.atlas-icon-btn {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  color: var(--vp-c-text-3);
  font-size: 17px;
  line-height: 1;
  transition: background 0.2s, color 0.2s;
}
.atlas-icon-btn:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.atlas-top-btn {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 13px;
  color: var(--vp-c-text-2);
  background: color-mix(in srgb, var(--vp-c-bg-elv) 92%, transparent);
  border: 1px solid var(--vp-c-divider);
  box-shadow: var(--card-shadow);
  backdrop-filter: blur(10px);
  transition: color 0.2s, box-shadow 0.2s;
}
.atlas-top-btn:hover {
  color: var(--vp-c-brand-1);
  box-shadow: var(--card-shadow-hover);
}

.atlas-resume {
  position: fixed;
  left: 50%;
  top: calc(var(--vp-nav-height) + 14px);
  transform: translateX(-50%);
  z-index: 60;
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: min(560px, calc(100vw - 32px));
  padding: 7px 8px 7px 16px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--vp-c-bg-elv) 94%, transparent);
  border: 1px solid var(--vp-c-divider);
  box-shadow: var(--card-shadow-hover);
  backdrop-filter: blur(10px);
  font-size: 13.5px;
}
.atlas-resume-text {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--vp-c-text-2);
}
.atlas-resume-text b {
  color: var(--vp-c-text-1);
  font-weight: 600;
}
.atlas-resume-go {
  flex: none;
  padding: 5px 12px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 13px;
  color: #fff;
  background: var(--vp-c-brand-1);
}
.dark .atlas-resume-go {
  color: #0d0f12;
}

.atlas-fade-enter-active,
.atlas-fade-leave-active {
  transition: opacity 0.25s var(--ease), transform 0.25s var(--ease);
}
.atlas-fade-enter-from,
.atlas-fade-leave-to {
  opacity: 0;
  transform: translateY(6px);
}
.atlas-resume.atlas-fade-enter-from,
.atlas-resume.atlas-fade-leave-to {
  transform: translate(-50%, -6px);
}

@media (max-width: 640px) {
  .atlas-return-label {
    max-width: 46vw;
  }
}
@media print {
  .atlas-progress,
  .atlas-float,
  .atlas-resume {
    display: none;
  }
}
</style>
