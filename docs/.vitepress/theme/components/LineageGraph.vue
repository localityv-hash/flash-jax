<script setup lang="ts">
/**
 * 谱系图：时间自上而下，泳道为列。悬停 / 点击节点时高亮它的祖先与后代，下方面板给出说明与跳转。
 * 数据来自 docs/.vitepress/data/lineage/<id>.yaml。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import type { LineageNode } from '../../data/types'
import { month } from '../lib/format'

const props = defineProps<{ graph: string }>()
const g = computed(() => data.lineages[props.graph])

const NODE_H = 46
const ROW_H = 64
const PAD_X = 50
const PAD_TOP = 54
const PAD_BOTTOM = 18
const MIN_LANE_W = 118
const MAX_LANE_W = 190

// 泳道宽度随容器自适应：放得下就不出现横向滚动条，太窄（手机）时才滚动。
const scroller = ref<HTMLElement | null>(null)
const containerW = ref(720)
let observer: ResizeObserver | undefined
const overflowRight = ref(false)
const overflowing = ref(false)
function updateOverflow(): void {
  const el = scroller.value
  if (!el) return
  overflowing.value = el.scrollWidth > el.clientWidth + 4
  overflowRight.value = el.scrollLeft + el.clientWidth < el.scrollWidth - 4
}
onMounted(() => {
  if (!scroller.value) return
  containerW.value = scroller.value.clientWidth
  observer = new ResizeObserver(() => {
    containerW.value = scroller.value?.clientWidth ?? containerW.value
    requestAnimationFrame(updateOverflow)
  })
  observer.observe(scroller.value)
  requestAnimationFrame(updateOverflow)
})
onBeforeUnmount(() => observer?.disconnect())

const laneW = computed(() => {
  const n = g.value?.lanes.length ?? 1
  return Math.max(MIN_LANE_W, Math.min(MAX_LANE_W, Math.floor((containerW.value - PAD_X - 12) / n)))
})
const nodeW = computed(() => laneW.value - 20)

interface Placed extends LineageNode {
  x: number
  y: number
  row: number
}

const layout = computed(() => {
  const graph = g.value
  if (!graph) return null
  const LANE_W = laneW.value
  const laneIndex = new Map(graph.lanes.map((l, i) => [l.id, i]))
  const nodes = [...graph.nodes].sort((a, b) => String(a.date).localeCompare(String(b.date)))
  // 按年份分带：同一年里，每条泳道的节点按时间自上而下堆叠；年份带的高度取各泳道堆叠数的最大值。
  const years = [...new Set(nodes.map((n) => String(n.date).slice(0, 4)))]
  const placed: Placed[] = []
  const yearMarks: { y: number; label: string }[] = []
  let row = 0
  for (const yr of years) {
    const inYear = nodes.filter((n) => String(n.date).startsWith(yr))
    const depth = new Map<number, number>()
    yearMarks.push({ y: PAD_TOP + row * ROW_H - 9, label: yr })
    let maxDepth = 0
    for (const n of inYear) {
      const lane = laneIndex.get(n.lane) ?? 0
      const d = depth.get(lane) ?? 0
      depth.set(lane, d + 1)
      maxDepth = Math.max(maxDepth, d + 1)
      const r = row + d
      placed.push({ ...n, row: r, x: PAD_X + lane * LANE_W + LANE_W / 2, y: PAD_TOP + r * ROW_H + NODE_H / 2 })
    }
    row += maxDepth
  }
  const byId = new Map(placed.map((p) => [p.id, p]))
  const edges = graph.edges
    .map((e) => ({ ...e, a: byId.get(e.from)!, b: byId.get(e.to)! }))
    .filter((e) => e.a && e.b)
  const width = PAD_X + graph.lanes.length * LANE_W + 12
  const height = PAD_TOP + row * ROW_H - (ROW_H - NODE_H) + PAD_BOTTOM
  return { placed, byId, edges, years: yearMarks, width, height, LANE_W }
})

const hovered = ref<string | null>(null)
const selected = ref<string | null>(null)
const focus = computed(() => hovered.value ?? selected.value)

/** 焦点节点的祖先与后代。 */
const related = computed(() => {
  const L = layout.value
  const id = focus.value
  if (!L || !id) return null
  const up = new Map<string, string[]>()
  const down = new Map<string, string[]>()
  for (const e of L.edges) {
    down.set(e.from, [...(down.get(e.from) ?? []), e.to])
    up.set(e.to, [...(up.get(e.to) ?? []), e.from])
  }
  const seen = new Set([id])
  const walk = (m: Map<string, string[]>, start: string) => {
    const stack = [start]
    while (stack.length) for (const n of m.get(stack.pop()!) ?? []) if (!seen.has(n)) (seen.add(n), stack.push(n))
  }
  walk(up, id)
  walk(down, id)
  return seen
})

const current = computed(() => (selected.value ? layout.value?.byId.get(selected.value) : null) ?? (hovered.value ? layout.value?.byId.get(hovered.value) : null))

function edgePath(a: Placed, b: Placed): string {
  const W = nodeW.value
  if (b.row === a.row) {
    // 同一行：从侧边连过去
    const dir = b.x > a.x ? 1 : -1
    const x1 = a.x + (dir * W) / 2
    const x2 = b.x - (dir * W) / 2 - dir * 4
    return `M ${x1} ${a.y} C ${(x1 + x2) / 2} ${a.y}, ${(x1 + x2) / 2} ${b.y}, ${x2} ${b.y}`
  }
  const down = b.row > a.row
  const x1 = a.x
  const y1 = a.y + (down ? NODE_H / 2 : -NODE_H / 2)
  const x2 = b.x
  const y2 = b.y + (down ? -NODE_H / 2 - 4 : NODE_H / 2 + 4)
  const dy = Math.max(18, Math.abs(y2 - y1) * 0.45) * (down ? 1 : -1)
  return `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`
}

/** 按显示宽度截断：中日韩字符按 2 个单位计。 */
function fit(text: string, px: number, unit = 7.2): string {
  const max = Math.floor(px / unit)
  let units = 0
  let out = ''
  for (const ch of text) {
    units += /[\u2e80-\uffff]/.test(ch) ? 2 : 1
    if (units > max) return `${out}…`
    out += ch
  }
  return out
}

function labelOf(n: LineageNode): string {
  return fit(n.label, nodeW.value - 14)
}

function hrefOf(n: LineageNode): string | null {
  if (n.href) return n.href.startsWith('#') ? n.href : withBase(n.href)
  if (n.entry) return withBase(`/library/?id=${n.entry}`)
  return null
}

function select(id: string): void {
  selected.value = selected.value === id ? null : id
}
</script>

<template>
  <figure v-if="g && layout" class="lineage atlas-keep-return">
    <div class="lineage-head">
      <p class="lineage-title">{{ g.title }}</p>
      <p v-if="g.description" class="lineage-desc">{{ g.description }}</p>
      <p v-if="overflowing" class="lineage-hint">← 左右滑动查看全图 →</p>
    </div>
    <div class="lineage-viewport" :class="{ 'fade-right': overflowRight }">
    <div ref="scroller" class="lineage-scroll" @scroll.passive="updateOverflow">
      <svg :viewBox="`0 0 ${layout.width} ${layout.height}`" :width="layout.width" :height="layout.height" role="img" :aria-label="g.title" class="lineage-svg" @mouseleave="hovered = null">
        <defs>
          <marker :id="`arrow-${g.id}`" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 9 5 L 0 9 z" class="lineage-arrow" />
          </marker>
          <marker :id="`arrow-on-${g.id}`" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 9 5 L 0 9 z" class="lineage-arrow on" />
          </marker>
        </defs>

        <g v-for="(lane, i) in g.lanes" :key="lane.id">
          <rect :x="PAD_X + i * layout.LANE_W + 5" y="8" :width="layout.LANE_W - 10" :height="layout.height - 16" rx="14" class="lineage-lane" />
          <text :x="PAD_X + i * layout.LANE_W + layout.LANE_W / 2" y="31" text-anchor="middle" class="lineage-lane-label">{{ fit(lane.label, layout.LANE_W - 14) }}<title>{{ lane.label }}</title></text>
        </g>

        <g v-for="yr in layout.years" :key="yr.label">
          <line :x1="10" :x2="layout.width - 10" :y1="yr.y" :y2="yr.y" class="lineage-year-line" />
          <text :x="14" :y="yr.y + 16" class="lineage-year">{{ yr.label }}</text>
        </g>

        <path
          v-for="e in layout.edges"
          :key="`${e.from}-${e.to}`"
          :d="edgePath(e.a, e.b)"
          class="lineage-edge"
          :class="{ on: related && related.has(e.from) && related.has(e.to), dim: related && !(related.has(e.from) && related.has(e.to)) }"
          :marker-end="`url(#${related && related.has(e.from) && related.has(e.to) ? 'arrow-on-' : 'arrow-'}${g.id})`"
        >
          <title>{{ e.label ?? '' }}</title>
        </path>

        <g
          v-for="n in layout.placed"
          :key="n.id"
          class="lineage-node"
          :class="{ dim: related && !related.has(n.id), focus: focus === n.id }"
          tabindex="0"
          role="button"
          :aria-label="`${n.label}，${month(n.date)}`"
          @mouseenter="hovered = n.id"
          @focus="hovered = n.id"
          @blur="hovered = null"
          @click="select(n.id)"
          @keydown.enter="select(n.id)"
        >
          <rect :x="n.x - nodeW / 2" :y="n.y - NODE_H / 2" :width="nodeW" :height="NODE_H" rx="11" />
          <text :x="n.x" :y="n.y - 3" text-anchor="middle" class="lineage-node-label">{{ labelOf(n) }}</text>
          <text :x="n.x" :y="n.y + 14" text-anchor="middle" class="lineage-node-date">{{ month(n.date) }}</text>
        </g>
      </svg>
    </div>
    </div>
    <figcaption class="lineage-panel">
      <template v-if="current">
        <p class="lineage-panel-title">
          <b>{{ current.label }}</b>
          <span>{{ month(current.date) }}</span>
          <a v-if="hrefOf(current)" :href="hrefOf(current)!" class="lineage-go">{{ current.href ? '跳到正文' : '查看条目' }} →</a>
          <a v-if="current.entry && current.href" :href="withBase(`/library/?id=${current.entry}`)" class="lineage-go">资料库 →</a>
        </p>
        <p v-if="current.note" class="lineage-panel-note">{{ current.note }}</p>
      </template>
      <p v-else class="lineage-panel-hint">悬停或点击节点：高亮它的前因后果，并在这里给出一句话说明与跳转。</p>
    </figcaption>
  </figure>
  <div v-else class="lineage-missing">未找到谱系图：{{ graph }}</div>
</template>

<style scoped>
.lineage {
  margin: 28px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--radius-lg);
  background: var(--vp-c-bg);
  overflow: hidden;
}
.lineage-head {
  padding: 16px 20px 4px;
}
.lineage-title {
  margin: 0 !important;
  font-weight: 700;
  font-size: 15px;
}
.lineage-desc {
  margin: 4px 0 0 !important;
  font-size: 13.5px !important;
  color: var(--vp-c-text-2);
  line-height: 1.7 !important;
}
.lineage-viewport {
  position: relative;
}
.lineage-viewport.fade-right::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 44px;
  pointer-events: none;
  background: linear-gradient(90deg, transparent, var(--vp-c-bg));
}
.lineage-scroll {
  overflow-x: auto;
  padding: 0 4px;
}
.lineage-hint {
  margin: 6px 0 0 !important;
  font-size: 12px !important;
  color: var(--vp-c-text-3);
}
.lineage-svg {
  display: block;
  margin: 0 auto;
  font-family: var(--vp-font-family-base);
}
.lineage-lane {
  fill: var(--vp-c-bg-alt);
  opacity: 0.7;
}
.lineage-lane-label {
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.03em;
  fill: var(--vp-c-text-3);
}
.lineage-year-line {
  stroke: var(--vp-c-divider);
  stroke-dasharray: 3 5;
}
.lineage-year {
  font-size: 12px;
  font-weight: 700;
  fill: var(--vp-c-text-3);
  font-variant-numeric: tabular-nums;
}
.lineage-edge {
  fill: none;
  stroke: var(--vp-c-border);
  stroke-width: 1.4;
  opacity: 0.85;
  transition: stroke 0.2s, opacity 0.2s;
}
.lineage-edge.on {
  stroke: var(--vp-c-brand-2);
  stroke-width: 2;
  opacity: 1;
}
.lineage-edge.dim {
  opacity: 0.25;
}
.lineage-arrow {
  fill: var(--vp-c-border);
}
.lineage-arrow.on {
  fill: var(--vp-c-brand-2);
}
.lineage-node {
  cursor: pointer;
  outline: none;
  transition: opacity 0.2s;
}
.lineage-node rect {
  fill: var(--card-bg);
  stroke: var(--vp-c-border);
  stroke-width: 1.2;
  transition: stroke 0.2s, fill 0.2s;
}
.lineage-node:hover rect,
.lineage-node.focus rect,
.lineage-node:focus-visible rect {
  stroke: var(--vp-c-brand-2);
  stroke-width: 1.8;
  fill: color-mix(in srgb, var(--vp-c-brand-soft) 60%, var(--card-bg));
}
.lineage-node.dim {
  opacity: 0.28;
}
.lineage-node-label {
  font-size: 13px;
  font-weight: 650;
  fill: var(--vp-c-text-1);
}
.lineage-node-date {
  font-size: 11px;
  fill: var(--vp-c-text-3);
  font-variant-numeric: tabular-nums;
}
.lineage-panel {
  min-height: 64px;
  padding: 12px 20px 14px;
  border-top: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
}
.lineage-panel-title {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 12px;
  margin: 0 !important;
  font-size: 14px !important;
}
.lineage-panel-title span {
  font-size: 12.5px;
  color: var(--vp-c-text-3);
}
.lineage-go {
  font-size: 13px;
  text-decoration: none !important;
}
.lineage-panel-note {
  margin: 6px 0 0 !important;
  font-size: 14px !important;
  line-height: 1.75 !important;
  color: var(--vp-c-text-2);
}
.lineage-panel-hint {
  margin: 8px 0 0 !important;
  font-size: 13px !important;
  color: var(--vp-c-text-3);
}
.lineage-missing {
  color: var(--box-pitfall-ink);
  font-size: 13px;
}
</style>
