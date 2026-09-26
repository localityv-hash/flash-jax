<script setup lang="ts">
/**
 * 谱系图：时间自上而下，泳道为列。悬停 / 点击节点时高亮它的祖先与后代，下方面板给出说明与跳转。
 * 数据来自 docs/.vitepress/data/lineage/<id>.yaml。
 */
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import type { LineageNode } from '../../data/types'
import { month } from '../lib/format'

const props = defineProps<{ graph: string }>()
const g = computed(() => data.lineages[props.graph])

const LANE_W = 168
const NODE_W = 138
const NODE_H = 46
const ROW_H = 74
const PAD_X = 64
const PAD_TOP = 58
const PAD_BOTTOM = 24

interface Placed extends LineageNode {
  x: number
  y: number
  row: number
}

const layout = computed(() => {
  const graph = g.value
  if (!graph) return null
  const laneIndex = new Map(graph.lanes.map((l, i) => [l.id, i]))
  const nodes = [...graph.nodes].sort((a, b) => String(a.date).localeCompare(String(b.date)))
  // 同一时间点的节点尽量同一行；同泳道冲突时另起一行。
  const placed: Placed[] = []
  let row = -1
  let rowDate = ''
  let used = new Set<number>()
  for (const n of nodes) {
    const lane = laneIndex.get(n.lane) ?? 0
    const d = String(n.date).slice(0, 7)
    if (d !== rowDate || used.has(lane)) {
      row += 1
      rowDate = d
      used = new Set()
    }
    used.add(lane)
    placed.push({ ...n, row, x: PAD_X + lane * LANE_W + LANE_W / 2, y: PAD_TOP + row * ROW_H + NODE_H / 2 })
  }
  const byId = new Map(placed.map((p) => [p.id, p]))
  const edges = graph.edges
    .map((e) => ({ ...e, a: byId.get(e.from)!, b: byId.get(e.to)! }))
    .filter((e) => e.a && e.b)
  // 年份刻度：每当年份变化，在左侧留一个标记
  const years: { y: number; label: string }[] = []
  let lastYear = ''
  for (const p of placed) {
    const yr = String(p.date).slice(0, 4)
    if (yr !== lastYear) {
      years.push({ y: p.y - NODE_H / 2 - 12, label: yr })
      lastYear = yr
    }
  }
  const width = PAD_X + graph.lanes.length * LANE_W + 16
  const height = PAD_TOP + (row + 1) * ROW_H + PAD_BOTTOM - (ROW_H - NODE_H)
  return { placed, byId, edges, years, width, height }
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
  const x1 = a.x
  const y1 = a.y + NODE_H / 2
  const x2 = b.x
  const y2 = b.y - NODE_H / 2 - 4
  const dy = Math.max(24, (y2 - y1) * 0.5)
  return `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`
}

function labelOf(n: LineageNode): string {
  return n.label.length > 16 ? `${n.label.slice(0, 15)}…` : n.label
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
    </div>
    <div class="lineage-scroll">
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
          <rect :x="PAD_X + i * LANE_W + 6" y="8" :width="LANE_W - 12" :height="layout.height - 16" rx="14" class="lineage-lane" />
          <text :x="PAD_X + i * LANE_W + LANE_W / 2" y="32" text-anchor="middle" class="lineage-lane-label">{{ lane.label }}</text>
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
          <rect :x="n.x - NODE_W / 2" :y="n.y - NODE_H / 2" :width="NODE_W" :height="NODE_H" rx="11" />
          <text :x="n.x" :y="n.y - 3" text-anchor="middle" class="lineage-node-label">{{ labelOf(n) }}</text>
          <text :x="n.x" :y="n.y + 14" text-anchor="middle" class="lineage-node-date">{{ month(n.date) }}</text>
        </g>
      </svg>
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
.lineage-scroll {
  overflow-x: auto;
  padding: 0 4px;
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
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
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
  stroke-width: 1.5;
  transition: stroke 0.2s, opacity 0.2s;
}
.lineage-edge.on {
  stroke: var(--vp-c-brand-2);
  stroke-width: 2;
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
