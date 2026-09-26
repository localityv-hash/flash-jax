<script setup lang="ts">
/** 资料库：按领域 / 视角 / 类型 / 分级筛选，支持搜索、排序，筛选状态同步到 URL，便于分享与跳回。 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { data } from '../../data/atlas.data'
import type { Area, Entry, Evidence, Facet, Kind, Tier } from '../../data/types'
import { AREA_ORDER, AREAS, EVIDENCE, FACET_ORDER, FACETS, KINDS, TIER_ORDER, TIERS } from '../lib/labels'
import { impact } from '../lib/format'
import EntryCard from './EntryCard.vue'

type SortKey = 'tier' | 'date' | 'impact'

const q = ref('')
const areas = ref<Area[]>([])
const facets = ref<Facet[]>([])
const kinds = ref<Kind[]>([])
const tiers = ref<Tier[]>([])
const evidence = ref<Evidence[]>([])
const sort = ref<SortKey>('tier')
const focusId = ref<string | null>(null)

const all = data.entries
const kindOrder = computed(() => (Object.keys(KINDS) as Kind[]).filter((k) => all.some((e) => e.kind === k)))

function matches(e: Entry, needle: string): boolean {
  if (!needle) return true
  const hay = [e.id, e.title, e.short, e.org, e.authors, e.summary, e.plain, ...(e.tags ?? [])]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
  return needle
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((w) => hay.includes(w))
}

function pass(e: Entry, skip?: string): boolean {
  return (
    (skip === 'area' || !areas.value.length || e.areas.some((a) => areas.value.includes(a))) &&
    (skip === 'facet' || !facets.value.length || e.facets.some((f) => facets.value.includes(f))) &&
    (skip === 'kind' || !kinds.value.length || kinds.value.includes(e.kind)) &&
    (skip === 'tier' || !tiers.value.length || tiers.value.includes(e.tier)) &&
    (skip === 'evidence' || !evidence.value.length || evidence.value.every((v) => e.evidence.includes(v))) &&
    matches(e, q.value.trim())
  )
}

const results = computed(() => {
  const list = all.filter((e) => pass(e))
  const byDate = (a: Entry, b: Entry) => b.date.localeCompare(a.date)
  if (sort.value === 'date') return [...list].sort(byDate)
  if (sort.value === 'impact') return [...list].sort((a, b) => impact(b) - impact(a) || byDate(a, b))
  return [...list].sort((a, b) => TIERS[a.tier].rank - TIERS[b.tier].rank || byDate(a, b))
})

/** 每个选项在「其余筛选条件不变」时的命中数，帮助判断点下去会剩多少。 */
function count(group: string, pred: (e: Entry) => boolean): number {
  return all.filter((e) => pass(e, group) && pred(e)).length
}

const groups = { area: areas, facet: facets, kind: kinds, tier: tiers, evidence } as const
type Group = keyof typeof groups

function toggle(group: Group, v: string): void {
  const list = groups[group] as unknown as { value: string[] }
  list.value = list.value.includes(v) ? list.value.filter((x) => x !== v) : [...list.value, v]
}

function reset(): void {
  q.value = ''
  areas.value = []
  facets.value = []
  kinds.value = []
  tiers.value = []
  evidence.value = []
  sort.value = 'tier'
}

const active = computed(
  () => !!(q.value || areas.value.length || facets.value.length || kinds.value.length || tiers.value.length || evidence.value.length),
)

function readUrl(): void {
  const p = new URLSearchParams(location.search)
  const list = (k: string) => (p.get(k) ? p.get(k)!.split(',').filter(Boolean) : [])
  q.value = p.get('q') ?? ''
  areas.value = list('area') as Area[]
  facets.value = list('facet') as Facet[]
  kinds.value = list('kind') as Kind[]
  tiers.value = list('tier') as Tier[]
  evidence.value = list('evidence') as Evidence[]
  sort.value = (p.get('sort') as SortKey) || 'tier'
  focusId.value = p.get('id')
}

function writeUrl(): void {
  const p = new URLSearchParams()
  if (q.value) p.set('q', q.value)
  if (areas.value.length) p.set('area', areas.value.join(','))
  if (facets.value.length) p.set('facet', facets.value.join(','))
  if (kinds.value.length) p.set('kind', kinds.value.join(','))
  if (tiers.value.length) p.set('tier', tiers.value.join(','))
  if (evidence.value.length) p.set('evidence', evidence.value.join(','))
  if (sort.value !== 'tier') p.set('sort', sort.value)
  const s = p.toString()
  history.replaceState(history.state, '', `${location.pathname}${s ? `?${s}` : ''}${location.hash}`)
}

onMounted(async () => {
  readUrl()
  if (focusId.value) {
    await nextTick()
    document.getElementById(`entry-${focusId.value}`)?.scrollIntoView({ block: 'center' })
  }
  watch([q, areas, facets, kinds, tiers, evidence, sort], writeUrl)
})
</script>

<template>
  <section class="lib">
    <div class="lib-bar">
      <label class="lib-search">
        <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><circle cx="9" cy="9" r="5.5" fill="none" stroke="currentColor" stroke-width="1.7" /><path d="m13.2 13.2 3.3 3.3" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" /></svg>
        <input v-model="q" type="search" placeholder="搜索标题、机构、标签、概括……" aria-label="搜索资料库" />
      </label>
      <div class="lib-sort" role="radiogroup" aria-label="排序">
        <button v-for="s in ([['tier', '按分级'], ['date', '按时间'], ['impact', '按影响力']] as const)" :key="s[0]" type="button" :class="{ on: sort === s[0] }" :aria-checked="sort === s[0]" role="radio" @click="sort = s[0]">
          {{ s[1] }}
        </button>
      </div>
    </div>

    <div class="lib-filters">
      <div class="lib-group">
        <span class="lib-group-label">领域</span>
        <button v-for="a in AREA_ORDER" :key="a" type="button" class="lib-chip" :class="{ on: areas.includes(a) }" :style="{ '--c': AREAS[a].color }" @click="toggle('area', a)">
          <span class="atlas-dot" />{{ AREAS[a].label }}<span class="lib-n">{{ count('area', (e) => e.areas.includes(a)) }}</span>
        </button>
      </div>
      <div class="lib-group">
        <span class="lib-group-label">视角</span>
        <button v-for="f in FACET_ORDER" :key="f" type="button" class="lib-chip" :class="{ on: facets.includes(f) }" @click="toggle('facet', f)">
          {{ FACETS[f].label }}<span class="lib-n">{{ count('facet', (e) => e.facets.includes(f)) }}</span>
        </button>
      </div>
      <div class="lib-group">
        <span class="lib-group-label">类型</span>
        <button v-for="k in kindOrder" :key="k" type="button" class="lib-chip" :class="{ on: kinds.includes(k) }" @click="toggle('kind', k)">
          {{ KINDS[k] }}<span class="lib-n">{{ count('kind', (e) => e.kind === k) }}</span>
        </button>
      </div>
      <div class="lib-group">
        <span class="lib-group-label">分级</span>
        <button v-for="t in TIER_ORDER" :key="t" type="button" class="lib-chip" :class="{ on: tiers.includes(t) }" :title="TIERS[t].hint" @click="toggle('tier', t)">
          {{ TIERS[t].label }}<span class="lib-n">{{ count('tier', (e) => e.tier === t) }}</span>
        </button>
      </div>
      <div class="lib-group">
        <span class="lib-group-label">证据</span>
        <button v-for="(v, k) in EVIDENCE" :key="k" type="button" class="lib-chip" :class="{ on: evidence.includes(k) }" :title="v.hint" @click="toggle('evidence', k)">
          {{ v.label }}<span class="lib-n">{{ count('evidence', (e) => e.evidence.includes(k)) }}</span>
        </button>
      </div>
    </div>

    <div class="lib-status">
      <span>共 {{ all.length }} 条，当前显示 <b>{{ results.length }}</b> 条</span>
      <button v-if="active" type="button" class="lib-reset" @click="reset">清空筛选</button>
      <span v-if="data.metricsUpdatedAt" class="lib-updated">指标更新于 {{ data.metricsUpdatedAt.slice(0, 10) }}</span>
    </div>

    <div class="lib-list">
      <EntryCard v-for="e in results" :key="e.id" :id="e.id" :highlight="e.id === focusId" :open="e.id === focusId" />
      <p v-if="!results.length" class="lib-empty">没有符合条件的条目，试试放宽筛选。</p>
    </div>
  </section>
</template>

<style scoped>
.lib {
  margin-top: 20px;
}
.lib-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.lib-search {
  flex: 1 1 280px;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 42px;
  padding: 0 14px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--card-bg);
  color: var(--vp-c-text-3);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.lib-search:focus-within {
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}
.lib-search input {
  flex: 1;
  min-width: 0;
  height: 100%;
  font-size: 14.5px;
  color: var(--vp-c-text-1);
  background: transparent;
}
.lib-sort {
  display: inline-flex;
  padding: 3px;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
}
.lib-sort button {
  padding: 7px 12px;
  border-radius: 9px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  transition: background 0.2s, color 0.2s;
}
.lib-sort button.on {
  background: var(--card-bg);
  color: var(--vp-c-text-1);
  font-weight: 600;
  box-shadow: var(--card-shadow);
}
.lib-filters {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 18px 0 8px;
  padding: 16px 16px 14px;
  border-radius: var(--radius-lg);
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
}
.lib-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.lib-group-label {
  width: 40px;
  flex: none;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-3);
}
.lib-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 12.5px;
  color: var(--vp-c-text-2);
  background: var(--card-bg);
  border: 1px solid var(--vp-c-divider);
  transition: all 0.18s var(--ease);
}
.lib-chip:hover {
  border-color: var(--vp-c-border);
  color: var(--vp-c-text-1);
}
.lib-chip.on {
  color: var(--vp-c-brand-1);
  border-color: color-mix(in srgb, var(--vp-c-brand-2) 55%, transparent);
  background: var(--vp-c-brand-soft);
  font-weight: 600;
}
.lib-n {
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-3);
}
.lib-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin: 14px 2px;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.lib-status b {
  color: var(--vp-c-text-1);
}
.lib-reset {
  font-size: 12.5px;
  color: var(--vp-c-brand-1);
}
.lib-updated {
  margin-left: auto;
}
.lib-list {
  display: grid;
  gap: 12px;
}
.lib-empty {
  padding: 40px 0;
  text-align: center;
  color: var(--vp-c-text-3);
}
@media (max-width: 640px) {
  .lib-group-label {
    width: 100%;
  }
}
</style>
