<script setup lang="ts">
/**
 * 在专题页里嵌入一组条目。两种用法：
 *   <EntryGrid :ids="['deepseek-r1', 'dapo']" />          显式挑选（推荐，顺序即展示顺序）
 *   <EntryGrid area="opd" tier="must" :limit="6" />       按条件筛选
 */
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import type { Area, Facet, Tier } from '../../data/types'
import { TIERS } from '../lib/labels'
import EntryCard from './EntryCard.vue'

const props = defineProps<{
  ids?: string[]
  area?: Area
  facet?: Facet
  tier?: Tier
  limit?: number
  cols?: 1 | 2
  more?: boolean
}>()

const list = computed(() => {
  if (props.ids?.length) return props.ids
  let es = data.entries
  if (props.area) es = es.filter((e) => e.areas.includes(props.area!))
  if (props.facet) es = es.filter((e) => e.facets.includes(props.facet!))
  if (props.tier) es = es.filter((e) => TIERS[e.tier].rank <= TIERS[props.tier!].rank)
  es = [...es].sort((a, b) => TIERS[a.tier].rank - TIERS[b.tier].rank || b.date.localeCompare(a.date))
  return es.slice(0, props.limit ?? 6).map((e) => e.id)
})

const moreHref = computed(() => {
  const q = new URLSearchParams()
  if (props.area) q.set('area', props.area)
  if (props.facet) q.set('facet', props.facet)
  const s = q.toString()
  return withBase(`/library/${s ? `?${s}` : ''}`)
})
</script>

<template>
  <div class="entry-grid" :class="`cols-${cols ?? 2}`">
    <EntryCard v-for="id in list" :key="id" :id="id" />
  </div>
  <p v-if="more !== false && !ids?.length" class="entry-grid-more">
    <a :href="moreHref">在资料库中查看全部 →</a>
  </p>
</template>

<style scoped>
.entry-grid {
  display: grid;
  gap: 14px;
  margin: 22px 0;
}
.entry-grid.cols-2 {
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
}
.entry-grid.cols-1 {
  grid-template-columns: 1fr;
}
.entry-grid-more {
  margin-top: -6px !important;
  text-align: right;
  font-size: 13.5px;
}
</style>
