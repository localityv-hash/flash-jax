<script setup lang="ts">
/** 资料库条目卡片：先给专业一句话，展开后给「说人话」、可执行结论与谱系。 */
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import { AREAS, EVIDENCE, KINDS, TIERS } from '../lib/labels'
import { compact, entryLinks, month, primaryUrl } from '../lib/format'

const props = withDefaults(defineProps<{ id: string; open?: boolean; highlight?: boolean }>(), {
  open: false,
  highlight: false,
})

const byId = new Map(data.entries.map((e) => [e.id, e]))
const entry = computed(() => byId.get(props.id))
const expanded = ref(props.open)

const parents = computed(() =>
  (entry.value?.builds_on ?? []).map((id) => ({ id, name: byId.get(id)?.short ?? byId.get(id)?.title ?? id })),
)
const metrics = computed(() => {
  const m = entry.value?.metrics
  if (!m) return []
  const out: { label: string; value: string; title: string }[] = []
  if (m.citations) out.push({ label: '引用', value: compact(m.citations), title: `Semantic Scholar 引用数 ${m.citations}` })
  if (m.hf_upvotes) out.push({ label: 'HF 赞', value: compact(m.hf_upvotes), title: `Hugging Face Daily Papers 点赞 ${m.hf_upvotes}` })
  if (m.stars) out.push({ label: 'Star', value: compact(m.stars), title: `GitHub star ${m.stars}` })
  if (m.forks) out.push({ label: 'Fork', value: compact(m.forks), title: `GitHub fork ${m.forks}` })
  return out
})
</script>

<template>
  <article
    v-if="entry"
    :id="`entry-${entry.id}`"
    class="entry"
    :class="[`tier-${entry.tier}`, { expanded, highlight }]"
  >
    <header class="entry-head">
      <span class="entry-tier" :title="TIERS[entry.tier].hint">{{ TIERS[entry.tier].label }}</span>
      <span class="entry-kind">{{ KINDS[entry.kind] }}</span>
      <span class="entry-meta" :title="entry.org"><span class="entry-org">{{ entry.org }}</span> · {{ month(entry.date) }}</span>
      <span class="entry-areas">
        <span v-for="a in entry.areas" :key="a" class="entry-area" :style="{ '--c': AREAS[a].color }" :title="AREAS[a].label">
          <span class="atlas-dot" />{{ AREAS[a].short }}
        </span>
      </span>
    </header>

    <h4 class="entry-title">
      <a :href="primaryUrl(entry)" target="_blank" rel="noopener noreferrer">{{ entry.title }}</a>
    </h4>
    <p v-if="entry.html?.summary" class="entry-summary" v-html="entry.html.summary" />
    <p v-else class="entry-summary">{{ entry.summary }}</p>

    <div class="entry-signals">
      <span v-for="ev in entry.evidence" :key="ev" class="entry-evidence" :title="EVIDENCE[ev].hint">
        <svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="m2.5 6.2 2.2 2.2 4.8-4.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" /></svg>
        {{ EVIDENCE[ev].label }}
      </span>
      <span v-for="m in metrics" :key="m.label" class="entry-metric" :title="m.title">
        <span class="entry-metric-label">{{ m.label }}</span>{{ m.value }}
      </span>
    </div>

    <div v-if="expanded" class="entry-more">
      <p class="entry-plain"><span class="entry-plain-label">说人话</span><span v-if="entry.html?.plain" v-html="entry.html.plain" /><template v-else>{{ entry.plain }}</template></p>
      <div v-if="entry.takeaways?.length" class="entry-takeaways">
        <p class="entry-sub">可执行结论</p>
        <ul>
          <li v-for="(t, i) in entry.takeaways" :key="i"><span v-if="entry.html?.takeaways?.[i]" v-html="entry.html.takeaways[i]" /><template v-else>{{ t }}</template></li>
        </ul>
      </div>
      <p class="entry-note"><span class="entry-sub-inline">可信度</span><span v-if="entry.html?.evidence_note" v-html="entry.html.evidence_note" /><template v-else>{{ entry.evidence_note }}</template></p>
      <p v-if="parents.length" class="entry-parents">
        <span class="entry-sub-inline">承接</span>
        <a v-for="p in parents" :key="p.id" :href="withBase(`/library/?id=${p.id}`)" class="entry-parent">{{ p.name }}</a>
      </p>
    </div>

    <footer class="entry-foot">
      <span class="entry-links">
        <a v-for="l in entryLinks(entry)" :key="l.url" :href="l.url" target="_blank" rel="noopener noreferrer">{{ l.label }}</a>
      </span>
      <button type="button" class="entry-toggle" :aria-expanded="expanded" @click="expanded = !expanded">
        {{ expanded ? '收起' : '说人话 · 要点' }}
      </button>
    </footer>
  </article>
  <div v-else class="entry entry-missing">未找到条目：{{ id }}</div>
</template>

<style scoped>
.entry {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px 18px 12px;
  border: 1px solid var(--card-border);
  border-radius: var(--radius-lg);
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  transition: box-shadow 0.25s var(--ease), border-color 0.25s var(--ease), transform 0.25s var(--ease);
  scroll-margin-top: 96px;
}
.entry:hover {
  box-shadow: var(--card-shadow-hover);
}
.entry.highlight {
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft), var(--card-shadow-hover);
}
.entry-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 10px;
  font-size: 12px;
  color: var(--vp-c-text-3);
}
.entry-tier {
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 11.5px;
}
.tier-must .entry-tier {
  color: #fff;
  background: var(--vp-c-brand-1);
}
.dark .tier-must .entry-tier {
  color: #0d0f12;
}
.tier-rec .entry-tier {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.tier-ref .entry-tier {
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
}
.entry-kind {
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.entry-meta {
  display: inline-flex;
  min-width: 0;
  max-width: 100%;
  white-space: nowrap;
}
.entry-org {
  display: inline-block;
  max-width: 190px;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: bottom;
}
.entry-areas {
  display: inline-flex;
  gap: 8px;
  margin-left: auto;
}
.entry-area {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.entry-title {
  margin: 2px 0 0 !important;
  font-size: 16.5px !important;
  line-height: 1.45 !important;
  font-weight: 680 !important;
  letter-spacing: -0.003em;
}
.entry-title a {
  color: var(--vp-c-text-1) !important;
  text-decoration: none !important;
  font-weight: inherit !important;
}
.entry-title a:hover {
  color: var(--vp-c-brand-1) !important;
}
.entry-summary {
  margin: 0 !important;
  font-size: 14.5px !important;
  line-height: 1.75 !important;
  color: var(--vp-c-text-2);
}
.entry-signals {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 2px;
}
.entry-evidence,
.entry-metric {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 22px;
  padding: 0 8px;
  border-radius: 6px;
  font-size: 11.5px;
  white-space: nowrap;
}
.entry-evidence {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.entry-metric {
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.entry-metric-label {
  font-weight: 400;
  color: var(--vp-c-text-3);
}
.entry-more {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 6px;
  padding-top: 12px;
  border-top: 1px dashed var(--vp-c-divider);
}
.entry-plain {
  margin: 0 !important;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--box-human-bg);
  font-size: 14.5px !important;
  line-height: 1.8 !important;
}
.entry-plain-label,
.entry-sub-inline {
  display: inline-block;
  margin-right: 8px;
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.06em;
}
.entry-plain-label {
  color: var(--box-human-ink);
}
.entry-sub,
.entry-sub-inline {
  color: var(--vp-c-text-3);
}
.entry-sub {
  margin: 0 0 4px !important;
  font-size: 12px !important;
  font-weight: 750;
  letter-spacing: 0.06em;
}
.entry-takeaways ul {
  margin: 0 !important;
  padding-left: 1.2em !important;
}
.entry-takeaways li {
  font-size: 14px !important;
  line-height: 1.75 !important;
  margin: 2px 0 !important;
}
.entry-note,
.entry-parents {
  margin: 0 !important;
  font-size: 13.5px !important;
  line-height: 1.7 !important;
  color: var(--vp-c-text-2);
}
.entry-parent {
  margin-right: 10px;
  font-size: 13px;
}
.entry-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: auto;
  padding-top: 4px;
}
.entry-links {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
}
.entry-links a {
  font-size: 13px;
  font-weight: 600 !important;
  text-decoration: none !important;
  color: var(--vp-c-text-2) !important;
}
.entry-links a:hover {
  color: var(--vp-c-brand-1) !important;
}
.entry-toggle {
  flex: none;
  font-size: 12.5px;
  font-weight: 600;
  padding: 4px 11px;
  border-radius: 999px;
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
  transition: color 0.2s, border-color 0.2s, background 0.2s;
}
.entry-toggle:hover,
.entry.expanded .entry-toggle {
  color: var(--vp-c-brand-1);
  border-color: color-mix(in srgb, var(--vp-c-brand-2) 45%, transparent);
  background: var(--vp-c-brand-soft);
}
.entry-missing {
  color: var(--box-pitfall-ink);
  font-size: 13px;
}
</style>
