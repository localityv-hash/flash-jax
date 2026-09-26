<script setup lang="ts">
/** 术语表：按领域分组，可即时过滤；每个术语有稳定锚点，供 <Term> 与正文跳转。 */
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import { AREA_ORDER, AREAS, pageTitle } from '../lib/labels'
import type { Area, GlossaryTerm } from '../../data/types'

const q = ref('')

const groups = computed(() => {
  const needle = q.value.trim().toLowerCase()
  const hit = (g: GlossaryTerm) =>
    !needle || [g.term, g.full, g.zh, g.plain, g.definition].filter(Boolean).join(' ').toLowerCase().includes(needle)
  const out: { key: Area | 'general'; label: string; terms: GlossaryTerm[] }[] = []
  const general = data.glossary.filter((g) => !g.area && hit(g))
  if (general.length) out.push({ key: 'general', label: '通用概念', terms: general })
  for (const a of AREA_ORDER) {
    const terms = data.glossary.filter((g) => g.area === a && hit(g))
    if (terms.length) out.push({ key: a, label: AREAS[a].label, terms })
  }
  return out
})

function link(href: string): string {
  return href.startsWith('/') ? withBase(href) : href
}
</script>

<template>
  <div class="gl">
    <label class="gl-search">
      <input v-model="q" type="search" placeholder="过滤术语：如 KL、优势、rollout……" aria-label="过滤术语" />
    </label>
    <nav class="gl-index" aria-label="术语分组">
      <a v-for="grp in groups" :key="grp.key" :href="`#group-${grp.key}`">{{ grp.label }}<span>{{ grp.terms.length }}</span></a>
    </nav>
    <section v-for="grp in groups" :key="grp.key" class="gl-group">
      <h2 :id="`group-${grp.key}`" class="gl-group-title">{{ grp.label }}</h2>
      <dl>
        <div v-for="t in grp.terms" :id="t.id" :key="t.id" class="gl-item">
          <dt>
            <b>{{ t.term }}</b>
            <span v-if="t.zh && !t.zh.startsWith(t.term)" class="gl-zh">{{ t.zh }}</span>
            <span v-else-if="t.zh && t.zh !== t.term" class="gl-zh">{{ t.zh.slice(t.term.length).trim() }}</span>
            <span v-if="t.full" class="gl-full">{{ t.full }}</span>
            <a class="gl-anchor" :href="`#${t.id}`" aria-label="术语链接">#</a>
          </dt>
          <dd>
            <p class="gl-plain"><span class="gl-plain-label">说人话</span><span v-if="t.html?.plain" v-html="t.html.plain" /><template v-else>{{ t.plain }}</template></p>
            <p v-if="t.html?.definition" class="gl-def" v-html="t.html.definition" />
            <p v-else class="gl-def">{{ t.definition }}</p>
            <p v-if="t.see?.length" class="gl-see">
              <span class="gl-see-label">延伸：</span>
              <a v-for="s in t.see" :key="s" :href="link(s)">{{ pageTitle(s) }}</a>
            </p>
          </dd>
        </div>
      </dl>
    </section>
    <p v-if="!groups.length" class="gl-empty">没有匹配的术语。</p>
  </div>
</template>

<style scoped>
.gl-search input {
  width: 100%;
  height: 42px;
  padding: 0 14px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--card-bg);
  font-size: 14.5px;
}
.gl-search input:focus {
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}
.gl-index {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 14px 0 6px;
}
.gl-index a {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 13px;
  text-decoration: none !important;
  color: var(--vp-c-text-2) !important;
  background: var(--vp-c-bg-soft);
}
.gl-index a span {
  font-size: 11px;
  color: var(--vp-c-text-3);
}
.gl-group-title {
  margin-top: 40px !important;
}
.gl-item {
  padding: 16px 0;
  border-bottom: 1px solid var(--vp-c-divider);
  scroll-margin-top: 96px;
}
.gl-item:target {
  background: linear-gradient(90deg, var(--vp-c-brand-soft), transparent 70%);
  border-radius: 10px;
  padding-left: 12px;
}
.gl-item dt {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 10px;
}
.gl-item dt b {
  font-size: 16.5px;
}
.gl-zh {
  color: var(--vp-c-text-1);
}
.gl-full {
  font-size: 13px;
  color: var(--vp-c-text-3);
}
.gl-anchor {
  opacity: 0;
  font-size: 13px;
  text-decoration: none !important;
}
.gl-item:hover .gl-anchor {
  opacity: 1;
}
.gl-item dd {
  margin: 6px 0 0;
}
.gl-item dd p {
  margin: 6px 0 !important;
  font-size: 14.5px;
  line-height: 1.8 !important;
}
.gl-plain-label {
  margin-right: 8px;
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.06em;
  color: var(--box-human-ink);
}
.gl-def {
  color: var(--vp-c-text-2);
}
.gl-see {
  font-size: 13px !important;
}
.gl-see-label {
  color: var(--vp-c-text-3);
}
.gl-see a {
  margin-right: 10px;
}
.gl-empty {
  color: var(--vp-c-text-3);
}
</style>
