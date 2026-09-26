<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data } from '../../data/atlas.data'
import { AREA_ORDER, EVIDENCE } from '../lib/labels'
import EntryCard from './EntryCard.vue'
import PipelineMap from './PipelineMap.vue'

const { frontmatter } = useData()

const stats = computed(() => ({
  entries: data.entries.length,
  must: data.entries.filter((e) => e.tier === 'must').length,
  graphs: Object.keys(data.lineages).length,
  terms: data.glossary.length,
}))

/** 精选：frontmatter.featured 显式指定；否则每个领域挑一篇最新的必读。 */
const featured = computed<string[]>(() => {
  const fixed: string[] = frontmatter.value.featured ?? []
  if (fixed.length) return fixed
  const picks: string[] = []
  for (const a of AREA_ORDER) {
    const e = data.entries.find((x) => x.tier === 'must' && x.areas.includes(a) && !picks.includes(x.id))
    if (e) picks.push(e.id)
  }
  return picks.slice(0, 6)
})

const paths = [
  {
    tag: '入门',
    title: '我想搞懂后训练到底在做什么',
    desc: '不预设 RL 基础，先建立直觉，再补公式。',
    steps: ['全景地图', 'SFT', 'LLM 强化学习', 'On-Policy 蒸馏', 'Agentic RL'],
    href: '/start/paths#beginner',
  },
  {
    tag: '工程',
    title: '我要亲手训一个推理 / 智能体模型',
    desc: '从最小可跑通的闭环出发，补齐数据、系统和评测。',
    steps: ['数学 RLVR 实践', '训练系统 Infra', '数据工作流', '评测', 'SWE 智能体 RL'],
    href: '/start/paths#engineer',
  },
  {
    tag: '研究',
    title: '我想知道哪些问题还悬而未决',
    desc: '顺着算法谱系看每一次修正在修什么，再看原理层面的争论。',
    steps: ['算法谱系与推导', '原理与可解释性', '多环境', 'OPD 前沿'],
    href: '/start/paths#researcher',
  },
]
</script>

<template>
  <div class="home">
    <section class="hero">
      <p class="hero-kicker">Post-Training Atlas · 中文知识地图</p>
      <h1 class="hero-title">后训练图谱</h1>
      <p class="hero-lead">
        从 Mid-training、SFT、On-Policy 蒸馏到 Agentic RL，只收录<b>经得起工业实践与复现检验</b>的论文、博客与项目。
        每一篇都先用专业的话讲清楚，再用人话讲一遍；关键算法给出谱系与推导。
      </p>
      <div class="hero-actions">
        <a class="btn primary" :href="withBase('/start/')">从全景开始</a>
        <a class="btn" :href="withBase('/start/paths')">挑一条阅读路线</a>
        <a class="btn ghost" :href="withBase('/library/')">打开资料库</a>
      </div>
      <dl class="hero-stats">
        <div><dt>精选条目</dt><dd>{{ stats.entries }}</dd></div>
        <div><dt>必读</dt><dd>{{ stats.must }}</dd></div>
        <div><dt>谱系图</dt><dd>{{ stats.graphs }}</dd></div>
        <div><dt>术语</dt><dd>{{ stats.terms }}</dd></div>
      </dl>
    </section>

    <section class="block">
      <header class="block-head">
        <h2>一张图看懂后训练</h2>
        <p>模型从“会说话”到“会做事”要经过几道工序。点任意一站，进入对应专题。</p>
      </header>
      <PipelineMap />
    </section>

    <section class="block">
      <header class="block-head">
        <h2>按你的目标来读</h2>
        <p>不必从头读到尾。选一条路线，每一步都有明确的“读完能做什么”。</p>
      </header>
      <div class="paths">
        <a v-for="p in paths" :key="p.tag" class="path" :href="withBase(p.href)">
          <span class="path-tag">{{ p.tag }}</span>
          <span class="path-title">{{ p.title }}</span>
          <span class="path-desc">{{ p.desc }}</span>
          <ol class="path-steps">
            <li v-for="(s, i) in p.steps" :key="s"><span>{{ i + 1 }}</span>{{ s }}</li>
          </ol>
          <span class="path-go">查看路线 →</span>
        </a>
      </div>
    </section>

    <section class="block">
      <header class="block-head">
        <h2>精选必读</h2>
        <p>每个方向挑一篇“读了就能少走弯路”的工作。展开卡片可以看到说人话版本与可执行结论。</p>
      </header>
      <div class="featured">
        <EntryCard v-for="id in featured" :key="id" :id="id" />
      </div>
      <p class="more"><a :href="withBase('/library/?tier=must')">查看全部必读 →</a></p>
    </section>

    <section class="block rubric">
      <header class="block-head">
        <h2>我们怎么挑</h2>
        <p>宁缺毋滥。每个条目都标注它满足哪几类证据，读者可以自己判断可信度。</p>
      </header>
      <ul class="rubric-list">
        <li v-for="(v, k) in EVIDENCE" :key="k">
          <b>{{ v.label }}</b>
          <span>{{ v.hint }}</span>
        </li>
      </ul>
      <p class="more"><a :href="withBase('/start/rubric')">完整的收录标准 →</a></p>
    </section>
  </div>
</template>

<style scoped>
.home {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 24px 96px;
}
.hero {
  position: relative;
  padding: 72px 0 36px;
}
.hero::before {
  content: '';
  position: absolute;
  inset: -40px -10vw auto -10vw;
  height: 420px;
  z-index: -1;
  background:
    radial-gradient(600px 240px at 12% 30%, color-mix(in srgb, var(--vp-c-brand-3) 13%, transparent), transparent 70%),
    radial-gradient(520px 220px at 78% 10%, color-mix(in srgb, var(--area-agentic) 9%, transparent), transparent 70%);
  pointer-events: none;
}
.hero-kicker {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--vp-c-brand-1);
}
.hero-title {
  margin: 10px 0 0;
  font-size: clamp(40px, 7vw, 68px);
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.hero-lead {
  max-width: 760px;
  margin: 20px 0 0;
  font-size: 17.5px;
  line-height: 1.85;
  color: var(--vp-c-text-2);
}
.hero-lead b {
  color: var(--vp-c-text-1);
  font-weight: 650;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 28px;
}
.btn {
  display: inline-flex;
  align-items: center;
  height: 42px;
  padding: 0 20px;
  border-radius: 12px;
  font-size: 14.5px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  background: var(--card-bg);
  border: 1px solid var(--vp-c-divider);
  box-shadow: var(--card-shadow);
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease), background 0.2s;
}
.btn:hover {
  transform: translateY(-1px);
  box-shadow: var(--card-shadow-hover);
}
.btn.primary {
  color: #fff;
  background: var(--vp-c-brand-1);
  border-color: transparent;
}
.dark .btn.primary {
  color: #0d0f12;
}
.btn.ghost {
  background: transparent;
  box-shadow: none;
}
.hero-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 40px;
  margin: 40px 0 0;
}
.hero-stats div {
  display: flex;
  flex-direction: column-reverse;
}
.hero-stats dt {
  font-size: 12.5px;
  color: var(--vp-c-text-3);
  letter-spacing: 0.06em;
}
.hero-stats dd {
  margin: 0;
  font-size: 30px;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}

.block {
  margin-top: 72px;
}
.block-head h2 {
  margin: 0;
  font-size: 26px;
  font-weight: 750;
  letter-spacing: -0.01em;
}
.block-head p {
  margin: 8px 0 22px;
  max-width: 720px;
  font-size: 15px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
}

.paths {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 14px;
}
.path {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 20px 20px 16px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  color: var(--vp-c-text-1);
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease);
}
.path:hover {
  transform: translateY(-3px);
  box-shadow: var(--card-shadow-hover);
}
.path-tag {
  align-self: flex-start;
  padding: 2px 9px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.path-title {
  font-size: 17px;
  font-weight: 700;
  line-height: 1.45;
}
.path-desc {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}
.path-steps {
  list-style: none;
  margin: 6px 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.path-steps li {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13.5px;
  color: var(--vp-c-text-1);
}
.path-steps li span {
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 700;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font-variant-numeric: tabular-nums;
}
.path-go {
  margin-top: auto;
  padding-top: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.featured {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
  gap: 14px;
}
.more {
  margin-top: 14px;
  text-align: right;
  font-size: 14px;
}
.more a {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.rubric-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.rubric-list li {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 14px 16px;
  border-radius: var(--radius-md);
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
}
.rubric-list b {
  font-size: 14.5px;
}
.rubric-list span {
  font-size: 13px;
  line-height: 1.65;
  color: var(--vp-c-text-2);
}

@media (max-width: 640px) {
  .home {
    padding: 0 16px 72px;
  }
  .hero {
    padding-top: 40px;
  }
  .hero-lead {
    font-size: 16px;
  }
  .block {
    margin-top: 56px;
  }
}
</style>
