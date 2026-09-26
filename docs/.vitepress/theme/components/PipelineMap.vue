<script setup lang="ts">
/** 首页与全景页的「后训练流水线」：每个阶段一句人话，点进对应专题。 */
import { withBase } from 'vitepress'

interface Stage {
  no: string
  name: string
  en: string
  plain: string
  keys: string[]
  href?: string
  color: string
  muted?: boolean
}

const before: Stage[] = [
  {
    no: '00',
    name: '预训练',
    en: 'Pre-training',
    plain: '读遍互联网，学会语言、知识与基本推理。',
    keys: ['下一个词预测', '数据配比', 'Scaling'],
    color: 'var(--vp-c-text-3)',
    muted: true,
  },
  {
    no: '01',
    name: '中训练',
    en: 'Mid-training',
    plain: '用高质量、推理密集与长文本数据“回炉”，给后训练打地基。',
    keys: ['退火', '长上下文', '推理数据'],
    href: '/topics/mid-training',
    color: 'var(--area-mid)',
  },
  {
    no: '02',
    name: '监督微调',
    en: 'SFT',
    plain: '照着好答案学：学会格式、指令遵循和长链思考的“样子”。',
    keys: ['指令数据', '长 CoT', '数据配方'],
    href: '/topics/sft',
    color: 'var(--area-sft)',
  },
]

const branch: Stage[] = [
  {
    no: '03a',
    name: 'On-Policy 蒸馏',
    en: 'OPD',
    plain: '自己作答，老师逐词打分：用极少算力学到老师的本事。',
    keys: ['反向 KL', '稠密信号', '小模型'],
    href: '/topics/opd',
    color: 'var(--area-opd)',
  },
  {
    no: '03b',
    name: 'LLM 强化学习',
    en: 'RLHF → RLVR',
    plain: '自己作答，按结果给奖励：学到“抄答案”学不到的策略。',
    keys: ['GRPO 家族', '可验证奖励', 'KL 约束'],
    href: '/topics/rl-for-llm',
    color: 'var(--area-rl)',
  },
]

const after: Stage = {
  no: '04',
  name: 'Agentic RL',
  en: 'Agentic RL',
  plain: '走进真实环境：多轮行动、调用工具，用任务结果来学习。',
  keys: ['多轮 rollout', '工具调用', '多环境'],
  href: '/topics/agentic-rl',
  color: 'var(--area-agentic)',
}

const lenses = [
  { name: '算法谱系', href: '/lenses/algorithms' },
  { name: '数据工作流', href: '/lenses/data' },
  { name: '训练系统', href: '/lenses/infra' },
  { name: '评测', href: '/lenses/eval' },
  { name: '原理与可解释性', href: '/lenses/principles' },
]
</script>

<template>
  <div class="pm">
    <div class="pm-flow">
      <template v-for="s in before" :key="s.no">
        <component :is="s.href ? 'a' : 'div'" :href="s.href ? withBase(s.href) : undefined" class="pm-stage" :class="{ muted: s.muted }" :style="{ '--c': s.color }">
          <span class="pm-no">{{ s.no }}</span>
          <span class="pm-name">{{ s.name }}</span>
          <span class="pm-en">{{ s.en }}</span>
          <span class="pm-plain">{{ s.plain }}</span>
          <span class="pm-keys"><span v-for="k in s.keys" :key="k">{{ k }}</span></span>
        </component>
        <span class="pm-arrow" aria-hidden="true" />
      </template>

      <div class="pm-branch">
        <a v-for="s in branch" :key="s.no" :href="withBase(s.href!)" class="pm-stage" :style="{ '--c': s.color }">
          <span class="pm-no">{{ s.no }}</span>
          <span class="pm-name">{{ s.name }}</span>
          <span class="pm-en">{{ s.en }}</span>
          <span class="pm-plain">{{ s.plain }}</span>
          <span class="pm-keys"><span v-for="k in s.keys" :key="k">{{ k }}</span></span>
        </a>
      </div>
      <span class="pm-arrow" aria-hidden="true" />

      <a :href="withBase(after.href!)" class="pm-stage" :style="{ '--c': after.color }">
        <span class="pm-no">{{ after.no }}</span>
        <span class="pm-name">{{ after.name }}</span>
        <span class="pm-en">{{ after.en }}</span>
        <span class="pm-plain">{{ after.plain }}</span>
        <span class="pm-keys"><span v-for="k in after.keys" :key="k">{{ k }}</span></span>
      </a>
    </div>

    <div class="pm-lenses">
      <span class="pm-lenses-label">贯穿每一步的视角</span>
      <a v-for="l in lenses" :key="l.href" :href="withBase(l.href)">{{ l.name }}</a>
    </div>
  </div>
</template>

<style scoped>
.pm {
  margin: 8px 0 0;
  container-type: inline-size;
}
.pm-flow {
  display: grid;
  grid-template-columns: 1fr 22px 1fr 22px 1fr 22px 1.15fr 22px 1fr;
  align-items: stretch;
}
.pm-stage {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 18px 16px 16px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
  color: var(--vp-c-text-1);
  text-decoration: none;
  overflow: hidden;
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), border-color 0.25s var(--ease);
}
.pm-stage::before {
  content: '';
  position: absolute;
  inset: 0 0 auto 0;
  height: 3px;
  background: var(--c);
  opacity: 0.9;
}
a.pm-stage:hover {
  transform: translateY(-3px);
  box-shadow: var(--card-shadow-hover);
  border-color: color-mix(in srgb, var(--c) 40%, var(--card-border));
}
.pm-stage.muted {
  background: transparent;
  box-shadow: none;
  border-style: dashed;
}
.pm-stage.muted::before {
  display: none;
}
.pm-no {
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: var(--c);
  font-variant-numeric: tabular-nums;
}
.pm-name {
  font-size: 17px;
  font-weight: 720;
  line-height: 1.35;
}
.pm-en {
  font-size: 12px;
  color: var(--vp-c-text-3);
  letter-spacing: 0.02em;
}
.pm-plain {
  margin-top: 6px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}
.pm-keys {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: auto;
  padding-top: 10px;
}
.pm-keys span {
  font-size: 11.5px;
  padding: 2px 8px;
  border-radius: 6px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
}
.pm-flow > .pm-stage {
  align-self: center;
}
.pm-branch {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.pm-arrow {
  position: relative;
  align-self: center;
  height: 2px;
  margin: 0 3px;
  background: var(--vp-c-border);
}
.pm-arrow::after {
  content: '';
  position: absolute;
  right: -1px;
  top: -4px;
  border-left: 7px solid var(--vp-c-border);
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
}
.pm-lenses {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
}
.pm-lenses-label {
  margin-right: 6px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-3);
}
.pm-lenses a {
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 550;
  color: var(--vp-c-text-2);
  background: var(--card-bg);
  border: 1px solid var(--vp-c-divider);
  text-decoration: none;
  transition: color 0.2s, border-color 0.2s;
}
.pm-lenses a:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-2);
}

@container (max-width: 980px) {
  .pm-flow {
    grid-template-columns: 1fr;
    gap: 0;
  }
  .pm-arrow {
    justify-self: center;
    width: 2px;
    height: 22px;
    margin: 3px 0;
  }
  .pm-arrow::after {
    right: auto;
    left: -4px;
    top: auto;
    bottom: -1px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid var(--vp-c-border);
    border-bottom: none;
  }
  .pm-branch {
    flex-direction: row;
  }
  .pm-flow > .pm-stage,
  .pm-branch .pm-stage {
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 12px;
    align-items: baseline;
    padding: 14px 16px;
  }
  .pm-no {
    grid-row: span 2;
  }
  .pm-plain,
  .pm-keys {
    grid-column: 2;
  }
  .pm-keys {
    padding-top: 6px;
  }
}
@container (max-width: 520px) {
  .pm-branch {
    flex-direction: column;
  }
}
</style>
