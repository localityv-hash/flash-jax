import fs from 'node:fs'
import path from 'node:path'
import katex from 'katex'
import yaml from 'js-yaml'
import { defineLoader } from 'vitepress'
import type { AtlasData, Entry, GlossaryTerm, Lineage } from './types'

declare const data: AtlasData
export { data }

function readYaml<T>(file: string): T {
  return (yaml.load(fs.readFileSync(file, 'utf8')) ?? []) as T
}

const MACROS = {
  '\\E': '\\mathbb{E}',
  '\\KL': '\\mathrm{KL}',
  '\\clip': '\\operatorname{clip}',
  '\\sg': '\\operatorname{sg}',
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * 条目与术语里的短文本支持 `$公式$` 与 `` `代码` ``，在构建时渲染成 HTML，浏览器端不需要 KaTeX 脚本。
 * 没有公式或代码的文本返回 undefined，组件直接显示纯文本。
 */
function rich(text: string | undefined): string | undefined {
  if (!text || !/[$`]/.test(text)) return undefined
  const out: string[] = []
  let last = 0
  for (const m of text.matchAll(/\$([^$\n]+?)\$|`([^`\n]+)`/g)) {
    out.push(escapeHtml(text.slice(last, m.index)))
    out.push(
      m[1] != null
        ? katex.renderToString(m[1], { throwOnError: false, strict: false, macros: MACROS })
        : `<code>${escapeHtml(m[2])}</code>`,
    )
    last = (m.index ?? 0) + m[0].length
  }
  out.push(escapeHtml(text.slice(last)))
  return out.join('')
}

/** 按日期倒序，同日期按 id，保证输出稳定。 */
function byDateDesc(a: Entry, b: Entry): number {
  return b.date.localeCompare(a.date) || a.id.localeCompare(b.id)
}

export default defineLoader({
  watch: ['./entries/*.yaml', './glossary/*.yaml', './lineage/*.yaml', './metrics.json'],
  load(files: string[]): AtlasData {
    const entries: Entry[] = []
    const glossary: GlossaryTerm[] = []
    const lineages: Record<string, Lineage> = {}
    let metrics: Record<string, Entry['metrics']> = {}
    let metricsUpdatedAt: string | null = null

    for (const file of [...files].sort()) {
      const dir = path.basename(path.dirname(file))
      if (file.endsWith('metrics.json')) {
        const raw = JSON.parse(fs.readFileSync(file, 'utf8'))
        metrics = raw.entries ?? {}
        metricsUpdatedAt = raw.generated_at ?? null
      } else if (dir === 'entries') {
        entries.push(...readYaml<Entry[]>(file))
      } else if (dir === 'glossary') {
        glossary.push(...readYaml<GlossaryTerm[]>(file))
      } else if (dir === 'lineage') {
        const graph = readYaml<Lineage>(file)
        lineages[graph.id] = graph
      }
    }

    for (const e of entries) {
      e.date = String(e.date)
      e.metrics = metrics[e.id] ?? null
      e.html = {
        summary: rich(e.summary),
        plain: rich(e.plain),
        evidence_note: rich(e.evidence_note),
        takeaways: e.takeaways?.map((t) => rich(t)),
      }
    }
    for (const t of glossary) t.html = { plain: rich(t.plain), definition: rich(t.definition) }
    entries.sort(byDateDesc)
    glossary.sort((a, b) => a.term.localeCompare(b.term, 'zh-Hans-CN'))
    return { entries, glossary, lineages, metricsUpdatedAt }
  },
})
