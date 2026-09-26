import fs from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'
import { defineLoader } from 'vitepress'
import type { AtlasData, Entry, GlossaryTerm, Lineage } from './types'

declare const data: AtlasData
export { data }

function readYaml<T>(file: string): T {
  return (yaml.load(fs.readFileSync(file, 'utf8')) ?? []) as T
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
    }
    entries.sort(byDateDesc)
    glossary.sort((a, b) => a.term.localeCompare(b.term, 'zh-Hans-CN'))
    return { entries, glossary, lineages, metricsUpdatedAt }
  },
})
