/** 资料库条目（论文 / 博客 / 项目 / 技术报告……）。字段说明见 CONTRIBUTING.md。 */
export type Area = 'agentic-rl' | 'opd' | 'sft' | 'mid-training' | 'rl-llm' | 'classic-rl'
export type Facet = 'algorithm' | 'data' | 'infra' | 'eval' | 'env' | 'theory'
export type Kind = 'paper' | 'blog' | 'project' | 'report' | 'benchmark' | 'dataset' | 'book' | 'talk'
export type Tier = 'must' | 'rec' | 'ref'
export type Evidence = 'industrial' | 'adopted' | 'ablation' | 'open-source' | 'community'

export interface EntryLink {
  label: string
  url: string
}

export interface Metrics {
  citations?: number
  influential?: number
  hf_upvotes?: number
  stars?: number
  forks?: number
  pushed_at?: string
}

export interface Entry {
  id: string
  title: string
  /** 图谱与紧凑卡片里用的短名，例如 "GRPO"、"DeepSeek-R1" */
  short?: string
  kind: Kind
  org: string
  authors?: string
  /** YYYY-MM 或 YYYY-MM-DD，首次公开日期 */
  date: string
  venue?: string
  arxiv?: string
  /** GitHub owner/repo，用于抓取 star/fork */
  repo?: string
  /** 主链接（博客、项目主页）；有 arxiv 时可省略 */
  url?: string
  links?: EntryLink[]
  areas: Area[]
  facets: Facet[]
  tier: Tier
  evidence: Evidence[]
  evidence_note: string
  summary: string
  plain: string
  takeaways?: string[]
  builds_on?: string[]
  tags?: string[]
  /** 由加载器合并进来的指标（来自 metrics.json） */
  metrics?: Metrics | null
}

export interface GlossaryTerm {
  id: string
  term: string
  full?: string
  zh?: string
  plain: string
  definition: string
  see?: string[]
  area?: Area
}

export interface LineageNode {
  id: string
  label: string
  date: string
  lane: string
  note?: string
  /** 站内锚点或页面链接 */
  href?: string
  /** 对应资料库条目 id */
  entry?: string
}

export interface LineageEdge {
  from: string
  to: string
  label?: string
}

export interface Lineage {
  id: string
  title: string
  description?: string
  lanes: { id: string; label: string }[]
  nodes: LineageNode[]
  edges: LineageEdge[]
}

export interface AtlasData {
  entries: Entry[]
  glossary: GlossaryTerm[]
  lineages: Record<string, Lineage>
  metricsUpdatedAt: string | null
}
