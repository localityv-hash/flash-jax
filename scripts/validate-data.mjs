#!/usr/bin/env node
// 校验资料库、术语、谱系数据，以及 Markdown 页面里对它们的引用。出错时以非零状态退出。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import yaml from 'js-yaml'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const docs = path.join(root, 'docs')
const dataDir = path.join(docs, '.vitepress', 'data')

const AREAS = ['agentic-rl', 'opd', 'sft', 'mid-training', 'rl-llm', 'classic-rl']
const FACETS = ['algorithm', 'data', 'infra', 'eval', 'env', 'theory']
const KINDS = ['paper', 'blog', 'project', 'report', 'benchmark', 'dataset', 'book', 'talk']
const TIERS = ['must', 'rec', 'ref']
const EVIDENCE = ['industrial', 'adopted', 'ablation', 'open-source', 'community']
const ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/
const DATE = /^\d{4}-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?$/
const ARXIV = /^\d{4}\.\d{4,5}$/
const REPO = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/

const errors = []
const warnings = []
const err = (where, msg) => errors.push(`${where}: ${msg}`)
const warn = (where, msg) => warnings.push(`${where}: ${msg}`)

function yamlFiles(dir) {
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.yaml') || f.endsWith('.yml'))
    .map((f) => path.join(dir, f))
}

function load(file) {
  try {
    return yaml.load(fs.readFileSync(file, 'utf8'))
  } catch (e) {
    err(path.relative(root, file), `YAML 解析失败：${e.message.split('\n')[0]}`)
    return null
  }
}

/** 站内路径 /topics/sft#x -> 对应的 md 文件是否存在 */
function pageExists(link) {
  const clean = link.split('#')[0].split('?')[0].replace(/\.html$/, '')
  if (!clean || clean === '/') return true
  const rel = clean.replace(/^\//, '').replace(/\/$/, '/index')
  return fs.existsSync(path.join(docs, `${rel}.md`)) || fs.existsSync(path.join(docs, rel, 'index.md'))
}

// ---------- 条目 ----------
const entries = new Map()
const seenArxiv = new Map()
const seenUrl = new Map()
for (const file of yamlFiles(path.join(dataDir, 'entries'))) {
  const rel = path.relative(root, file)
  const list = load(file)
  if (list == null) continue
  if (!Array.isArray(list)) {
    err(rel, '顶层必须是列表')
    continue
  }
  for (const e of list) {
    const where = `${rel} › ${e?.id ?? '(无 id)'}`
    if (!e || typeof e !== 'object') {
      err(rel, '条目必须是对象')
      continue
    }
    if (!ID.test(e.id ?? '')) err(where, 'id 必须是小写 kebab-case')
    if (entries.has(e.id)) err(where, `id 重复（已在 ${entries.get(e.id).file} 中定义）`)
    for (const k of ['title', 'kind', 'org', 'date', 'areas', 'facets', 'tier', 'evidence', 'evidence_note', 'summary', 'plain'])
      if (e[k] == null || e[k] === '' || (Array.isArray(e[k]) && !e[k].length)) err(where, `缺少字段 ${k}`)
    if (e.kind && !KINDS.includes(e.kind)) err(where, `kind 非法：${e.kind}`)
    if (e.tier && !TIERS.includes(e.tier)) err(where, `tier 非法：${e.tier}`)
    if (e.date != null && !DATE.test(String(e.date))) err(where, `date 需为 YYYY-MM 或 YYYY-MM-DD：${e.date}`)
    for (const a of e.areas ?? []) if (!AREAS.includes(a)) err(where, `area 非法：${a}`)
    for (const f of e.facets ?? []) if (!FACETS.includes(f)) err(where, `facet 非法：${f}`)
    for (const v of e.evidence ?? []) if (!EVIDENCE.includes(v)) err(where, `evidence 非法：${v}`)
    if ((e.areas ?? []).length > 2) warn(where, 'areas 建议不超过 2 个')
    if ((e.facets ?? []).length > 3) warn(where, 'facets 建议不超过 3 个')
    if (e.arxiv != null) {
      const a = String(e.arxiv)
      if (!ARXIV.test(a)) err(where, `arxiv 编号格式不对：${a}`)
      else if (seenArxiv.has(a)) err(where, `arxiv ${a} 与 ${seenArxiv.get(a)} 重复`)
      else seenArxiv.set(a, e.id)
    }
    if (e.repo != null && !REPO.test(e.repo)) err(where, `repo 需为 owner/repo：${e.repo}`)
    if (e.url != null) {
      if (!/^https?:\/\//.test(e.url)) err(where, `url 需为 http(s) 链接：${e.url}`)
      const u = e.url.replace(/\/$/, '')
      if (seenUrl.has(u)) err(where, `url 与 ${seenUrl.get(u)} 重复`)
      else seenUrl.set(u, e.id)
    }
    if (!e.arxiv && !e.url && !e.repo && !(e.links ?? []).length) err(where, '至少需要 arxiv / url / repo / links 之一')
    for (const l of e.links ?? []) if (!l?.label || !/^https?:\/\//.test(l?.url ?? '')) err(where, `links 项需要 label 与 http(s) url`)
    if (e.tier === 'must') {
      const ev = e.evidence ?? []
      if (ev.length < 3 || !(ev.includes('industrial') || ev.includes('adopted')))
        err(where, '必读需至少 3 类证据且包含 industrial 或 adopted')
    }
    if (e.tier === 'rec' && (e.evidence ?? []).length < 2) err(where, '推荐需至少 2 类证据')
    // 公式按一个字符计：`$\\pi_\\theta$` 的 LaTeX 源码不该算进阅读长度
    const len = (s) => [...String(s ?? '').replace(/\$[^$\n]+?\$/g, 'x')].length
    if (len(e.summary) > 130) warn(where, `summary 偏长（${len(e.summary)} 字）`)
    if (len(e.plain) > 130) warn(where, `plain 偏长（${len(e.plain)} 字）`)
    if ((e.takeaways ?? []).length > 3) warn(where, 'takeaways 建议不超过 3 条')
    entries.set(e.id, { ...e, file: rel })
  }
}
for (const e of entries.values())
  for (const p of e.builds_on ?? []) if (!entries.has(p)) err(`${e.file} › ${e.id}`, `builds_on 引用了不存在的条目：${p}`)

// ---------- 术语 ----------
const terms = new Map()
for (const file of yamlFiles(path.join(dataDir, 'glossary'))) {
  const rel = path.relative(root, file)
  const list = load(file)
  if (!Array.isArray(list)) {
    if (list != null) err(rel, '顶层必须是列表')
    continue
  }
  for (const t of list) {
    const where = `${rel} › ${t?.id ?? '(无 id)'}`
    if (!ID.test(t?.id ?? '')) err(where, 'id 必须是小写 kebab-case')
    if (terms.has(t.id)) err(where, `术语 id 重复（已在 ${terms.get(t.id)} 中定义）`)
    for (const k of ['term', 'plain', 'definition']) if (!t[k]) err(where, `缺少字段 ${k}`)
    if (t.area && !AREAS.includes(t.area)) err(where, `area 非法：${t.area}`)
    for (const s of t.see ?? []) if (String(s).startsWith('/') && !pageExists(s)) err(where, `see 链接指向不存在的页面：${s}`)
    terms.set(t.id, rel)
  }
}

// ---------- 谱系 ----------
const graphs = new Map()
for (const file of yamlFiles(path.join(dataDir, 'lineage'))) {
  const rel = path.relative(root, file)
  const g = load(file)
  if (!g) continue
  if (!ID.test(g.id ?? '')) err(rel, 'id 必须是小写 kebab-case')
  if (graphs.has(g.id)) err(rel, `谱系 id 重复：${g.id}`)
  graphs.set(g.id, rel)
  const lanes = new Set((g.lanes ?? []).map((l) => l.id))
  if (!lanes.size) err(rel, '至少需要一条泳道')
  const nodes = new Set()
  for (const n of g.nodes ?? []) {
    const where = `${rel} › ${n.id}`
    if (nodes.has(n.id)) err(where, '节点 id 重复')
    nodes.add(n.id)
    if (!lanes.has(n.lane)) err(where, `泳道不存在：${n.lane}`)
    if (!DATE.test(String(n.date))) err(where, `date 格式不对：${n.date}`)
    if (n.entry && !entries.has(n.entry)) err(where, `entry 引用了不存在的条目：${n.entry}`)
    if (n.href && n.href.startsWith('/') && !pageExists(n.href)) err(where, `href 指向不存在的页面：${n.href}`)
    if ([...String(n.label ?? '')].length > 16) warn(where, 'label 超过 16 个字符，会被截断')
  }
  for (const e of g.edges ?? [])
    if (!nodes.has(e.from) || !nodes.has(e.to)) err(`${rel} › ${e.from}->${e.to}`, '边引用了不存在的节点')
}

// ---------- Markdown 引用 ----------
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    if (d.name.startsWith('.') || d.name === 'node_modules' || d.name === 'public') return []
    const p = path.join(dir, d.name)
    return d.isDirectory() ? walk(p) : p.endsWith('.md') ? [p] : []
  })
}
for (const file of walk(docs)) {
  const rel = path.relative(root, file)
  const src = fs.readFileSync(file, 'utf8')
  const body = src.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
  for (const m of body.matchAll(/<EntryCard\s+id="([^"]+)"/g)) if (!entries.has(m[1])) err(rel, `EntryCard 引用了不存在的条目：${m[1]}`)
  for (const m of body.matchAll(/:ids="\[([^\]]*)\]"/g))
    for (const id of m[1].split(',').map((s) => s.trim().replace(/^'|'$/g, '')).filter(Boolean))
      if (!entries.has(id)) err(rel, `EntryGrid 引用了不存在的条目：${id}`)
  for (const m of body.matchAll(/<Term\s+t="([^"]+)"/g)) if (!terms.has(m[1])) err(rel, `Term 引用了不存在的术语：${m[1]}`)
  for (const m of body.matchAll(/<LineageGraph\s+graph="([^"]+)"/g)) if (!graphs.has(m[1])) err(rel, `LineageGraph 引用了不存在的谱系：${m[1]}`)
  for (const m of body.matchAll(/\]\((\/[^)\s]*)\)/g)) if (!pageExists(m[1])) err(rel, `站内链接指向不存在的页面：${m[1]}`)
  for (const m of body.matchAll(/\/library\/\?id=([a-z0-9-]+)/g)) if (!entries.has(m[1])) err(rel, `资料库链接引用了不存在的条目：${m[1]}`)
}

// ---------- 输出 ----------
for (const w of warnings) console.warn(`  ⚠ ${w}`)
for (const e of errors) console.error(`  ✗ ${e}`)
console.log(
  `校验完成：${entries.size} 个条目，${terms.size} 个术语，${graphs.size} 张谱系图；${errors.length} 个错误，${warnings.length} 个警告。`,
)
process.exit(errors.length ? 1 : 0)
