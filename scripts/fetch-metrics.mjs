#!/usr/bin/env node
// 为资料库条目抓取社区指标，写入 docs/.vitepress/data/metrics.json。
//   - Semantic Scholar：引用数、有影响力引用数（按 arXiv 编号批量查询）
//   - Hugging Face Daily Papers：点赞数
//   - GitHub：star、fork、最近推送时间
// 任何一个来源失败都不会中断构建：保留上一次的值。可选环境变量：GITHUB_TOKEN、S2_API_KEY。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import yaml from 'js-yaml'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const dataDir = path.join(root, 'docs', '.vitepress', 'data')
const outFile = path.join(dataDir, 'metrics.json')

const entries = fs
  .readdirSync(path.join(dataDir, 'entries'))
  .filter((f) => f.endsWith('.yaml'))
  .flatMap((f) => yaml.load(fs.readFileSync(path.join(dataDir, 'entries', f), 'utf8')) ?? [])

const previous = fs.existsSync(outFile) ? JSON.parse(fs.readFileSync(outFile, 'utf8')) : { entries: {} }
const metrics = structuredClone(previous.entries ?? {})
const stats = { s2: 0, hf: 0, gh: 0, failed: 0 }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function request(url, init = {}, tries = 4) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { ...init, signal: AbortSignal.timeout(20000) })
      if (res.status === 404) return null
      if (res.status === 429 || res.status >= 500) {
        await sleep(1500 * 2 ** i)
        continue
      }
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      return await res.json()
    } catch (e) {
      if (i === tries - 1) throw e
      await sleep(1000 * 2 ** i)
    }
  }
  throw new Error(`请求多次失败：${url}`)
}

async function pool(items, limit, fn) {
  const queue = [...items]
  await Promise.all(
    Array.from({ length: limit }, async () => {
      while (queue.length) await fn(queue.shift())
    }),
  )
}

function set(id, patch) {
  metrics[id] = { ...(metrics[id] ?? {}), ...patch }
}

// ---------- Semantic Scholar ----------
async function semanticScholar() {
  const withArxiv = entries.filter((e) => e.arxiv)
  const headers = { 'Content-Type': 'application/json' }
  if (process.env.S2_API_KEY) headers['x-api-key'] = process.env.S2_API_KEY
  for (let i = 0; i < withArxiv.length; i += 400) {
    const chunk = withArxiv.slice(i, i + 400)
    try {
      const res = await request('https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount,influentialCitationCount', {
        method: 'POST',
        headers,
        body: JSON.stringify({ ids: chunk.map((e) => `ARXIV:${e.arxiv}`) }),
      })
      res?.forEach((paper, j) => {
        if (!paper) return
        set(chunk[j].id, { citations: paper.citationCount ?? undefined, influential: paper.influentialCitationCount ?? undefined })
        stats.s2++
      })
    } catch (e) {
      stats.failed++
      console.warn(`Semantic Scholar 批量查询失败：${e.message}`)
    }
  }
}

// ---------- Hugging Face Daily Papers ----------
async function huggingFace() {
  await pool(
    entries.filter((e) => e.arxiv),
    4,
    async (e) => {
      try {
        const paper = await request(`https://huggingface.co/api/papers/${e.arxiv}`)
        if (paper && typeof paper.upvotes === 'number') {
          set(e.id, { hf_upvotes: paper.upvotes })
          stats.hf++
        }
      } catch (err) {
        stats.failed++
        console.warn(`HF ${e.arxiv} 失败：${err.message}`)
      }
    },
  )
}

// ---------- GitHub ----------
async function github() {
  const headers = { Accept: 'application/vnd.github+json', 'User-Agent': 'post-training-atlas-metrics' }
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  await pool(
    entries.filter((e) => e.repo),
    4,
    async (e) => {
      try {
        const repo = await request(`https://api.github.com/repos/${e.repo}`, { headers })
        if (repo) {
          set(e.id, { stars: repo.stargazers_count, forks: repo.forks_count, pushed_at: repo.pushed_at?.slice(0, 10) })
          stats.gh++
        }
      } catch (err) {
        stats.failed++
        console.warn(`GitHub ${e.repo} 失败：${err.message}`)
      }
    },
  )
}

await Promise.all([semanticScholar(), huggingFace(), github()])

// 删除已不在资料库中的条目
const ids = new Set(entries.map((e) => e.id))
for (const id of Object.keys(metrics)) if (!ids.has(id)) delete metrics[id]

fs.writeFileSync(
  outFile,
  `${JSON.stringify({ generated_at: new Date().toISOString(), sources: ['semantic-scholar', 'huggingface', 'github'], entries: metrics }, null, 1)}\n`,
)
console.log(`指标已更新：S2 ${stats.s2}，HF ${stats.hf}，GitHub ${stats.gh}，失败 ${stats.failed}。写入 ${path.relative(root, outFile)}`)
