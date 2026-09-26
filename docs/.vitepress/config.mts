import { defineConfig } from 'vitepress'
import { katex } from '@mdit/plugin-katex'
import footnote from 'markdown-it-footnote'
import { containers } from './markdown/containers'
import { mermaidFence } from './markdown/mermaid'

const REPO = 'https://github.com/localityv-hash/flash-jax'

// GitHub Pages 的项目站点挂在 /<repo>/ 下；CI 会从 actions/configure-pages 传入真实路径。
const base = process.env.SITE_BASE || '/flash-jax/'

export default defineConfig({
  lang: 'zh-CN',
  title: '后训练图谱',
  titleTemplate: ':title · 后训练图谱',
  description:
    'Agentic RL、On-Policy 蒸馏、SFT、Mid-training 与 LLM 强化学习的中文知识地图：只收录经得起工业实践与复现检验的工作。',
  base,
  cleanUrls: true,
  // 本地预览未完成的页面时可设 ALLOW_DEAD_LINKS=1；CI 中始终检查死链。
  ignoreDeadLinks: process.env.ALLOW_DEAD_LINKS === '1',
  lastUpdated: true,
  appearance: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}logo.svg` }],
    ['meta', { name: 'theme-color', content: '#0f766e' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: '后训练图谱 · Post-Training Atlas' }],
    [
      'meta',
      {
        property: 'og:description',
        content: 'Agentic RL · On-Policy 蒸馏 · SFT · Mid-training：高质量论文、博客与项目的中文精读与谱系。',
      },
    ],
  ],

  markdown: {
    theme: { light: 'github-light', dark: 'github-dark' },
    config(md) {
      md.use(katex, {
        delimiters: 'dollars',
        throwOnError: false,
        strict: false,
        macros: {
          '\\E': '\\mathbb{E}',
          '\\KL': '\\mathrm{KL}',
          '\\clip': '\\operatorname{clip}',
          '\\sg': '\\operatorname{sg}',
        },
      })
      md.use(footnote)
      containers(md)
      mermaidFence(md)
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: '后训练图谱',

    nav: [
      { text: '开始', link: '/start/', activeMatch: '^/start/' },
      {
        text: '专题',
        activeMatch: '^/(topics|lenses)/',
        items: [
          {
            text: '训练阶段',
            items: [
              { text: 'Mid-training 中训练', link: '/topics/mid-training' },
              { text: 'SFT 监督微调', link: '/topics/sft' },
              { text: 'On-Policy 蒸馏', link: '/topics/opd' },
              { text: 'LLM 强化学习', link: '/topics/rl-for-llm' },
            ],
          },
          {
            text: '智能体',
            items: [
              { text: 'Agentic RL', link: '/topics/agentic-rl' },
              { text: '多环境与环境工程', link: '/topics/multi-env' },
            ],
          },
          {
            text: '横切视角',
            items: [
              { text: '算法谱系与推导', link: '/lenses/algorithms' },
              { text: '数据工作流', link: '/lenses/data' },
              { text: '训练系统 Infra', link: '/lenses/infra' },
              { text: '评测', link: '/lenses/eval' },
              { text: '原理与可解释性', link: '/lenses/principles' },
            ],
          },
          { text: '经典 RL 五年', items: [{ text: '经典 RL 关键进展', link: '/topics/classic-rl' }] },
        ],
      },
      { text: '实践', link: '/practice/', activeMatch: '^/practice/' },
      { text: '资料库', link: '/library/', activeMatch: '^/library/' },
      { text: '术语表', link: '/glossary', activeMatch: '^/glossary' },
    ],

    sidebar: [
      {
        text: '开始',
        items: [
          { text: '全景：后训练在做什么', link: '/start/' },
          { text: '阅读路线', link: '/start/paths' },
          { text: '收录标准', link: '/start/rubric' },
        ],
      },
      {
        text: '训练阶段',
        items: [
          { text: 'Mid-training 中训练', link: '/topics/mid-training' },
          { text: 'SFT 监督微调', link: '/topics/sft' },
          { text: 'On-Policy 蒸馏', link: '/topics/opd' },
          { text: 'LLM 强化学习', link: '/topics/rl-for-llm' },
        ],
      },
      {
        text: '智能体',
        items: [
          { text: 'Agentic RL', link: '/topics/agentic-rl' },
          { text: '多环境与环境工程', link: '/topics/multi-env' },
        ],
      },
      {
        text: '横切视角',
        items: [
          { text: '算法谱系与推导', link: '/lenses/algorithms' },
          { text: '数据工作流', link: '/lenses/data' },
          { text: '训练系统 Infra', link: '/lenses/infra' },
          { text: '评测', link: '/lenses/eval' },
          { text: '原理与可解释性', link: '/lenses/principles' },
        ],
      },
      {
        text: '经典 RL',
        items: [{ text: '经典 RL 五年关键进展', link: '/topics/classic-rl' }],
      },
      {
        text: '实践单元',
        collapsed: false,
        items: [
          { text: '总览', link: '/practice/' },
          { text: '数学 RLVR：GRPO → DAPO', link: '/practice/rlvr-math' },
          { text: '一次 On-Policy 蒸馏', link: '/practice/opd' },
          { text: '搜索智能体 RL', link: '/practice/search-agent' },
          { text: 'SWE 智能体 RL', link: '/practice/swe-agent' },
        ],
      },
      {
        text: '工具',
        items: [
          { text: '资料库', link: '/library/' },
          { text: '术语表', link: '/glossary' },
          { text: '参与贡献', link: '/contributing' },
        ],
      },
    ],

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新', formatOptions: { dateStyle: 'medium' } },
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    externalLinkIcon: true,
    notFound: {
      title: '这一页还没写',
      quote: '可能链接失效了，或者这里正在施工。',
      linkText: '回到首页',
    },

    editLink: {
      pattern: `${REPO}/edit/main/docs/:path`,
      text: '在 GitHub 上修改此页',
    },
    socialLinks: [{ icon: 'github', link: REPO }],
    footer: {
      message: '内容以中文撰写，指标由 CI 定期从 Semantic Scholar、Hugging Face 与 GitHub 自动刷新。',
      copyright: '后训练图谱 · Post-Training Atlas',
    },

    search: {
      provider: 'local',
      options: {
        detailedView: true,
        miniSearch: {
          options: {
            // 中文切成重叠的二元组，英文按整词。建索引（Node）和搜索（浏览器）必须切得完全一样，
            // 所以不用各环境词典不同的 Intl.Segmenter。函数会被序列化到浏览器端，必须自包含。
            tokenize: (text: string) => {
              const out: string[] = []
              for (const m of text.matchAll(/[㐀-鿿豈-﫿]+|[A-Za-z0-9_@+-]+/g)) {
                const s = m[0]
                if (/[㐀-鿿豈-﫿]/.test(s[0])) {
                  if (s.length === 1) out.push(s)
                  else for (let i = 0; i < s.length - 1; i++) out.push(s.slice(i, i + 2))
                } else out.push(s)
              }
              return out
            },
          },
          searchOptions: {
            prefix: true,
            fuzzy: 0.15,
            combineWith: 'AND',
            boost: { title: 4, text: 2, titles: 1 },
          },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详细列表',
            resetButtonTitle: '清除',
            backButtonTitle: '关闭搜索',
            noResultsText: '没有找到相关结果',
            footer: {
              selectText: '选择',
              selectKeyAriaLabel: '回车',
              navigateText: '切换',
              navigateUpKeyAriaLabel: '上',
              navigateDownKeyAriaLabel: '下',
              closeText: '关闭',
              closeKeyAriaLabel: 'Esc',
            },
          },
        },
      },
    },
  },

  vite: {
    optimizeDeps: { include: ['mermaid'] },
    ssr: { noExternal: [] },
  },
})
