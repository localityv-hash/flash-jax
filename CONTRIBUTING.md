# 参与贡献与写作规范

这份文档是本站所有内容的“宪法”：收什么、怎么写、数据怎么填、组件怎么用。新增条目或页面前请通读一遍。

## 1. 定位

- **精选，不求全。** 宁可少收，也不收“看起来很热闹”的工作。每个条目都要能回答：*为什么值得读？证据是什么？*
- **先专业，再说人话。** 每个概念先给准确的专业表述，再给一段新手也能懂的解释（“说人话”）。两者都不能错。
- **可执行。** 读者读完应该知道“遇到什么情况该怎么做”，而不只是“这篇论文做了什么”。
- **不为分类而分类。** 专题按训练阶段组织；数据、算法、Infra、评测、原理作为横切视角；实践单元围绕一个可跑通的最小闭环聚合。

## 2. 收录标准

### 2.1 五类证据

| 证据 | 判定标准（满足其一即可） |
|---|---|
| `industrial` 工业实践 | 作者团队用它训练了公开发布的工业级模型或系统（DeepSeek、Qwen、Kimi、Seed、MiniMax、GLM、OpenAI、Anthropic、Google DeepMind、Meta、NVIDIA、Mistral、Microsoft、AI2 等的正式模型）；或它本身就是生产级系统。 |
| `adopted` 被工业采用 | 至少一个**其他团队**的工业级工作在技术报告或代码里明确采用、改进了它。`evidence_note` 里写明是谁。 |
| `ablation` 实验扎实 | 有控制变量的消融、足够的规模与多基准验证，结论能直接指导实践。 |
| `open-source` 开源可验证 | 有真实可运行的代码、数据或权重（给出 `repo`），并能从多个信息源交叉验证其内容。 |
| `community` 社区认可 | 论文高引用或在 Hugging Face Daily Papers / alphaXiv 上高赞；项目高 star/fork；博客被广泛转引、讨论且内容正确。参考量级：引用 ≥ 100（发布一年内 ≥ 30），HF 赞 ≥ 50，star ≥ 1k。 |

### 2.2 分级

| 分级 | 规则 | 每个专题大约 |
|---|---|---|
| `must` 必读 | 至少 3 类证据，且包含 `industrial` 或 `adopted`；并且改变了大家的做法。 | 5–8 篇 |
| `rec` 推荐 | 至少 2 类证据。 | 若干 |
| `ref` 参考 | 1 类强证据，解决一个具体问题；或非常新（3 个月内）但来源可靠、值得跟踪。 | 按需 |

**不收：** 只刷榜不解释、无代码无复现且无工业背书、结论不可执行的工作；营销稿；无法核实来源的二手转述。

### 2.3 指标不手填

引用数、点赞数、star/fork 由 CI 在部署时从 Semantic Scholar、Hugging Face、GitHub 自动抓取，写入 `docs/.vitepress/data/metrics.json`。条目里**只填标识**（`arxiv`、`repo`），不要手写数字。`evidence_note` 可以写量级判断（如“GitHub 上最常用的开源 RL 训练框架之一”），但不写具体数字。

## 3. 条目格式（`docs/.vitepress/data/entries/*.yaml`）

```yaml
- id: dapo                       # 小写 kebab-case，全站唯一且稳定
  title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"   # 原标题
  short: DAPO                    # 图谱与紧凑视图里的短名
  kind: paper                    # paper | blog | project | report | benchmark | dataset | book | talk
  org: ByteDance Seed · 清华 AIR # 机构，多个用 · 分隔
  authors: "Qiying Yu et al."    # 可选，第一作者 et al.
  date: 2025-03                  # 首次公开时间，YYYY-MM 或 YYYY-MM-DD
  venue: NeurIPS 2025            # 可选
  arxiv: "2503.14476"            # 可选，只写编号，不带版本号
  repo: BytedTsinghua-SIA/DAPO   # 可选，GitHub owner/repo
  url: https://dapo-sia.github.io/  # 可选，主链接（博客/项目页）；有 arxiv 时可省略
  links:                         # 可选，额外链接
    - { label: 数据集, url: https://… }
  areas: [rl-llm]                # 1–2 个：agentic-rl | opd | sft | mid-training | rl-llm | classic-rl
  facets: [algorithm]            # 1–3 个：algorithm | data | infra | eval | env | theory
  tier: must                     # must | rec | ref
  evidence: [industrial, adopted, ablation, open-source, community]
  evidence_note: "……具体写谁采用、复现情况、为何可信……"
  summary: "专业一句话（40–90 字）：做了什么 + 关键机制 + 结论。"
  plain: "说人话（30–90 字）：打个比方也行，但不能错。"
  takeaways:                     # 1–3 条可执行结论
    - "如果你……，就……"
  builds_on: [grpo]              # 前驱条目 id，用于谱系
  tags: [RLVR, 长 CoT]
```

**硬性要求**

- 日期、机构、arXiv 编号、仓库名必须有可靠来源；不确定的可选字段宁可省略，**绝不编造**。
- `summary` 与 `plain` 都写中文；专有名词保留英文（GRPO、rollout、KL）。
- 条目与术语里的短文本（`summary`、`plain`、`takeaways`、`evidence_note`、`definition`）支持 `$公式$` 与 `` `代码` ``，构建时渲染；不要写 `π_θ(a|s)` 这类伪公式，写成 `$\pi_\theta(a\mid s)$`。
- `takeaways` 必须是可执行的判断或做法，不是摘要的复述。
- 同一工作只收一次；有论文又有博客时，以更完整、更可信的那个为主条目，其余放 `links`。

## 4. 页面写作规范

### 4.1 专题页结构

1. **一句话定义**（专业）+ `::: human` 说人话
2. **在流水线中的位置**：它解决什么问题、输入输出是什么、和前后阶段怎么衔接
3. **核心概念与推导**：公式 + 每个符号的直觉；长推导放进 `::: derive`
4. **演化脉络**：谱系图（`<LineageGraph>`）+ 每次改进在修什么
5. **关键工作精读**：`<EntryGrid>` + 对每篇的“一段话点评”
6. **可执行结论**：`::: takeaway`，3–6 条
7. **常见坑**：`::: pitfall`
8. **延伸阅读**：链接到资料库筛选视图和相关专题

### 4.2 语言

- 读者：有机器学习基础的工程师和研究生，同时照顾新手。
- 术语首次出现写成“中文（English，缩写）”，之后用缩写；也可以用 `<Term>` 挂上悬浮解释。
- 短句，一段一个意思；少用形容词，多用数字、对比和因果。避免翻译腔（“被进行”“对……进行……”）。
- 博客描述要专业：说清作者、背景、核心论点、证据强度，以及它在社区里引发了什么讨论。
- “说人话”块不超过 3 句，用生活类比，但类比不能引入错误直觉。
- 控制信息密度：每节不超过 5 个要点，列表不超过 7 项；大表格或次要细节放进 `::: derive` 或 `::: details`。
- 结论要标出处：正文里链接到原文或资料库条目；关键数字用脚注 `[^1]` 注明来源。

### 4.3 记号约定

| 记号 | 含义 |
|---|---|
| $x$ | 提示（prompt） |
| $y=(y_1,\dots,y_{\lvert y\rvert})$ | 回答（token 序列），$y_{<t}$ 为前缀 |
| $\pi_\theta$ | 正在训练的策略（模型） |
| $\pi_{\theta_\text{old}}$ | 生成这批样本时的策略 |
| $\pi_\text{ref}$ | 参考策略（通常是 SFT 后的模型） |
| $\pi_T,\ \pi_S$ | 蒸馏中的教师、学生 |
| $r(x,y)$ | 序列级奖励 |
| $\hat A_t$ | 第 $t$ 个 token 的优势估计 |
| $\rho_t(\theta)=\dfrac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\theta_\text{old}}(y_t\mid x,y_{<t})}$ | 重要性比率 |
| $\varepsilon$ | 裁剪范围 |
| $G$ | 每个提示的采样数（组大小） |
| $\beta$ | KL 系数 |
| $\gamma,\ \lambda$ | 折扣因子、GAE 参数 |
| $\mathcal D$ | 提示数据集 |

可用宏：`\E`（期望）、`\KL`、`\clip`、`\sg`（stop-gradient）。行内公式 `$…$`，独立公式 `$$…$$`。

## 5. 组件与语法

| 写法 | 效果 |
|---|---|
| `::: tldr` … `:::` | 页首“30 秒速读”：3–5 条要点 + 建议先读哪一节 |
| `::: human` … `:::` | “说人话”盒子 |
| `::: takeaway` / `::: pitfall` / `::: insight` / `::: evidence` | 可执行结论 / 踩坑提示 / 关键洞见 / 可信度说明（后面可跟副标题） |
| `::: derive 标题` … `:::` | 默认折叠的推导 |
| `::: timeline` + 以 `**年份**` 开头的列表 | 竖排时间线 |
| ` ```mermaid 图标题 ` | 流程图、时序图等，自动适配明暗主题，可放大 |
| `<Term t="grpo">GRPO</Term>` | 术语悬浮解释（id 见术语表数据） |
| `<EntryCard id="dapo" />` | 单个条目卡片 |
| `<EntryGrid :ids="['dapo', 'dr-grpo']" />` | 指定条目网格（推荐，顺序即展示顺序） |
| `<EntryGrid area="opd" tier="must" :limit="6" />` | 按条件自动挑选 |
| `<LineageGraph graph="policy-gradient" />` | 谱系图（数据见下） |
| `<PipelineMap />` | 后训练流水线总图 |

页面 frontmatter：

```yaml
---
title: On-Policy 蒸馏
kicker: 训练阶段          # 标题上方的眉题
level: 进阶               # 入门 | 进阶 | 深入
prereq:                   # 可选：建议先读
  - { text: SFT, link: /topics/sft }
---
```

### 5.1 术语（`docs/.vitepress/data/glossary/*.yaml`）

```yaml
- id: grpo                # 小写 kebab-case，全站唯一
  term: GRPO
  full: Group Relative Policy Optimization
  zh: 组相对策略优化
  area: rl-llm            # 可选；不填归入“通用概念”
  plain: "说人话解释，一两句。"
  definition: "专业定义，一两句，可含公式的文字描述。"
  see: [/topics/rl-for-llm#grpo]   # 站内延伸链接
```

### 5.2 谱系图（`docs/.vitepress/data/lineage/<id>.yaml`）

```yaml
id: policy-gradient
title: 策略梯度家族：每一次修正在修什么
description: 一句话说明这张图怎么读。
lanes:                                  # 列（2–5 条泳道）
  - { id: classic, label: 经典 RL }
nodes:
  - id: ppo
    label: PPO                          # 不超过 15 个字符
    date: 2017-07
    lane: classic
    note: "一句话：它修了前任的什么问题。"
    href: "#ppo"                        # 可选：站内锚点或页面
    entry: ppo                          # 可选：资料库条目 id
edges:
  - { from: ppo, to: grpo, label: 去掉价值网络 }
```

## 6. 本地开发

```bash
npm install
npm run dev        # 本地预览（热更新）
npm run validate   # 校验数据与站内引用
npm run build      # 校验 + 构建静态站点到 docs/.vitepress/dist
```
