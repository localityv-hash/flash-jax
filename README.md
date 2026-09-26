# 后训练图谱 · Post-Training Atlas

一个中文的后训练知识地图：收集近期高质量的 **Agentic RL、On-Policy 蒸馏（OPD）、SFT、Mid-training** 论文、博客与开源项目，以及近五年 **经典 RL** 与 **LLM 强化学习** 的关键工作。

- **只收经得起检验的工作**：工业实践、被工业采用、实验扎实、开源可验证、社区认可，五类证据逐条标注。
- **先专业，再说人话**：每个概念先给准确表述，再给新手也能懂的解释。
- **谱系与推导**：算法演化用可交互谱系图串起来，关键目标函数给出完整推导。
- **读起来不累**：分级筛选、术语悬浮解释、返回阅读点、续读提醒、全文搜索。

## 本地运行

```bash
npm install
npm run dev        # 本地预览
npm run build      # 校验数据并构建到 docs/.vitepress/dist
```

## 目录

```
docs/
  index.md                 首页
  start/                   全景、阅读路线、收录标准
  topics/                  专题：中训练、SFT、OPD、LLM 强化学习、Agentic RL、多环境、经典 RL
  lenses/                  横切视角：算法谱系、数据、Infra、评测、原理
  practice/                实践单元（最小可跑通的闭环）
  library/                 资料库（可筛选）
  glossary.md              术语表
  .vitepress/data/         条目、术语、谱系图数据（YAML）与指标
scripts/                   数据校验、指标抓取
```

## 部署到 GitHub Pages

仓库设置 → Pages → Source 选择 **GitHub Actions**。推送到 `main` 后 `.github/workflows/deploy.yml` 会自动构建并发布；每周一还会重新抓取引用数、Hugging Face 点赞与 GitHub star。

## 参与贡献

写作规范、数据格式与收录标准见 [CONTRIBUTING.md](CONTRIBUTING.md)。
