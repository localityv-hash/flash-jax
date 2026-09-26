import container from 'markdown-it-container'
import type MarkdownIt from 'markdown-it'

/** 站内统一的内容盒子。用法：`::: human 可选副标题` … `:::` */
const BOXES: Record<string, string> = {
  human: '说人话',
  takeaway: '可执行结论',
  pitfall: '踩坑提示',
  insight: '关键洞见',
  evidence: '可信度',
}

function subtitle(md: MarkdownIt, info: string, name: string): string {
  const rest = info.trim().slice(name.length).trim()
  return rest ? `<span class="box-sub">${md.renderInline(rest)}</span>` : ''
}

export function containers(md: MarkdownIt): void {
  for (const [name, label] of Object.entries(BOXES)) {
    md.use(container, name, {
      render(tokens: any[], idx: number) {
        const token = tokens[idx]
        if (token.nesting === 1) {
          return `<div class="box box-${name}"><p class="box-title"><span class="box-label">${label}</span>${subtitle(md, token.info, name)}</p>\n`
        }
        return '</div>\n'
      },
    })
  }

  // 推导默认折叠，避免正文信息过载；想看细节的读者自己展开。
  md.use(container, 'derive', {
    render(tokens: any[], idx: number) {
      const token = tokens[idx]
      if (token.nesting === 1) {
        const sub = subtitle(md, token.info, 'derive') || '<span class="box-sub">展开完整推导</span>'
        return `<details class="box box-derive"><summary><span class="box-label">推导</span>${sub}</summary>\n`
      }
      return '</details>\n'
    },
  })
}
