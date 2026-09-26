import type MarkdownIt from 'markdown-it'

/**
 * 把 ```mermaid 标题``` 代码块渲染成 <MermaidDiagram>，在浏览器端按当前明暗主题绘制。
 * 代码经 encodeURIComponent 编码，避免与 Vue 模板语法冲突。
 */
export function mermaidFence(md: MarkdownIt): void {
  const fallback = md.renderer.rules.fence!
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const [lang, ...rest] = token.info.trim().split(/\s+/)
    if (lang !== 'mermaid') return fallback(tokens, idx, options, env, self)
    const caption = rest.join(' ')
    const cap = caption ? ` caption="${md.utils.escapeHtml(caption)}"` : ''
    return `<MermaidDiagram code="${encodeURIComponent(token.content)}"${cap} />\n`
  }
}
