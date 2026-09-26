import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import './styles/vars.css'
import './styles/base.css'

import Layout from './Layout.vue'
import EntryCard from './components/EntryCard.vue'
import EntryGrid from './components/EntryGrid.vue'
import GlossaryList from './components/GlossaryList.vue'
import HomePage from './components/HomePage.vue'
import Library from './components/Library.vue'
import LineageGraph from './components/LineageGraph.vue'
import MermaidDiagram from './components/MermaidDiagram.vue'
import PipelineMap from './components/PipelineMap.vue'
import Term from './components/Term.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('EntryCard', EntryCard)
    app.component('EntryGrid', EntryGrid)
    app.component('GlossaryList', GlossaryList)
    app.component('HomePage', HomePage)
    app.component('Library', Library)
    app.component('LineageGraph', LineageGraph)
    app.component('MermaidDiagram', MermaidDiagram)
    app.component('PipelineMap', PipelineMap)
    app.component('Term', Term)
  },
} satisfies Theme
