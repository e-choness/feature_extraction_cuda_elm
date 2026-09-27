import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const repo = 'https://github.com/e-choness/feature_extraction_cuda_elm'

// GitHub Pages serves the site from /<repo>/; override with DOCS_BASE for other hosts.
const base = process.env.DOCS_BASE ?? '/feature_extraction_cuda_elm/'

export default withMermaid(
  defineConfig({
    title: 'Feature ELM',
    description: 'GPU-accelerated Extreme Learning Machine feature extraction in C++20 and CUDA',
    base,
    lang: 'en-US',
    cleanUrls: true,
    lastUpdated: true,
    srcExclude: ['README.md', 'generated/**'],
    // localhost links point at a locally running demo.
    ignoreDeadLinks: [/^https?:\/\/localhost/],
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}logo.svg` }],
      ['meta', { name: 'theme-color', content: '#76b900' }],
    ],
    markdown: {
      math: false,
      lineNumbers: false,
    },
    themeConfig: {
      logo: '/logo.svg',
      siteTitle: 'Feature ELM',
      nav: [
        { text: 'Guide', link: '/getting-started', activeMatch: '^/(getting-started|quickstart|choosing-a-model)' },
        { text: 'Algorithms', link: '/architecture', activeMatch: '^/(architecture|elm|batch_elm|os_elm|reos_fos_elm|os_celm|elm_ae|ml_elm|rbf|h_os_elm)' },
        { text: 'Demos', link: '/demos' },
        { text: 'Benchmarks', link: '/benchmarks' },
        { text: 'API', link: '/api' },
        {
          text: 'More',
          items: [
            { text: 'Changelog', link: `${repo}/blob/master/CHANGELOG.md` },
            { text: 'Roadmap', link: '/roadmap' },
            { text: 'Citation', link: '/CITATION' },
          ],
        },
      ],
      sidebar: [
        {
          text: 'Getting started',
          items: [
            { text: 'Getting started', link: '/getting-started' },
            { text: 'Quickstart', link: '/quickstart' },
            { text: 'Choosing a model', link: '/choosing-a-model' },
          ],
        },
        {
          text: 'Concepts and architecture',
          collapsed: false,
          items: [
            { text: 'Architecture', link: '/architecture' },
            { text: 'Batch ELM', link: '/elm' },
            { text: 'Batch ELM quick reference', link: '/batch_elm' },
            { text: 'OS-ELM', link: '/os_elm' },
            { text: 'ReOS-ELM and FOS-ELM', link: '/reos_fos_elm' },
            { text: 'OS-CELM', link: '/os_celm' },
            { text: 'ELM-AE', link: '/elm_ae' },
            { text: 'ML-ELM', link: '/ml_elm' },
            { text: 'RBF features', link: '/rbf' },
            { text: 'RBF utilities', link: '/rbf_features' },
            { text: 'H-OS-ELM', link: '/h_os_elm' },
          ],
        },
        { text: 'Configuration', items: [{ text: 'Configuration', link: '/configuration' }] },
        {
          text: 'Operations',
          collapsed: false,
          items: [
            { text: 'Deployment', link: '/deployment' },
            { text: 'Troubleshooting', link: '/troubleshooting' },
            { text: 'Demos', link: '/demos' },
            { text: 'Benchmarks', link: '/benchmarks' },
          ],
        },
        {
          text: 'Contributor',
          collapsed: true,
          items: [
            { text: 'Building', link: '/building' },
            { text: 'Testing', link: '/testing' },
            { text: 'Style', link: '/style' },
            { text: 'API reference', link: '/api' },
            { text: 'Migration from v1 to v2', link: '/migration-v1-to-v2' },
            { text: 'Glossary', link: '/glossary' },
            { text: 'Roadmap', link: '/roadmap' },
          ],
        },
        {
          text: 'Upgrade notes',
          collapsed: true,
          items: [
            { text: 'Upgrade baseline', link: '/upgrade/baseline' },
            { text: 'H-OS-ELM migration', link: '/upgrade/h_os_elm' },
          ],
        },
        {
          text: 'Legal',
          collapsed: true,
          items: [
            { text: 'License', link: '/LICENSE' },
            { text: 'Citation', link: '/CITATION' },
          ],
        },
      ],
      socialLinks: [{ icon: 'github', link: repo }],
      editLink: {
        pattern: `${repo}/edit/master/docs/:path`,
        text: 'Edit this page on GitHub',
      },
      search: { provider: 'local' },
      outline: { level: [2, 3] },
      footer: {
        message: 'Released under the MIT License.',
        copyright: 'Copyright © Feature Extraction CUDA ELM contributors',
      },
    },
    mermaid: {
      theme: 'default',
    },
  }),
)
