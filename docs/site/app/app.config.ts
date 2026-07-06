export default defineAppConfig({
  ui: {
    colors: {
      primary: 'primary',
      neutral: 'zinc',
    },
  },
  seo: {
    siteName: 'PTWM',
  },
  header: {
    title: '',
    to: '/',
    logo: {
      alt: 'PTWM',
      light: '/logo.svg',
      dark: '/logo-dark.svg',
    },
    search: true,
    colorMode: true,
    links: [
      {
        'icon': 'i-simple-icons-github',
        'to': 'https://github.com/khwstolle/ptwm',
        'target': '_blank',
        'aria-label': 'GitHub',
      },
    ],
  },
  footer: {
    credits: `MIT · © 2022-${new Date().getFullYear()} Kurt H. W. Stolle`,
    colorMode: false,
    links: [
      {
        'icon': 'i-simple-icons-github',
        'to': 'https://github.com/khwstolle/ptwm',
        'target': '_blank',
        'aria-label': 'PTWM on GitHub',
      },
    ],
  },
  toc: {
    title: 'On this page',
    bottom: {
      title: 'Project',
      edit: 'https://github.com/khwstolle/ptwm/edit/master/docs/site/content',
      links: [
        {
          icon: 'i-lucide-star',
          label: 'Star on GitHub',
          to: 'https://github.com/khwstolle/ptwm',
          target: '_blank',
        },
        {
          icon: 'i-lucide-book-open',
          label: 'PyPI',
          to: 'https://pypi.org/project/ptwm/',
          target: '_blank',
        },
      ],
    },
  },
})
