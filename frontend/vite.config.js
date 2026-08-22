import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://127.0.0.1:5001'

// Bare .html pages (admin.html, enrollment.html, …) are served by Flask in
// production; mirroring them here keeps post-login redirects working on the
// Vite dev server instead of falling back to the React homepage.
// Regex keys (^/(…)\.html$) match before Vite's SPA fallback kicks in.
const HTML_RE = '^/([a-z_]+\\.html)$'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/video_feed': BACKEND,
      '/models': BACKEND,
      // Any bare .html page → Flask (login, admin, enrollment, monitoring, …)
      [HTML_RE]: BACKEND,
    },
  },
})
