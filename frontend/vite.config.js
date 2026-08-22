import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = process.env.VITE_BACKEND_URL || `http://127.0.0.1:${process.env.PORT || 5001}`

// Bare .html pages and their legacy assets (css, js, images) are served by Flask in
// production; mirroring them here keeps post-login redirects and styling working on the
// Vite dev server instead of falling back to the React homepage.
const LEGACY_ASSETS_RE = '^/([a-zA-Z0-9_-]+\\.(html|css|js|png|jpg|jpeg|svg|mp4|webm|woff2?|json))$'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/video_feed': BACKEND,
      '/models': BACKEND,
      // Any bare legacy page & asset → Flask (replay.html, replay.css, theme.css, etc.)
      [LEGACY_ASSETS_RE]: BACKEND,
    },
  },
})
