import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

function resolveLegacyPath(rawPath) {
  if (!rawPath) return '/legacy/index.html'
  let clean = rawPath.trim().replace(/^#\/?/, '').replace(/^\//, '')
  if (!clean) return '/legacy/index.html'
  if (clean.startsWith('legacy/')) return `/${clean}`
  if (clean.endsWith('.html')) return `/legacy/${clean}`
  return `/legacy/${clean}.html`
}

function getInitialPage() {
  // 1. Prioritize window hash (e.g. #monitoring.html or #login.html)
  const hash = window.location.hash
  if (hash && hash.length > 1) {
    return resolveLegacyPath(hash)
  }

  // 2. Check saved session page
  const savedSession = sessionStorage.getItem('proctorai_active_page')
  if (savedSession) {
    return resolveLegacyPath(savedSession)
  }

  // 3. Check saved local page
  const savedLocal = localStorage.getItem('proctorai_active_page')
  if (savedLocal) {
    return resolveLegacyPath(savedLocal)
  }

  return '/legacy/index.html'
}

function App() {
  const iframeRef = useRef(null)
  const [initialSrc] = useState(getInitialPage)

  useEffect(() => {
    const iframe = iframeRef.current
    if (!iframe) return

    const syncCurrentPage = () => {
      try {
        const cw = iframe.contentWindow
        if (!cw || !cw.location) return
        const path = cw.location.pathname
        if (path && path !== 'about:blank') {
          const cleanName = path.replace(/^\/legacy\//, '').replace(/^\//, '')
          sessionStorage.setItem('proctorai_active_page', path)
          localStorage.setItem('proctorai_active_page', path)

          // Keep browser URL hash updated so refreshes and bookmarks preserve the page
          if (cleanName && cleanName !== 'index.html') {
            const newHash = `#${cleanName}`
            if (window.location.hash !== newHash) {
              window.history.replaceState(null, '', newHash)
            }
          } else {
            if (window.location.hash) {
              window.history.replaceState(null, '', window.location.pathname + window.location.search)
            }
          }
        }
      } catch (e) {
        // Cross-origin safety guard
      }
    }

    iframe.addEventListener('load', syncCurrentPage)

    // Periodic safety check in case page changes without full iframe reload
    const interval = setInterval(syncCurrentPage, 800)

    // Listen to hash changes in top window to navigate iframe if user presses browser back/forward
    const handleHashChange = () => {
      try {
        const targetPath = resolveLegacyPath(window.location.hash)
        const currentIframePath = iframe.contentWindow?.location?.pathname
        if (currentIframePath && currentIframePath !== targetPath) {
          iframe.contentWindow.location.replace(targetPath)
        }
      } catch (e) {}
    }

    window.addEventListener('hashchange', handleHashChange)

    return () => {
      iframe.removeEventListener('load', syncCurrentPage)
      clearInterval(interval)
      window.removeEventListener('hashchange', handleHashChange)
    }
  }, [])

  return (
    <iframe
      ref={iframeRef}
      className="legacy-app-frame"
      src={initialSrc}
      title="ProctorAI Enterprise Examination Integrity Platform"
    />
  )
}

createRoot(document.getElementById('root')).render(<App />)
