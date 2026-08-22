import React from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

// The existing landing page is the source of truth for the visual design.
// React owns the application shell while the original HTML/CSS/JS runs intact
// inside the same-origin frame, preserving every animation and interaction.
function App() {
  return <iframe
    className="legacy-app-frame"
    src="/legacy/index.html"
    title="ProctorAI Enterprise Examination Integrity Platform"
  />
}

createRoot(document.getElementById('root')).render(<App />)
