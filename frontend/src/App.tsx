
import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowUp,
  CheckCircle2,
  FileText,
  FileUp,
  Files,
  FolderOpen,
  LoaderCircle,
  LockKeyhole,
  MessageSquare,
  Plus,
  ScanText,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import './App.css'
import PdfViewer from './PdfViewer'
import {
  getPresentationState,
  getStatusLabel,
} from './presentation'

const API_BASE = 'http://localhost:8000'

export type Citation = {
  chunk_id: string
  page_number: number
  section: string
  text: string
  bboxes: number[][]
}

type AskResponse = {
  document_id: string
  question: string
  answer: string
  insufficient_evidence: boolean
  citations: Citation[]
}

type DocumentItem = {
  document_id: string
  filename: string
  page_count: number
  chunk_count: number
  indexed: boolean
  uploaded_at: string
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  insufficient?: boolean
}

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])

  const [selectedId, setSelectedId] = useState<string | null>(() => {
    return localStorage.getItem('civiclens_selected_document')
  })

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null)
  const [pdfFiles, setPdfFiles] = useState<Record<string, File>>({})

  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)

  const selectedDocument =
    documents.find((document) => document.document_id === selectedId) || null

  useEffect(() => {
    let cancelled = false

    async function restoreDocuments() {
      try {
        const response = await fetch(`${API_BASE}/documents`)

        if (!response.ok) {
          throw new Error('Unable to restore saved documents.')
        }

        const data: { documents: DocumentItem[] } = await response.json()
        const restoredDocuments = data.documents

        if (cancelled) return

        setDocuments(restoredDocuments)

        const savedId = localStorage.getItem(
          'civiclens_selected_document',
        )
        const selectedDocument = restoredDocuments.find(
          (document) => document.document_id === savedId,
        )
        setSelectedId(
          selectedDocument?.document_id ??
            restoredDocuments[0]?.document_id ??
            null,
        )
        setError('')
      } catch (err) {
        if (cancelled) return
        setError(
          err instanceof Error
            ? err.message
            : 'Unable to restore saved documents.',
        )
      }
    }

    void restoreDocuments()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (selectedId) {
      localStorage.setItem('civiclens_selected_document', selectedId)
    } else {
      localStorage.removeItem('civiclens_selected_document')
    }
  }, [selectedId])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  function updateDocument(updated: DocumentItem) {
    setDocuments((current) =>
      current.map((document) =>
        document.document_id === updated.document_id ? updated : document,
      ),
    )
  }

  async function uploadDocument(file?: File) {
    if (!file) return

    setError('')
    setNotice('')

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.')
      return
    }

    if (file.size > 25 * 1024 * 1024) {
      setError('The maximum upload size is 25 MB.')
      return
    }

    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const uploadResponse = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        body: formData,
      })

      const uploadData = await uploadResponse.json()

      if (!uploadResponse.ok) {
        throw new Error(uploadData.detail || 'PDF upload failed.')
      }

      const newDocument: DocumentItem = {
        document_id: uploadData.document_id,
        filename: uploadData.filename,
        page_count: uploadData.page_count,
        chunk_count: uploadData.chunk_count,
        indexed: false,
        uploaded_at: new Date().toISOString(),
      }

      setDocuments((current) => [newDocument, ...current])
      setPdfFiles((current) => ({
        ...current,
        [newDocument.document_id]: file,
      }))
      setSelectedId(newDocument.document_id)
      setMessages([])

      setNotice('PDF uploaded. Preparing its searchable index...')

      const indexResponse = await fetch(
        `${API_BASE}/documents/${newDocument.document_id}/index`,
        { method: 'POST' },
      )

      const indexData = await indexResponse.json()

      if (!indexResponse.ok) {
        throw new Error(
          indexData.detail || 'PDF uploaded, but indexing failed.',
        )
      }

      updateDocument({ ...newDocument, indexed: true })
      setNotice(
        `Ready to explore. ${indexData.indexed_chunks} text chunks indexed.`,
      )
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while processing the PDF.',
      )
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function askQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!selectedDocument || !question.trim() || asking) return

    const currentQuestion = question.trim()

    setQuestion('')
    setError('')
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: currentQuestion,
      },
    ])
    setAsking(true)

    try {
      const response = await fetch(
        `${API_BASE}/documents/${selectedDocument.document_id}/ask`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: currentQuestion,
            top_k: 5,
          }),
        },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to generate an answer.')
      }

      const result = data as AskResponse

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          citations: result.citations,
          insufficient: result.insufficient_evidence,
        },
      ])
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to reach CivicLens.',
      )
    } finally {
      setAsking(false)
    }
  }

  async function deleteDocument() {
    if (!selectedDocument || deleting) return

    const confirmed = window.confirm(
      `Delete "${selectedDocument.filename}" and its associated files? This cannot be undone.`,
    )

    if (!confirmed) return

    setDeleting(true)
    setError('')
    setNotice('')

    try {
      const response = await fetch(
        `${API_BASE}/documents/${selectedDocument.document_id}`,
        { method: 'DELETE' },
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Document deletion failed.')
      }

      setDocuments((current) =>
        current.filter(
          (document) =>
            document.document_id !== selectedDocument.document_id,
        ),
      )
      setPdfFiles((current) => {
        const updated = { ...current }
        delete updated[selectedDocument.document_id]
        return updated
      })
      setSelectedId(null)
      setMessages([])
      setActiveCitation(null)
      setNotice('Document and associated files deleted.')
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to delete this document.',
      )
    } finally {
      setDeleting(false)
    }
  }

  function selectDocument(document: DocumentItem) {
    setSelectedId(document.document_id)
    setMessages([])
    setActiveCitation(null)
    setError('')
    setNotice('')
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    uploadDocument(event.dataTransfer.files[0])
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ScanText size={21} />
          </div>
          <div>
            <div className="brand-name">CivicLens</div>
            <div className="brand-caption">DOCUMENT INTELLIGENCE</div>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">WORKSPACE</div>
          <button className="nav-item active">
            <Activity size={17} />
            Overview
          </button>
          <button
            className="nav-item"
            onClick={() => fileInputRef.current?.click()}
          >
            <Files size={17} />
            Add document
            <Plus size={15} className="nav-trailing" />
          </button>
        </div>

        <div className="sidebar-section document-section">
          <div className="sidebar-heading">
            <span className="sidebar-label">YOUR DOCUMENTS</span>
            <span className="document-count">{documents.length}</span>
          </div>

          {documents.length === 0 ? (
            <div className="sidebar-empty">
              <FolderOpen size={20} />
              <span>No documents yet</span>
            </div>
          ) : (
            <div className="document-list">
              {documents.map((document) => (
                <button
                  key={document.document_id}
                  className={`document-nav-item ${
                    selectedId === document.document_id ? 'selected' : ''
                  }`}
                  onClick={() => selectDocument(document)}
                >
                  <FileText size={16} />
                  <span className="document-nav-name">{document.filename}</span>
                  {document.indexed && (
                    <span className="document-ready-dot" title="Indexed" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="privacy-card">
            <ShieldCheck size={17} />
            <div>
              <strong>Privacy first</strong>
              <span>Processed locally on your machine.</span>
            </div>
          </div>
          <div className="sidebar-version">CivicLens · Local workspace</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>{selectedDocument ? 'Document workspace' : 'Overview'}</strong>
          </div>
          <div className="topbar-right">
            <span className="local-indicator">
              <span />
              Local environment
            </span>
            <div className="topbar-avatar">C</div>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">
                <Sparkles size={14} />
                YOUR DOCUMENT, MADE CLEARER
              </div>
              <h1>Understand the fine print.</h1>
              <p className="welcome-description">
                Turn complex official documents into clear answers and
                actionable next steps.
              </p>
            </div>
            <button
              className="primary-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              <Plus size={17} />
              Add document
            </button>
          </section>

          {(error || notice) && (
            <div className={`feedback ${error ? 'feedback-error' : 'feedback-success'}`}>
              {error ? <AlertCircle size={17} /> : <CheckCircle2 size={17} />}
              <span>{error || notice}</span>
              <button
                className="feedback-close"
                onClick={() => {
                  setError('')
                  setNotice('')
                }}
                aria-label="Dismiss message"
              >
                <X size={15} />
              </button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            hidden
            onChange={(event) => uploadDocument(event.target.files?.[0])}
          />

          {!selectedDocument ? (
            <>
              <section
                className={`upload-panel ${dragging ? 'dragging' : ''}`}
                onDragOver={(event) => {
                  event.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <div className="upload-icon">
                  {uploading ? (
                    <LoaderCircle className="spin" size={27} />
                  ) : (
                    <FileUp size={27} />
                  )}
                </div>
                <h2>{uploading ? 'Processing your document...' : 'Start with a document'}</h2>
                <p>
                  {uploading
                    ? 'Extracting text and building your searchable index. This may take a moment.'
                    : 'Drop an official PDF here, or browse your files to upload.'}
                </p>
                <button
                  className="primary-button upload-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                >
                  {uploading ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <Upload size={17} />
                  )}
                  {uploading ? 'Please wait...' : 'Choose PDF'}
                </button>
                <span className="upload-footnote">
                  PDF files only · Maximum 25 MB
                </span>
              </section>

              <section className="feature-grid">
                <div className="feature-card">
                  <div className="feature-icon blue">
                    <MessageSquare size={19} />
                  </div>
                  <h3>Ask questions</h3>
                  <p>
                    Get answers grounded in the actual content of your
                    document.
                  </p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon violet">
                    <ScanText size={19} />
                  </div>
                  <h3>Trace every answer</h3>
                  <p>
                    See the source passages and page references behind
                    responses.
                  </p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon green">
                    <LockKeyhole size={19} />
                  </div>
                  <h3>Keep it private</h3>
                  <p>
                    Your documents are processed using your local application
                    setup.
                  </p>
                </div>
              </section>
            </>
          ) : (
            <section className="workspace">
              <div className="workspace-header">
                <div className="selected-file">
                  <div className="file-icon">
                    <FileText size={21} />
                  </div>
                  <div className="selected-file-info">
                    <h2 title={selectedDocument.filename}>
                      {selectedDocument.filename}
                    </h2>
                    <span>
                      {selectedDocument.page_count} pages
                      <span className="metadata-separator">·</span>
                      {selectedDocument.chunk_count} text chunks
                    </span>
                  </div>
                </div>
                <div className="workspace-actions">
                  <span
                    className={`index-status ${
                      selectedDocument.indexed ? 'indexed' : 'not-indexed'
                    }`}
                  >
                    {selectedDocument.indexed ? (
                      <CheckCircle2 size={14} />
                    ) : (
                      <AlertCircle size={14} />
                    )}
                    {selectedDocument.indexed ? 'Indexed' : 'Not indexed'}
                  </span>
                  <button
                    className="icon-button danger"
                    onClick={deleteDocument}
                    disabled={deleting}
                    title="Delete document"
                  >
                    {deleting ? (
                      <LoaderCircle className="spin" size={17} />
                    ) : (
                      <Trash2 size={17} />
                    )}
                  </button>
                </div>
              </div>

              <div className="chat-panel">
                <div className="chat-heading">
                  <div className="chat-heading-icon">
                    <Sparkles size={17} />
                  </div>
                  <div>
                    <h3>Ask your document</h3>
                    <p>Answers are generated from retrieved document evidence.</p>
                  </div>
                </div>

                <div className="chat-messages">
                  {messages.length === 0 && (
                    <div className="chat-empty">
                      <div className="chat-empty-icon">
                        <MessageSquare size={23} />
                      </div>
                      <h3>What would you like to understand?</h3>
                      <p>
                        Ask about eligibility, application requirements,
                        deadlines or any detail in this document.
                      </p>
                      <div className="suggested-questions">
                        {[
                          'What are the eligibility requirements?',
                          'What documents do I need to submit?',
                          'What are the important deadlines?',
                        ].map((suggestion) => (
                          <button
                            key={suggestion}
                            onClick={() => setQuestion(suggestion)}
                            disabled={!selectedDocument.indexed || asking}
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {messages.map((message) => (
                    <div
                      className={`message ${
                        message.role === 'user' ? 'user-message' : 'assistant-message'
                      }`}
                      key={message.id}
                    >
                      <div className="message-avatar">
                        {message.role === 'user' ? 'Y' : <Sparkles size={15} />}
                      </div>
                      <div className="message-content">
                        <div className="message-author">
                          {message.role === 'user' ? 'You' : 'CivicLens'}
                          {(() => {
                            const state = getPresentationState(
                              message.insufficient,
                              message.citations,
                            )
                            const label = getStatusLabel(state)
                            if (!label) return null
                            return (
                              <span
                                className={`insufficient-label insufficient-label--${state}`}
                                data-presentation-state={state}
                              >
                                {label}
                              </span>
                            )
                          })()}
                        </div>
                        <p className="answer-text">{message.content}</p>

                        {message.citations && message.citations.length > 0 && (
                          <div className="citation-list">
                            <div className="citation-heading">
                              <ShieldCheck size={14} />
                              SOURCES FROM YOUR DOCUMENT
                            </div>
                            {message.citations.map((citation) => (
                              <button
                                key={citation.chunk_id}
                                className={`citation-card ${
                                  activeCitation?.chunk_id === citation.chunk_id
                                    ? 'citation-active'
                                    : ''
                                }`}
                                onClick={() => setActiveCitation(citation)}
                              >
                                <div className="citation-card-top">
                                  <span>Page {citation.page_number}</span>
                                  <span>{citation.section}</span>
                                </div>
                                <p>{citation.text}</p>
                                <span className="citation-action">
                                  <ScanText size={13} />
                                  View source passage
                                </span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {asking && (
                    <div className="thinking-row">
                      <div className="message-avatar assistant-avatar">
                        <Sparkles size={15} />
                      </div>
                      <div className="thinking-bubble">
                        <LoaderCircle size={15} className="spin" />
                        Searching evidence and preparing an answer...
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <form className="chat-composer" onSubmit={askQuestion}>
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder={
                      selectedDocument.indexed
                        ? 'Ask anything about this document...'
                        : 'Index this document before asking questions.'
                    }
                    rows={2}
                    disabled={!selectedDocument.indexed || asking}
                    onKeyDown={(event) => {
                      if (
                        event.key === 'Enter' &&
                        !event.shiftKey &&
                        !event.nativeEvent.isComposing
                      ) {
                        event.preventDefault()
                        event.currentTarget.form?.requestSubmit()
                      }
                    }}
                  />
                  <div className="composer-bottom">
                    <span>
                      <LockKeyhole size={12} />
                      Grounded answers · Local LLM
                    </span>
                    <button
                      type="submit"
                      className="send-button"
                      disabled={!question.trim() || asking || !selectedDocument.indexed}
                      aria-label="Send question"
                    >
                      {asking ? (
                        <LoaderCircle size={17} className="spin" />
                      ) : (
                        <ArrowUp size={18} />
                      )}
                    </button>
                  </div>
                </form>
              </div>

              <div className="source-panel">
                <div className="source-panel-heading">
                  <div>
                    <span className="eyebrow">DOCUMENT READER</span>
                    <h3>{selectedDocument.filename}</h3>
                  </div>
                </div>

                <PdfViewer
                  file={pdfFiles[selectedDocument.document_id] || null}
                  pdfUrl={`${API_BASE}/documents/${selectedDocument.document_id}/file`}
                  citation={activeCitation}
                />
              </div>
            </section>
          )}

          <footer className="page-footer">
            <span>© 2026 CivicLens</span>
            <span>
              <ShieldCheck size={13} />
              Privacy-first document intelligence
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
