
import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  FileWarning,
  LoaderCircle,
} from 'lucide-react'

import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

type Citation = {
  chunk_id: string
  page_number: number
  section: string
  text: string
  bboxes: number[][]
}

type PdfViewerProps = {
  file: File | null
  pdfUrl: string
  citation: Citation | null
}

export default function PdfViewer({
  file,
  pdfUrl,
  citation,
}: PdfViewerProps) {
  const [numPages, setNumPages] = useState(0)
  const [pageNumber, setPageNumber] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [containerWidth, setContainerWidth] = useState(700)
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 })
  const [error, setError] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)

  const [fileUrl, setFileUrl] = useState<string | null>(null)

useEffect(() => {
  if (!file) {
    setFileUrl(null)
    return
  }

  const url = URL.createObjectURL(file)
  setFileUrl(url)

  return () => {
    URL.revokeObjectURL(url)
  }
}, [file])

  useEffect(() => {
    const element = containerRef.current
    if (!element) return

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width
      if (width) setContainerWidth(Math.max(280, width - 32))
    })

    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (citation) {
      setPageNumber(citation.page_number)
    }
  }, [citation])

  useEffect(() => {
    setPageNumber(1)
    setNumPages(0)
    setZoom(1)
    setPageDimensions({ width: 0, height: 0 })
    setError('')
  }, [file, pdfUrl])

  const pdfSource = fileUrl || pdfUrl

  if (!pdfSource) {
    return (
      <div className="pdf-viewer-empty">
        <FileWarning size={30} />
        <h3>PDF preview unavailable</h3>
        <p>The original PDF could not be located.</p>
      </div>
    )
  }

  const pageWidth = Math.max(280, containerWidth * zoom)

  function handlePageLoadSuccess(page: {
    getViewport: (options: { scale: number }) => {
      width: number
      height: number
    }
  }) {
    const viewport = page.getViewport({ scale: 1 })
    setPageDimensions({
      width: viewport.width,
      height: viewport.height,
    })
  }

  const scale =
    pageDimensions.width > 0 ? pageWidth / pageDimensions.width : 1

  return (
    <section className="pdf-viewer">
      <div className="pdf-toolbar">
        <div className="pdf-toolbar-title">
          <span>DOCUMENT VIEWER</span>
          <strong>
            Page {pageNumber} {numPages > 0 && `of ${numPages}`}
          </strong>
        </div>

        <div className="pdf-controls">
          <button
            className="pdf-control-button"
            onClick={() => setZoom((value) => Math.max(0.6, value - 0.1))}
            disabled={zoom <= 0.6}
            title="Zoom out"
          >
            <ZoomOut size={16} />
          </button>
          <span className="pdf-zoom-value">{Math.round(zoom * 100)}%</span>
          <button
            className="pdf-control-button"
            onClick={() => setZoom((value) => Math.min(2, value + 0.1))}
            disabled={zoom >= 2}
            title="Zoom in"
          >
            <ZoomIn size={16} />
          </button>
          <span className="pdf-control-divider" />
          <button
            className="pdf-control-button"
            onClick={() => setPageNumber((value) => Math.max(1, value - 1))}
            disabled={pageNumber <= 1}
            title="Previous page"
          >
            <ChevronLeft size={17} />
          </button>
          <button
            className="pdf-control-button"
            onClick={() =>
              setPageNumber((value) => Math.min(numPages, value + 1))
            }
            disabled={pageNumber >= numPages}
            title="Next page"
          >
            <ChevronRight size={17} />
          </button>
        </div>
      </div>

      {citation && (
        <div className="pdf-citation-notice">
          <span />
          Showing citation from page {citation.page_number}
          {citation.section ? ` · ${citation.section}` : ''}
        </div>
      )}

      {error && <div className="pdf-error">{error}</div>}

      <div className="pdf-page-container" ref={containerRef}>
        <Document
          file={pdfSource}
          onLoadSuccess={({ numPages: total }) => {
            setNumPages(total)
            setError('')
          }}
          onLoadError={(reason) =>
            setError(`Unable to load PDF: ${reason.message}`)
          }
          loading={
            <div className="pdf-loading">
              <LoaderCircle className="spin" size={24} />
              Loading document...
            </div>
          }
          error={
            <div className="pdf-loading">
              Unable to render this PDF.
            </div>
          }
        >
          <div className="pdf-page-wrapper">
            <Page
              pageNumber={pageNumber}
              width={pageWidth}
              onLoadSuccess={handlePageLoadSuccess}
              loading={
                <div className="pdf-loading">
                  <LoaderCircle className="spin" size={22} />
                  Rendering page...
                </div>
              }
            />

            {citation?.page_number === pageNumber &&
              pageDimensions.width > 0 &&
              citation.bboxes?.map((bbox, index) => {
                if (bbox.length < 4) return null

                const [x0, y0, x1, y1] = bbox

                if (![x0, y0, x1, y1].every(Number.isFinite)) return null

                return (
                  <div
                    key={`${citation.chunk_id}-${index}`}
                    className="pdf-highlight"
                    style={{
                      left: x0 * scale,
                      top: y0 * scale,
                      width: Math.max(0, (x1 - x0) * scale),
                      height: Math.max(0, (y1 - y0) * scale),
                    }}
                  />
                )
              })}
          </div>
        </Document>
      </div>
    </section>
  )
}
