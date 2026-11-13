# CivicLens Frontend

The user interface for CivicLens — a privacy-first, local document intelligence assistant. Built with React 19, TypeScript, Vite, and PDF.js / React-PDF.

## Features

- **Document Management Sidebar**: Upload PDF documents, inspect page/chunk counts, and delete documents with instant local storage cleanup.
- **Interactive PDF Viewer**: Embedded PDF rendering powered by `@react-pdf-viewer/core` with dynamic bounding-box citation highlights.
- **Evidence-Grounded Q&A**: Question interface with 3-state evidence representation:
  - **Supported Answer**: Displays answer grounded in citations.
  - **Requirement Not Established**: Clarifies when retrieved evidence discusses the topic but does not establish mandatory criteria.
  - **Evidence Not Found**: Clear indication when no relevant passage exists.
- **Deep Citation Linking**: Clickable citations jump directly to the target page and highlight the exact text bounding box.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend runs at `http://localhost:5173` and proxies API requests to `http://localhost:8000`.

## Production Build

```bash
npm run build
```

The production bundle is generated in `dist/`.
