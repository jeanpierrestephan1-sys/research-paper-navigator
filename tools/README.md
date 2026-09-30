# tools

Scripts that turn the course's PDFs into what the checkpoints read. All three use
Apple PDFKit and need macOS. Usage is in the main README under Corpus.

- `extract_all.swift` — one `.txt` per PDF, with a `PAGE n` line before each page,
  so every chunk can cite a page. Skips files already extracted.
- `index.swift` — one line per PDF, filename then title, read from the first page.
  Its output for the course corpus is saved here as `paper-index.tsv`.
- `pdftext.swift` — prints one PDF's text, optionally the first N pages only. Used
  for spot checks.
