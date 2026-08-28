"""
agents/ingestion_agent.py
──────────────────────────
IngestionAgent — multi-format file parsing, chunking, and embedding.

Supported formats:
  • .pdf   — via PyPDF2
  • .md    — plain text parse
  • .txt   — plain text
  • .docx  — via python-docx
  • .xlsx  — via openpyxl  (each row → one chunk)
  • .csv   — via csv stdlib

"Books" collection:
  Files saved to ./data/books/ are indexed into a separate ChromaDB
  collection ("acd_books") so that the KnowledgeAgent can prioritise them.

LangGraph node: processes a list of (filename, bytes) pairs from state.
"""

from __future__ import annotations
import csv
import io
import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document


# ── Collection names ──────────────────────────────────────────────────────────
COLLECTION_NOTES = "acd_mentor"   # lecture notes / user uploads
COLLECTION_BOOKS = "acd_books"    # textbooks (separate collection)

BOOKS_DIR = Path("./data/books")
DOCS_DIR  = Path("./data/uploaded_docs")


# ── File parsers ──────────────────────────────────────────────────────────────

def _parse_pdf(data: bytes, filename: str) -> List[Document]:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1, "type": "pdf"},
                ))
        return docs
    except Exception as exc:
        print(f"[IngestionAgent] PDF parse error {filename}: {exc}")
        return []


def _parse_docx(data: bytes, filename: str) -> List[Document]:
    try:
        from docx import Document as DocxDocument  # python-docx
        doc = DocxDocument(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return [Document(page_content=text, metadata={"source": filename, "type": "docx"})]
    except Exception as exc:
        print(f"[IngestionAgent] DOCX parse error {filename}: {exc}")
        return []


def _parse_xlsx(data: bytes, filename: str) -> List[Document]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        docs = []
        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header = [str(c) if c is not None else "" for c in rows[0]]
            for row_idx, row in enumerate(rows[1:], 2):
                cells = [str(c) if c is not None else "" for c in row]
                content = "; ".join(f"{h}: {v}" for h, v in zip(header, cells) if v)
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": filename, "sheet": sheet.title, "row": row_idx, "type": "xlsx"},
                    ))
        return docs
    except Exception as exc:
        print(f"[IngestionAgent] XLSX parse error {filename}: {exc}")
        return []


def _parse_csv(data: bytes, filename: str) -> List[Document]:
    try:
        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        docs = []
        for i, row in enumerate(reader, 1):
            content = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename, "row": i, "type": "csv"},
                ))
        return docs
    except Exception as exc:
        print(f"[IngestionAgent] CSV parse error {filename}: {exc}")
        return []


def _parse_md(data: bytes, filename: str) -> List[Document]:
    text = data.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": filename, "type": "md"})]


def _parse_text(data: bytes, filename: str) -> List[Document]:
    text = data.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": filename, "type": "text"})]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def parse_file(filename: str, data: bytes) -> List[Document]:
    """Route file to the correct parser based on extension."""
    ext = Path(filename).suffix.lower()
    parsers = {
        ".pdf":  _parse_pdf,
        ".docx": _parse_docx,
        ".xlsx": _parse_xlsx,
        ".csv":  _parse_csv,
        ".md":   _parse_md,
        ".txt":  _parse_text,
    }
    parser = parsers.get(ext, _parse_text)
    return parser(data, filename)


# ── Index to ChromaDB ─────────────────────────────────────────────────────────

def _index_docs(docs: List[Document], collection: str) -> int:
    """Chunk and embed documents into specified ChromaDB collection."""
    if not docs:
        return 0
    try:
        from rag.vector_store import build_or_load, add_documents
        build_or_load(collection_name=collection)
        return add_documents(docs)
    except Exception as exc:
        print(f"[IngestionAgent] Indexing error into {collection}: {exc}")
        return 0


def ingest_file(filename: str, data: bytes, is_book: bool = False) -> Tuple[int, str]:
    """
    Parse and index a single file.

    Parameters
    ----------
    filename : str   — original filename (used as source metadata)
    data     : bytes — raw file content
    is_book  : bool  — True → store in 'acd_books' collection

    Returns (chunks_indexed, error_message)
    """
    try:
        docs = parse_file(filename, data)
        if not docs:
            return 0, f"No content extracted from {filename}"

        collection = COLLECTION_BOOKS if is_book else COLLECTION_NOTES
        count = _index_docs(docs, collection)

        # Also save bytes to disk for future reference
        target_dir = BOOKS_DIR if is_book else DOCS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / filename).write_bytes(data)

        return count, ""
    except Exception as exc:
        return 0, str(exc)


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node for IngestionAgent.

    Reads state["pending_files"] = list of {"filename": str, "data": bytes, "is_book": bool}
    Produces state["ingestion_results"] = list of {"filename", "chunks", "error"}
    """
    pending = state.get("pending_files", [])
    results = []
    total_chunks = 0

    for item in pending:
        fname   = item.get("filename", "unknown")
        data    = item.get("data", b"")
        is_book = item.get("is_book", False)

        chunks, err = ingest_file(fname, data, is_book=is_book)
        results.append({"filename": fname, "chunks": chunks, "error": err})
        total_chunks += chunks

    state["ingestion_results"] = results
    state["total_chunks_indexed"] = total_chunks
    return state
