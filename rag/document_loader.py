"""
rag/document_loader.py
───────────────────────
Loads and extracts text from:
  • PDF  files (via PyPDF2)
  • PPT  files (via python-pptx)
  • Plain text / markdown files

Returns a list of LangChain Document objects.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_core.documents import Document


# ── PDF ───────────────────────────────────────────────────────────────────────
def _load_pdf(file_path: Path) -> List[Document]:
    try:
        import PyPDF2
        docs: List[Document] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path.name, "page": i + 1, "type": "pdf"}
                    ))
        return docs
    except Exception as exc:
        print(f"[PDF Loader] Error reading {file_path}: {exc}")
        return []


# ── PPTX ──────────────────────────────────────────────────────────────────────
def _load_pptx(file_path: Path) -> List[Document]:
    try:
        from pptx import Presentation
        prs = Presentation(str(file_path))
        docs: List[Document] = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            combined = "\n".join(texts)
            if combined:
                docs.append(Document(
                    page_content=combined,
                    metadata={"source": file_path.name, "slide": slide_num, "type": "pptx"}
                ))
        return docs
    except Exception as exc:
        print(f"[PPTX Loader] Error reading {file_path}: {exc}")
        return []


# ── Plain text ────────────────────────────────────────────────────────────────
def _load_text(file_path: Path) -> List[Document]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [Document(
            page_content=text,
            metadata={"source": file_path.name, "type": "text"}
        )]
    except Exception as exc:
        print(f"[Text Loader] Error reading {file_path}: {exc}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────
def load_uploaded_file(file_path: str | Path) -> List[Document]:
    """
    Auto-detect file type and return a list of LangChain Documents.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".pptx", ".ppt"):
        return _load_pptx(path)
    else:
        return _load_text(path)


def load_raw_text_as_docs(text: str, source: str = "external") -> List[Document]:
    """Wrap a raw string into Document objects (used for GFG articles)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": source})
        for chunk in chunks
    ]
