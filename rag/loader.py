# rag/loader.py
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .retrieval import embed_texts, upsert_vectors
from .llm import detect_domain_llm
from .models import UploadedDocument, DocumentChunk


# ---------------------- File text extraction ----------------------


def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text) from a PDF.
    Tries pdfplumber, falls back to pypdf.
    """
    pages: List[Tuple[int, str]] = []

    # Try pdfplumber first
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                pages.append((i + 1, txt))
        if pages:
            return pages
    except Exception as e:
        print("[PDF] pdfplumber failed → fallback to pypdf:", e)

    # Fallback: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages.append((i + 1, txt))
    except Exception as e:
        print("[PDF] pypdf failed:", e)

    return pages


def extract_text_from_docx(path: str) -> List[Tuple[int, str]]:
    try:
        import docx  # python-docx
    except Exception:
        raise RuntimeError("python-docx is required to parse .docx files")

    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [(1, text)]


def extract_text_from_txt(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]


def extract_text(path: str) -> List[Tuple[int, str]]:
    """
    Detect file type by extension and use appropriate parser.
    Returns: list of (page_num, text).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext in [".txt", ".md"]:
        return extract_text_from_txt(path)

    # Fallback: try pdf, then txt
    try:
        return extract_text_from_pdf(path)
    except Exception:
        return extract_text_from_txt(path)


# ---------------------- Section detection & chunking ----------------------


def detect_sections(page_text: str) -> List[Tuple[str, str]]:
    """
    Simple heuristic: detect headings & split sections.
    Returns list of (section_title, section_text).
    """
    lines = page_text.splitlines()
    headings = []

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        # Heuristic: ALL CAPS, numbered headings, or "Section"/"Chapter"
        if s.isupper() and len(s) > 3:
            headings.append((i, s))
        elif re.match(r"^\d+[\.\)]\s+\S+", s):
            headings.append((i, s))
        elif s.lower().startswith("section") or s.lower().startswith("chapter"):
            headings.append((i, s))

    if not headings:
        return [("body", page_text)]

    sections: List[Tuple[str, str]] = []
    for idx, (line_idx, title) in enumerate(headings):
        start = line_idx
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        sec_text = "\n".join(lines[start:end]).strip()
        sections.append((title, sec_text))
    return sections


def chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """
    Char-based chunking with overlap. Good enough for many use cases.
    """
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_chars:
        return [clean]

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = start + chunk_chars
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = end - overlap_chars
    return chunks


# ---------------------- Main indexing function ----------------------


def index_document(
    file_path: str,
    user_id: Optional[int] = None,
    original_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High level:
    - Extract text pages from file
    - Use LLM to detect free-text domain (no fixed list)
    - Detect sections per page
    - Chunk sections
    - Embed chunks (HF Inference)
    - Upsert to Pinecone with metadata

    Returns:
    {
      "document_id": str,
      "chunks_indexed": int,
      "domain": str
    }
    """
    pages = extract_text(file_path)
    if not pages:
        return {"document_id": None, "chunks_indexed": 0, "domain": "general"}

    full_text = "\n".join(t for _, t in pages)
    domain_label = detect_domain_llm(full_text)

    # Persist UploadedDocument
    doc = UploadedDocument.objects.create(
        user_id=user_id,
        title=original_name or Path(file_path).name,
        original_filename=original_name or Path(file_path).name,
        domain=domain_label,
    )

    vectors_to_upsert: List[Dict[str, Any]] = []

    for page_num, page_text in pages:
        sections = detect_sections(page_text)
        for sidx, (sec_title, sec_text) in enumerate(sections):
            chunks = chunk_text(sec_text)
            for cidx, ctext in enumerate(chunks):
                chunk_uuid = str(uuid.uuid4())
                snippet = ctext[:400]

                metadata = {
                    "user_id": str(user_id) if user_id is not None else None,
                    "document_id": str(doc.id),
                    "domain": domain_label,
                    "page_num": page_num,
                    "section_idx": sidx,
                    "section_title": sec_title,
                    "chunk_idx": cidx,
                    "original_name": original_name or Path(file_path).name,
                    "snippet": snippet,
                    "chunk_text": ctext,  # full chunk for retrieval context
                }

                vectors_to_upsert.append(
                    {
                        "id": chunk_uuid,
                        "text": ctext,
                        "metadata": metadata,
                    }
                )

                DocumentChunk.objects.create(
                    id=chunk_uuid,
                    document=doc,
                    chunk_idx=cidx,
                    section_idx=sidx,
                    section_title=sec_title,
                    page_num=page_num,
                    snippet=snippet,
                    pinecone_id=chunk_uuid,
                )

    if not vectors_to_upsert:
        return {"document_id": str(doc.id), "chunks_indexed": 0, "domain": domain_label}

    texts = [v["text"] for v in vectors_to_upsert]
    embeddings = embed_texts(texts)

    upsert_items = []
    for item, emb in zip(vectors_to_upsert, embeddings):
        upsert_items.append(
            {
                "id": item["id"],
                "values": emb,
                "metadata": item["metadata"],
            }
        )

    upsert_vectors(upsert_items)

    return {
        "document_id": str(doc.id),
        "chunks_indexed": len(upsert_items),
        "domain": domain_label,
    }
