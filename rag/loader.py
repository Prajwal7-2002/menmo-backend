# rag/loader.py
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .retrieval import embed_texts, upsert_vectors
from .llm import detect_domain_llm
from .models import UploadedDocument, DocumentChunk


# ============================================================
#  TEXT EXTRACTION HELPERS
# ============================================================

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """Extract text page-wise from PDF using pdfplumber → pypdf fallback."""
    pages: List[Tuple[int, str]] = []

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages.append((i + 1, page.extract_text() or ""))
        if pages:
            return pages
    except Exception as e:
        print("[PDF] pdfplumber failed → pypdf fallback:", e)

    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            pages.append((i + 1, page.extract_text() or ""))
    except Exception as e:
        print("[PDF] pypdf failed:", e)

    return pages


def extract_text_from_docx(path: str) -> List[Tuple[int, str]]:
    import docx
    doc = docx.Document(path)
    return [(1, "\n".join(p.text for p in doc.paragraphs))]


def extract_text_from_txt(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]


def extract_text(path: str) -> List[Tuple[int, str]]:
    """Auto-detect extension and extract text."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext in [".txt", ".md"]:
        return extract_text_from_txt(path)

    # fallback
    try:
        return extract_text_from_pdf(path)
    except:
        return extract_text_from_txt(path)


# ============================================================
#  SECTION + CHUNKING
# ============================================================

def detect_sections(page_text: str) -> List[Tuple[str, str]]:
    """Split text by headings."""
    lines = page_text.splitlines()
    headings = []

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        # Heading heuristics
        if s.isupper() and len(s) > 3:
            headings.append((i, s))
        elif re.match(r"^\d+[\.\)]\s+\S+", s):
            headings.append((i, s))
        elif s.lower().startswith(("section", "chapter")):
            headings.append((i, s))

    if not headings:
        return [("body", page_text)]

    sections = []
    for idx, (line_idx, title) in enumerate(headings):
        start = line_idx
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        sec_text = "\n".join(lines[start:end]).strip()
        sections.append((title, sec_text))

    return sections


def chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """Chunk text with overlap."""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= chunk_chars:
        return [clean] if clean else []

    chunks = []
    start = 0
    while start < len(clean):
        end = start + chunk_chars
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = end - overlap_chars
    return chunks


# ============================================================
#  CLEAN DOMAIN DETECTION
# ============================================================

def normalize_domain(raw: str) -> str:
    """
    Clean and normalize domain names.
    Converts fallback messages into 'general'.
    """
    if not raw:
        return "general"

    r = raw.strip().lower()

    # bad / fallback domains
    bad_patterns = [
        "i don't know",
        "i_don’t_know",
        "i_dont_know",
        "please_rephrase",
        "available_documentation",
    ]

    if any(bad in r for bad in bad_patterns):
        return "general"

    # only allow alphabets, numbers, hyphens
    r = re.sub(r"[^a-z0-9\- ]+", "", r)
    r = r.replace(" ", "_").strip("_")

    if not r:
        return "general"

    return r[:50]   # prevent overly long labels


# ============================================================
#  MAIN INDEX FUNCTION
# ============================================================

def index_document(
    file_path: str,
    user_id: Optional[int] = None,
    original_name: Optional[str] = None,
) -> Dict[str, Any]:

    pages = extract_text(file_path)
    if not pages:
        return {"document_id": None, "chunks_indexed": 0, "domain": "general"}

    full_text = "\n".join(t for _, t in pages)

    # ------------------------------
    # DOMAIN FIX APPLIED HERE
    # ------------------------------
    raw_domain = detect_domain_llm(full_text)
    domain_label = normalize_domain(raw_domain)

    # Save document
    doc = UploadedDocument.objects.create(
        user_id=user_id,
        title=original_name or Path(file_path).name,
        original_filename=original_name or Path(file_path).name,
        domain=domain_label,
    )

    vectors_to_upsert = []

    # process each page → section → chunk
    for page_num, page_text in pages:
        sections = detect_sections(page_text)

        for sidx, (sec_title, sec_text) in enumerate(sections):
            chunks = chunk_text(sec_text)

            for cidx, ctext in enumerate(chunks):
                chunk_uuid = str(uuid.uuid4())
                snippet = ctext[:400]

                metadata = {
                    "user_id": str(user_id),
                    "document_id": str(doc.id),
                    "domain": domain_label,
                    "page_num": page_num,
                    "section_idx": sidx,
                    "section_title": sec_title,
                    "chunk_idx": cidx,
                    "original_name": original_name or Path(file_path).name,
                    "snippet": snippet,
                    "chunk_text": ctext,
                }

                vectors_to_upsert.append(
                    {"id": chunk_uuid, "text": ctext, "metadata": metadata}
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

    # embed + upsert
    texts = [v["text"] for v in vectors_to_upsert]
    embeddings = embed_texts(texts)

    upsert_items = [
        {"id": item["id"], "values": emb, "metadata": item["metadata"]}
        for item, emb in zip(vectors_to_upsert, embeddings)
    ]

    upsert_vectors(upsert_items)

    return {
        "document_id": str(doc.id),
        "chunks_indexed": len(upsert_items),
        "domain": domain_label,
    }
