# rag/loader.py
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .retrieval import embed_texts, upsert_vectors
from .llm import detect_domain_llm
from .models import UploadedDocument, DocumentChunk

# -------------------------- TEXT EXTRACTION ---------------------------

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, pg in enumerate(pdf.pages):
                try:
                    txt = pg.extract_text() or ""
                except Exception:
                    txt = ""
                pages.append((i + 1, txt))
        if any(p[1].strip() for p in pages):
            print("[PDF] Extracted via pdfplumber")
            return pages
    except Exception as e:
        print("[PDF] pdfplumber failed:", e)

    # 2) pypdf fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append((i + 1, txt))
        if any(p[1].strip() for p in pages):
            print("[PDF] Extracted via pypdf")
            return pages
    except Exception as e:
        print("[PDF] pypdf failed:", e)

    # 3) OCR fallback
    try:
        from pdf2image import convert_from_path
        import pytesseract
        print("[PDF] OCR fallback enabled")
        pages = []
        images = convert_from_path(path, dpi=300)
        for i, img in enumerate(images):
            txt = pytesseract.image_to_string(img)
            pages.append((i + 1, txt))
        return pages
    except Exception as e:
        print("[PDF] OCR failed:", e)

    return pages


def extract_text_from_docx(path: str) -> List[Tuple[int, str]]:
    try:
        import docx
        doc = docx.Document(path)
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    except Exception:
        return []


def extract_text_from_txt(path: str) -> List[Tuple[int, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]
    except Exception:
        return []


def extract_text(path: str) -> List[Tuple[int, str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext in [".txt", ".md"]:
        return extract_text_from_txt(path)
    # fallback
    return extract_text_from_pdf(path)

# -------------------------- SECTION DETECTION -------------------------

def detect_sections(page_text: str):
    lines = page_text.splitlines()
    headings = []
    sections = []

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        # uppercase headings or numbered headings
        if s.isupper() and len(s) > 3:
            headings.append((i, s))
        elif re.match(r"^\d+[\.\)]\s+\S+", s):
            headings.append((i, s))

    # if no headings, return one section with whole page
    if not headings:
        return [("body", page_text.strip())]

    for idx, (line_idx, title) in enumerate(headings):
        start = line_idx
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        sec = "\n".join(lines[start:end]).strip()
        sections.append((title or "section", sec))
    return sections

# -------------------------- CHUNKING -------------------------------

def chunk_text(text: str, size: int = 500, overlap: int = 100):
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]
    res = []
    start = 0
    while start < len(clean):
        end = start + size
        res.append(clean[start:end])
        if end >= len(clean):
            break
        start = end - overlap
    return res

# -------------------------- DOMAIN NORMALIZE -----------------------

def normalize_domain(raw: Optional[str]) -> str:
    if not raw:
        return "general"
    r = re.sub(r"[^a-z0-9 ]+", "", raw.lower()).strip().replace(" ", "_")
    if r.isdigit() or len(r) < 3:
        return "general"
    return r[:50]

# -------------------------- INDEX PIPELINE -------------------------

def index_document(file_path: str, user_id: int, original_name: str) -> Dict[str, Any]:
    print(f"[index_document] Processing → {file_path}")
    pages = extract_text(file_path)
    print(f"[index_document] Pages extracted = {len(pages)}")

    if not pages:
        print("[index_document] No text extracted — aborting")
        return {"document_id": None, "chunks_indexed": 0, "domain": "general"}

    raw_text = "\n".join(p[1] for p in pages)
    try:
        raw_domain = detect_domain_llm(raw_text)
    except Exception:
        raw_domain = "general"
    domain = normalize_domain(raw_domain)
    print(f"[index_document] Domain → {domain}")

    doc = UploadedDocument.objects.create(
        user_id=user_id,
        original_filename=original_name,
        domain=domain,
    )
    print(f"[index_document] Document ID = {doc.id}")

    vectors = []
    global_idx = 0
    for page_num, page_text in pages:
        sections = detect_sections(page_text)
        for s_idx, (title, sec_text) in enumerate(sections):
            chunks = chunk_text(sec_text)
            for c in chunks:
                cid = str(uuid.uuid4())
                snippet = c[:300]
                metadata = {
                    "user_id": str(user_id),
                    "document_id": str(doc.id),
                    "page_num": page_num,
                    "section_idx": s_idx,
                    "section_title": title,
                    "chunk_idx": global_idx,
                    "snippet": snippet,
                    "chunk_text": c,
                    "domain": domain,
                }
                vectors.append({"id": cid, "text": c, "metadata": metadata})

                DocumentChunk.objects.create(
                    id=cid,
                    document=doc,
                    chunk_idx=global_idx,
                    section_idx=s_idx,
                    section_title=title,
                    page_num=page_num,
                    pinecone_id=cid,
                )
                global_idx += 1

    print(f"[index_document] Total Chunks = {global_idx}")

    try:
        texts = [v["text"] for v in vectors]
        embs = embed_texts(texts)
        upsert_vectors([{"id": v["id"], "values": e, "metadata": v["metadata"]} for v, e in zip(vectors, embs)])
        return {"document_id": str(doc.id), "chunks_indexed": global_idx, "domain": domain}
    except Exception as e:
        print("❌ Embedding/upsert failed:", e)
        return {"document_id": str(doc.id), "chunks_indexed": 0, "domain": domain}
