import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .retrieval import embed_texts, upsert_vectors
from .llm import detect_domain_llm
from .models import UploadedDocument, DocumentChunk

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                pages.append((i + 1, txt))
        if any(p[1].strip() for p in pages):
            print("[PDF] Extracted via pdfplumber")
            return pages
    except Exception as e:
        print("[PDF] pdfplumber failed:", e)

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

    try:
        from pdf2image import convert_from_path
        import pytesseract
        print("[PDF] OCR fallback enabled")
        pages = []
        for i, img in enumerate(convert_from_path(path, dpi=300)):
            text = pytesseract.image_to_string(img)
            pages.append((i + 1, text))
        return pages
    except Exception as e:
        print("[PDF] OCR failed:", e)

    return []

def extract_text_from_docx(path: str) -> List[Tuple[int, str]]:
    import docx
    doc = docx.Document(path)
    return [(1, "\n".join(p.text for p in doc.paragraphs))]

def extract_text_from_txt(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]

def extract_text(path: str) -> List[Tuple[int, str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf": return extract_text_from_pdf(path)
    if ext == ".docx": return extract_text_from_docx(path)
    if ext in [".txt", ".md"]: return extract_text_from_txt(path)
    return extract_text_from_pdf(path)

def detect_sections(page_text: str):
    lines = page_text.splitlines()
    headings = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s: continue
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
        sections.append((title, "\n".join(lines[start:end]).strip()))
    return sections

def chunk_text(text: str, size: int = 500, overlap: int = 100):
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]
    chunks = []
    start = 0
    while start < len(clean):
        end = start + size
        chunks.append(clean[start:end])
        if end >= len(clean): break
        start = end - overlap
    return chunks

def normalize_domain(raw: str) -> str:
    if not raw: return "general"
    r = re.sub(r"[^a-z0-9 ]+", "", raw.lower()).strip().replace(" ", "_")
    if r.isdigit() or len(r) < 3: return "general"
    return r[:50]

def index_document(file_path: str, user_id: int, original_name: str) -> Dict[str, Any]:
    print(f"\n[index_document] Processing → {file_path}")
    pages = extract_text(file_path)
    print(f"[index_document] Pages extracted = {len(pages)}")
    if not pages:
        return {"document_id": None, "chunks_indexed": 0, "domain": "general"}

    raw_text = "\n".join(p[1] for p in pages)
    # In index_document (patch)
    try:
        raw_domain = detect_domain_llm(raw_text)
    except Exception:
        raw_domain = "general"
    domain = normalize_domain(raw_domain).lower()
    print(f"[index_document] Domain → {domain}")

    doc = UploadedDocument.objects.create(user_id=user_id, title=original_name, original_filename=original_name, domain=domain)


    vectors = []
    global_idx = 0

    for page_num, page_text in pages:
        for s_idx, (title, sec_text) in enumerate(detect_sections(page_text)):
            for chunk in chunk_text(sec_text):
                cid = str(uuid.uuid4())
                snippet = chunk[:300]
                vectors.append({
                    "id": cid,
                    "text": chunk,
                    "metadata": {
                        "user_id": str(user_id),
                        "document_id": str(doc.id),
                        "domain": domain,
                        "page_num": page_num,
                        "section_idx": s_idx,
                        "section_title": title,
                        "chunk_idx": global_idx,
                        "snippet": snippet,
                        "chunk_text": chunk,
                    }
                })
                DocumentChunk.objects.create(id=cid, document=doc, chunk_idx=global_idx, section_idx=s_idx, section_title=title, page_num=page_num, snippet=snippet, pinecone_id=cid)
                global_idx += 1

    print(f"[index_document] Total Chunks = {global_idx}")

    try:
        txts = [v["text"] for v in vectors]
        embs = embed_texts(txts)
        upsert_vectors([{"id": v["id"], "values": e, "metadata": v["metadata"]} for v, e in zip(vectors, embs)])
        return {"document_id": str(doc.id), "chunks_indexed": global_idx, "domain": domain}
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return {"document_id": str(doc.id), "chunks_indexed": 0, "domain": domain}
