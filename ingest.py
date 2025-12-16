# ingest.py
from __future__ import annotations
import os, glob, uuid, io, sys, re, json, hashlib, math, time
from typing import List, Dict, Iterable, Optional
from collections import deque
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import HttpResponseError

# OpenAI
from openai import OpenAI

# Fallback extractor
from PyPDF2 import PdfReader

load_dotenv()

# ------------------- Azure Search / OpenAI -------------------

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
AZURE_SEARCH_KEY      = os.environ["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX", "hr-policies")

DOCS_DIR              = os.getenv("DOCS_DIR", "./docs")
EMBED_MODEL           = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_EMBED           = int(os.getenv("EMBED_BATCH", "64"))

# Optional: document type for this ingest run: auto|policy|daq
DOC_TYPE              = os.getenv("DOC_TYPE", "auto").lower()

# Chunking knobs
CHUNK_MAX_CHARS       = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
CHUNK_OVERLAP_CHARS   = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
HEADING_CONTEXT_DEPTH = int(os.getenv("HEADING_CONTEXT_DEPTH", "2"))

# Azure Document Intelligence (DI)
AZURE_DI_ENDPOINT     = os.getenv("AZURE_DI_ENDPOINT") or os.getenv("AZURE_FORMRECOGNIZER_ENDPOINT") or os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
AZURE_DI_KEY          = os.getenv("AZURE_DI_KEY") or os.getenv("AZURE_FORMRECOGNIZER_KEY") or os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY")
AZURE_DI_MODEL        = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")

# Safer Azure Search batch limits (payload + count)
MAX_BATCH_DOCS        = int(os.getenv("MAX_BATCH_DOCS", "500"))
MAX_BATCH_BYTES       = int(os.getenv("MAX_BATCH_BYTES", str(8 * 1024 * 1024)))  # ~8MB soft cap
RETRY_BACKOFF_SEC     = float(os.getenv("RETRY_BACKOFF_SEC", "1.0"))
DETERMINISTIC_IDS     = os.getenv("DETERMINISTIC_IDS", "true").lower() == "true"

search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY))
oai           = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------------- Utilities -------------------

_ws            = re.compile(r"\s+")
_bullet_like   = re.compile(r'^\s*(?:[-–•●▪○*]|(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdmIVXLCDM]+)[\.\)\]])\s+')
_toc_line      = re.compile(r'.*?\.{2,}\s*\d+$')
_q_start       = re.compile(r'^(?:Q[:\-]\s*)?(What|How|Can|Do|Does|Is|Are|When|Where|Who|Whom|Why|Which|Should|Could|Would|May|Will)\b.*\?$', re.IGNORECASE)

def _norm(s: str) -> str:
    return _ws.sub(" ", (s or "").strip())

def _is_heading_role(role: Optional[str]) -> bool:
    return role in {"title", "sectionHeading", "heading"}

def _is_noise_role(role: Optional[str]) -> bool:
    return role in {"pageHeader", "pageFooter", "footnote"}

def _starts_with_number_token(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    head = t.split(" ")[0].rstrip(".").rstrip(")").rstrip("]")
    return (
        head.isdigit() or
        re.fullmatch(r"[A-Za-z]", head) is not None or
        re.fullmatch(r"[ivxlcdmIVXLCDM]+", head) is not None or
        re.fullmatch(r"\d+(?:\.\d+)+", head) is not None
    )

def _is_toc_page(lines: list[str]) -> bool:
    if not lines:
        return False
    matches = sum(1 for t in lines if _toc_line.match(t))
    return matches >= 5 or (matches / max(1, len(lines))) > 0.4

def _is_question_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith("?"):
        return True
    if s.lower().startswith("q:"):
        return True
    return _q_start.match(s) is not None

def _split_with_overlap_for_embed(text: str, max_chars: int, overlap: int) -> List[Dict]:
    """
    Returns a list of {"content": clean_segment, "embed": segment_with_overlap}
    Only embed carries the overlap. Stored content remains clean (no duplication).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [{"content": text, "embed": text}]

    # sentence-ish segmentation
    sents = re.split(r'(?<=[\.\!\?])\s+(?=[A-Z(])', text)
    segs: List[str] = []
    buf = ""
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                segs.append(buf)
            buf = s
    if buf:
        segs.append(buf)

    # hard-wrap any too-long segment
    final: List[str] = []
    for seg in segs:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            for i in range(0, len(seg), max_chars):
                final.append(seg[i:i+max_chars])

    out: List[Dict] = []
    for i, seg in enumerate(final):
        if i == 0 or overlap <= 0:
            out.append({"content": seg, "embed": seg})
        else:
            prev_tail = final[i-1][-overlap:] if len(final[i-1]) > overlap else final[i-1]
            out.append({"content": seg, "embed": (prev_tail + " " + seg).strip()})
    return out

# ------------------- Azure DI client -------------------

def _get_di_client():
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return None, None
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
        client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
        return client, "documentintelligence"
    except Exception:
        pass
    try:
        from azure.ai.documentanalysis import DocumentAnalysisClient  # type: ignore
        client = DocumentAnalysisClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
        return client, "documentanalysis"
    except Exception:
        pass
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient  # type: ignore
        client = DocumentAnalysisClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
        return client, "formrecognizer"
    except Exception:
        pass
    return None, None

# ------------------- DI -> normalized elements -------------------

def di_extract_elements(pdf_bytes: bytes) -> List[Dict]:
    di_client, _ = _get_di_client()
    if not di_client:
        raise RuntimeError("Azure Document Intelligence client not configured or not installed.")

    poller = di_client.begin_analyze_document(AZURE_DI_MODEL, document=pdf_bytes)
    result = poller.result()

    paragraphs = getattr(result, "paragraphs", None)
    if paragraphs is None:
        raise RuntimeError("DI result lacks 'paragraphs' (older SDK); falling back.")

    elements: List[Dict] = []
    page_texts: Dict[int, List[str]] = {}

    for p in paragraphs:
        role = getattr(p, "role", None)
        if _is_noise_role(role):
            continue
        txt = _norm(getattr(p, "content", "")) or ""
        if not txt:
            continue
        page = 1
        regions = getattr(p, "bounding_regions", None)
        if regions:
            page = int(regions[0].page_number or 1)
        page_texts.setdefault(page, []).append(txt)
        kind = "header" if _is_heading_role(role) else "paragraph"
        if kind == "paragraph" and _bullet_like.match(txt):
            txt = _norm(_bullet_like.sub("", txt))
            kind = "list_item"
        elements.append({"type": kind, "content": txt, "page_number": page})

    for t in getattr(result, "tables", []) or []:
        page = 1
        regions = getattr(t, "bounding_regions", None)
        if regions:
            page = int(regions[0].page_number or 1)
        nrows, ncols = t.row_count, t.column_count
        grid = [["" for _ in range(ncols)] for __ in range(nrows)]
        for c in t.cells:
            grid[c.row_index][c.column_index] = _norm(getattr(c, "content", ""))
        if nrows:
            md = "| " + " | ".join(grid[0]) + " |\n"
            md += "| " + " | ".join("---" for _ in range(ncols)) + " |\n"
            for r in range(1, nrows):
                md += "| " + " | ".join(grid[r]) + " |\n"
            md = md.strip()
            page_texts.setdefault(page, []).append("[[TABLE]]")
            elements.append({"type": "table", "content": md, "page_number": page})

    toc_pages = {pg for pg, lines in page_texts.items() if _is_toc_page(lines)}
    elements = [e for e in elements if e["page_number"] not in toc_pages]
    elements.sort(key=lambda e: (e["page_number"]))
    return elements

# ------------------- Strategy A: Policy chunker -------------------

def chunk_policy(elements: List[Dict], max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[Dict]:
    chunks: List[Dict] = []
    current_lines: List[str] = []
    current_pages: set[int] = set()
    heading_ctx: deque[str] = deque(maxlen=HEADING_CONTEXT_DEPTH)

    def flush():
        nonlocal current_lines, current_pages
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        prefix = "\n".join(h for h in heading_ctx if h).strip()
        full = f"{prefix}\n{body}".strip() if prefix else body
        for seg in _split_with_overlap_for_embed(full, max_chars, overlap):
            chunks.append({
                "content": seg["content"],
                "embed_text": seg["embed"],
                "page": min(current_pages) if current_pages else 1
            })
        current_lines = []
        current_pages = set()

    i = 0
    while i < len(elements):
        e = elements[i]
        t = e["type"]; c = e["content"]; p = e["page_number"]

        if t == "header":
            flush()
            heading_ctx.append(c)
            i += 1
            continue

        starts_numbered = _starts_with_number_token(c)
        if starts_numbered and current_lines:
            flush()

        if t == "list_item":
            bullets = [c]
            j = i + 1
            while j < len(elements) and elements[j]["type"] == "list_item" and elements[j]["page_number"] == p:
                bullets.append(elements[j]["content"])
                j += 1
            block = "\n".join(f"• {b}" for b in bullets)
            current_lines.append(block)
            current_pages.add(p)
            i = j
            continue

        current_lines.append(c)
        current_pages.add(p)
        i += 1

    flush()
    return chunks

# ------------------- Strategy B: DAQ/FAQ chunker -------------------

def chunk_daq(elements: List[Dict], max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS, max_gap=8) -> List[Dict]:
    chunks: List[Dict] = []
    q: Optional[str] = None
    ans_lines: List[str] = []
    pages: set[int] = set()
    gap = 0

    def flush():
        nonlocal q, ans_lines, pages, gap
        if q and ans_lines:
            full = f"Q: {q}\nA: {' '.join(ans_lines).strip()}"
            for seg in _split_with_overlap_for_embed(full, max_chars, overlap):
                chunks.append({
                    "content": seg["content"],
                    "embed_text": seg["embed"],
                    "page": min(pages) if pages else 1
                })
        q, ans_lines, pages, gap = None, [], set(), 0

    for e in elements:
        line = e["content"].strip()
        if not line:
            gap += 1
            if gap > max_gap:
                q, ans_lines, pages, gap = None, [], set(), 0
            continue

        if line.lower().startswith("a:"):
            ans_lines.append(line[2:].strip())
            pages.add(e["page_number"])
            gap = 0
            continue

        if _is_question_line(line):
            if q and ans_lines:
                flush()
            q = re.sub(r'^(?:Q[:\-]\s*)', "", line, flags=re.IGNORECASE).strip()
            ans_lines = []
            pages = {e["page_number"]}
            gap = 0
        else:
            if q:
                ans_lines.append(line)
                pages.add(e["page_number"])
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    q, ans_lines, pages, gap = None, [], set(), 0

    if q and ans_lines:
        flush()

    if not chunks and elements:
        all_text = "\n".join(e["content"] for e in elements if e["content"].strip())
        for seg in _split_with_overlap_for_embed(all_text, max_chars, overlap):
            chunks.append({
                "content": seg["content"],
                "embed_text": seg["embed"],
                "page": min(e["page_number"] for e in elements)
            })

    return chunks

# ------------------- Auto routing -------------------

def infer_doc_type(elements: List[Dict]) -> str:
    if not elements:
        return "policy"
    lines = [e["content"] for e in elements]
    q_like = sum(1 for l in lines if _is_question_line(l))
    return "daq" if (q_like >= 3 and q_like / max(1, len(lines)) >= 0.12) else "policy"

# ------------------- Public parse entry -------------------

def parse_pdf_into_chunks(pdf_bytes: bytes, filename: str, doc_type: str = "auto") -> List[Dict]:
    """
    Returns: [{"page": int, "content": str, "embed_text": str, "source": filename}, ...]
    """
    try:
        elements = di_extract_elements(pdf_bytes)
    except Exception:
        return parse_with_pypdf2(pdf_bytes, filename)

    kind = (doc_type or "auto").lower()
    if kind not in {"policy", "daq"}:
        kind = infer_doc_type(elements)

    grouped = chunk_daq(elements) if kind == "daq" else chunk_policy(elements)

    for g in grouped:
        g["source"] = filename
        g["page"]   = g.get("page", 1)
    return grouped

# ------------------- Fallback parser (PyPDF2) -------------------

def parse_with_pypdf2(pdf_bytes: bytes, filename: str) -> List[Dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[Dict] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _norm(text)
        if not text:
            continue
        for seg in _split_with_overlap_for_embed(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            chunks.append({
                "content": seg["content"],
                "embed_text": seg["embed"],
                "page": page_no,
                "source": filename
            })
    return chunks

# ------------------- Embeddings -------------------

def embed_batch(texts: List[str], batch: int = BATCH_EMBED) -> List[List[float]]:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=chunk)
        # keep order by index
        for d in sorted(resp.data, key=lambda d: d.index):
            vecs.append(d.embedding)
    return vecs

# ------------------- Size-safe upsert -------------------

def _estimate_bytes(doc: Dict) -> int:
    # Rough but close enough for request sizing
    return len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))

def _iter_size_bounded_batches(docs: List[Dict], max_docs: int, max_bytes: int):
    batch, size = [], 0
    for d in docs:
        b = _estimate_bytes(d)
        # Extremely large single doc guard: split content if needed
        if b > max_bytes:
            # last-resort slice of content field to fit (keeps rest intact)
            content = d.get("content", "")
            if isinstance(content, str) and len(content) > 0:
                keep = max(1, int(len(content) * (max_bytes / b * 0.9)))
                d = {**d, "content": content[:keep]}
                b = _estimate_bytes(d)
        if batch and (len(batch) >= max_docs or size + b > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(d)
        size += b
    if batch:
        yield batch

def upsert_safely(search_client: SearchClient, docs: List[Dict]):
    """
    Upload in batches bounded by doc count and payload size.
    Retries once with halved limits on payload errors.
    """
    def _upload_with_limits(md: int, mb: int):
        for batch in _iter_size_bounded_batches(docs, md, mb):
            search_client.upload_documents(batch)

    try:
        _upload_with_limits(MAX_BATCH_DOCS, MAX_BATCH_BYTES)
    except HttpResponseError as e:
        msg = (getattr(e, "message", "") or str(e)).lower()
        if "disabled" in msg or "suspended" in msg:
            raise RuntimeError("Azure AI Search service is disabled/suspended.") from e
        if "request body is too large" in msg or "payload" in msg or "413" in msg or "400" in msg:
            # retry once with tighter caps
            time.sleep(RETRY_BACKOFF_SEC)
            half_docs = max(50, MAX_BATCH_DOCS // 2)
            half_bytes = max(1024 * 1024, MAX_BATCH_BYTES // 2)
            _upload_with_limits(half_docs, half_bytes)
        else:
            raise

# ------------------- Ingest folder -------------------

def _make_id(title: str, page: int, content: str) -> str:
    if not DETERMINISTIC_IDS:
        return str(uuid.uuid4())
    h = hashlib.sha1(f"{title}|{page}|{content[:128]}".encode("utf-8")).hexdigest()
    return h

def ingest_folder(folder: str = DOCS_DIR, doc_type: str = DOC_TYPE):
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdfs:
        print(f"No PDFs in {folder}")
        return
    grand_total = 0

    # fail fast if service unreachable/disabled
    try:
        _ = search_client.get_document_count()
    except HttpResponseError as e:
        raise RuntimeError(f"Azure AI Search not available: {(e.message or str(e))}") from e

    for pdf_path in pdfs:
        title = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            data = f.read()

        try:
            chunks = parse_pdf_into_chunks(data, title, doc_type=doc_type)
        except Exception as e:
            print(f"[WARN] DI parse failed for {title}: {e}")
            chunks = parse_with_pypdf2(data, title)

        if not chunks:
            print(f"[SKIP] No text extracted: {title}")
            continue

        # Embeddings
        embed_inputs = [c.get("embed_text", c["content"]) for c in chunks]
        vectors = embed_batch(embed_inputs, batch=BATCH_EMBED)

        batch_docs: List[Dict] = []
        for c, vec in zip(chunks, vectors):
            doc = {
                "id": _make_id(title, c["page"], c["content"]),
                "title": title,
                "content": c["content"],   # clean (no overlap duplication)
                "page": c["page"],
                "source": c["source"],
                "embedding": vec,
            }
            batch_docs.append(doc)

        if not batch_docs:
            print(f"[SKIP] Nothing to upload for: {title}")
            continue

        try:
            upsert_safely(search_client, batch_docs)
        except RuntimeError as e:
            raise  # bubble up clear error (disabled service, etc.)

        grand_total += len(batch_docs)
        print(f"Ingested {title}: {len(batch_docs)} chunks")

    print(f"Total chunks uploaded: {grand_total}")

# ------------------- CLI -------------------

if __name__ == "__main__":
    folder = DOCS_DIR
    dtype  = DOC_TYPE
    if len(sys.argv) >= 2:
        folder = sys.argv[1]
    if len(sys.argv) >= 3:
        dtype = sys.argv[2].lower()
    if dtype not in {"auto", "policy", "daq"}:
        dtype = "auto"
    ingest_folder(folder, dtype)
