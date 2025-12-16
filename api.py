from __future__ import annotations
import os, io, uuid, re, logging, zipfile
from typing import List, Dict, Optional, Tuple, Set
from collections import deque, defaultdict
import math

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient

# Optional import of semantic query enums if available (SDK >= 11.6)
try:
    from azure.search.documents.models import QueryType
except Exception:
    QueryType = None

from openai import OpenAI
from PyPDF2 import PdfReader  # fallback only

# ---------- Optional deps: pdfminer / pypdfium2 / reportlab ----------
HAVE_PDFMINER = False
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LAParams
    HAVE_PDFMINER = True
except Exception:
    pass

HAVE_PDFIUM = False
try:
    import pypdfium2 as pdfium
    HAVE_PDFIUM = True
except Exception:
    pass

HAVE_REPORTLAB = False
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except Exception:
    pass
else:
    HAVE_REPORTLAB = True

import json, time
from azure.core.exceptions import HttpResponseError

MAX_BATCH_DOCS  = int(os.getenv("MAX_BATCH_DOCS", "500"))
MAX_BATCH_BYTES = int(os.getenv("MAX_BATCH_BYTES", str(8 * 1024 * 1024)))  # ~8MB
RETRY_BACKOFF   = float(os.getenv("RETRY_BACKOFF_SEC", "1.0"))

def _estimate_bytes(doc: dict) -> int:
    return len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))

def _iter_size_bounded_batches(docs: List[Dict], max_docs: int, max_bytes: int):
    batch, size = [], 0
    for d in docs:
        b = _estimate_bytes(d)
        if b > max_bytes:
            # very large single record → trim content conservatively
            content = d.get("content")
            if isinstance(content, str) and content:
                keep = max(1, int(len(content) * (max_bytes / b * 0.9)))
                d = {**d, "content": content[:keep]}
                b = _estimate_bytes(d)
        if batch and (len(batch) >= max_docs or size + b > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(d); size += b
    if batch:
        yield batch



# ---------- Azure Document Intelligence (READ) ----------
HAVE_DI = False
try:
    # New SDK (azure-ai-documentintelligence >= 1.0)
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    HAVE_DI = True
except Exception:
    try:
        # Old SDK compat (azure-ai-formrecognizer <= 3.x)
        from azure.ai.formrecognizer import DocumentAnalysisClient as DocumentIntelligenceClient  # type: ignore
        AnalyzeDocumentRequest = None  # old SDK path doesn't use this type
        HAVE_DI = True
    except Exception:
        HAVE_DI = False

load_dotenv()
log = logging.getLogger("api")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# ================= ENV =================
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY      = os.environ["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX", "hr-policies")

OPENAI_API_KEY        = os.environ["OPENAI_API_KEY"]
EMBED_MODEL           = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL            = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SEMANTIC_CONFIG       = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "semantic-hr")

AZURE_DI_ENDPOINT     = os.getenv("AZURE_DI_ENDPOINT")
AZURE_DI_KEY          = os.getenv("AZURE_DI_KEY")

# Chunking knobs
CHUNK_MAX_CHARS       = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
CHUNK_OVERLAP_CHARS   = int(os.getenv("CHUNK_OVERLAP_CHARS", "240"))
HEADING_CONTEXT_DEPTH = int(os.getenv("HEADING_CONTEXT_DEPTH", "3"))
CROSS_PAGE_CHUNKING   = os.getenv("CROSS_PAGE_CHUNKING", "1") not in {"0","false","False"}

# OCR/rendering knobs
RENDER_DPI            = int(os.getenv("RENDER_DPI", "300"))

# Header/footer detection
HEADER_FOOTER_BAND    = float(os.getenv("HEADER_FOOTER_BAND", "0.12"))   # top/bottom 12%
HEADER_FOOTER_SUPPORT = float(os.getenv("HEADER_FOOTER_SUPPORT", "0.55"))# must appear on >=55% pages

# ================= Clients =================
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY))
oai = OpenAI(api_key=OPENAI_API_KEY)

di_client: Optional[DocumentIntelligenceClient] = None
if HAVE_DI and AZURE_DI_ENDPOINT and AZURE_DI_KEY:
    di_client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
    log.info("Azure Document Intelligence client configured.")
else:
    log.warning("Azure Document Intelligence not configured; OCR fallback disabled.")

# ================= App =================
app = FastAPI(title="RAG HR (Azure AI Search + Azure Document Intelligence READ)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

## ================= Regex / helpers =================
_ws = re.compile(r'\s+')

# bullets and ordered markers like "•", "-", "–", "1.", "1.2.", "(a)", "i)", "1]"
_bullet_like = re.compile(
    r'^\s*(?:[-–•●▪○*]|\(?(?:\d+(?:\.\d+)*|[A-Za-z]|[IVXLCDMivxlcdm]+)[\.\)\]])\s+'
)

# ordered-heading patterns (1., 1.1., A., (A), i., (i), also allows trailing '.' or ')' or ']')
_numbered_hdr = re.compile(r"""
^
(?:                                 # alternatives
    \(?\d+(?:\.\d+){0,6}            # 1  or 1.2 or 1.2.3 ... up to 6 levels
  | \(?[A-Z]                        # A or (A
  | \(?[IVXLCDMivxlcdm]+            # roman numerals or (i)
)
(?:[.\)\]])?                        # optional trailing ., ) or ]
\s+                                 # require at least some space after the marker
""", re.I | re.VERBOSE)


_annexure_hdr = re.compile(r'^(ANNEXURE|APPENDIX|SCHEDULE)\b', re.I)

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "").replace("-\n", "")  # soft hyphen + hyphenation
    s = s.replace("\uf0b7","•").replace("\u2022","•").replace("\u25CF","●")
    s = s.replace("\r", "\n")
    # collapse whitespace but keep single newlines (we’ll rejoin lines later)
    return s.replace("\n", " ").strip()

def _is_all_caps_words(t: str) -> bool:
    words = [w for w in re.split(r'\W+', t) if w]
    if not words:
        return False
    letters = "".join(w for w in t if w.isalpha())
    if not letters:
        return False
    up = sum(1 for c in letters if c.isupper())
    return up/len(letters) >= 0.9

def _looks_like_heading_text(text: str) -> bool:
    """
    Prefer enumerated headings first; then fall back to colon-ended, ALL CAPS, etc.
    This avoids demoting lines like '1. Introduction' to list items.
    """
    t = text.strip()
    if not t or len(t) > 180:
        return False

    # Prefer enumerated headings if the trailing text looks like a title
    m = _numbered_hdr.match(t)
    if m:
        tail = t[m.end():].strip().rstrip(':')
        if 0 < len(tail) <= 120:
            # short-ish, title-ish, and not ending with a period
            if _is_all_caps_words(tail) or (len(tail.split()) <= 16 and not tail.endswith('.')):
                return True

    # Otherwise, bullets are not headings
    if _bullet_like.match(t):
        return False

    if _annexure_hdr.match(t):
        return True
    if t.endswith(":") and len(t.split()) <= 16:
        return True
    if _is_all_caps_words(t) and len(t.split()) <= 12 and not t.endswith('.'):
        return True
    return False

def _heading_level(text: str) -> int:
    """Infer a reasonable level from numbering/lettering."""
    raw = text.strip()
    m = re.match(r"""
        ^
        (                               # capture the whole marker
            \(?\d+(?:\.\d+){0,6}        # 1 or 1.2.3...
          | \(?[A-Z]                    # A or (A
          | \(?[ivxlcdm]+               # roman numerals or (i)
        )
        [.\)\]]?                        # optional trailing punctuation
        \s+                             # space after
        """, raw, re.I | re.VERBOSE)
    if m:
        pref = m.group(1).lstrip('(').rstrip(').]')
        if pref and pref[0].isdigit():
            return 1 + pref.count('.')
        if pref and len(pref) == 1 and pref.isalpha():
            return 2
        return 3  # roman numerals → level 3
    if _annexure_hdr.match(raw):
        return 1
    if _is_all_caps_words(raw) and len(raw.split()) <= 12:
        return 1
    if raw.endswith(":"):
        return 2
    return 1


def _split_chars_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    # soft sentence split
    sents = re.split(r'(?<=[\.\!\?])\s+(?=[A-Z(])', text)
    out, buf = [], ""
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    # hard wrap + overlap
    step = max(1, max_chars - overlap)
    out2: List[str] = []
    for seg in out:
        if len(seg) <= max_chars:
            out2.append(seg)
        else:
            for i in range(0, len(seg), step):
                out2.append(seg[i:i+max_chars])
    if overlap and len(out2) > 1:
        stitched = [out2[0]]
        for i in range(1, len(out2)):
            tail = stitched[-1][-overlap:] if len(stitched[-1]) > overlap else stitched[-1]
            stitched.append((tail + " " + out2[i]).strip())
        return stitched
    return out2


# ================= Azure READ helpers =================
def _analyze_read(pdf_or_image_bytes: bytes):
    if not di_client:
        raise RuntimeError("Azure Document Intelligence client is not configured.")
    if AnalyzeDocumentRequest:
        poller = di_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=pdf_or_image_bytes),
        )
    else:
        poller = di_client.begin_analyze_document("prebuilt-read", document=pdf_or_image_bytes)
    return poller.result()

def _lines_from_read(result) -> List[Dict]:
    """Collect lines for ALL pages with coordinates (Azure READ origin: top-left)."""
    lines: List[Dict] = []
    for pg in getattr(result, "pages", []) or []:
        pno = int(getattr(pg, "page_number", 1) or 1)
        for li in getattr(pg, "lines", []) or []:
            txt = _norm(getattr(li, "content", ""))
            if not txt:
                continue
            poly = getattr(li, "polygon", []) or []
            xs = poly[0::2] or [0.0]
            ys = poly[1::2] or [0.0]
            x0 = float(min(xs)); y0 = float(min(ys))
            x1 = float(max(xs)); y1 = float(max(ys))
            lines.append({"page": pno, "x": x0, "y": y0, "x1": x1, "y1": y1, "text": txt, "src": "ocr"})
    return lines

# ================= pdfminer helpers =================
def _avg_font_size(line_obj: LTTextLine) -> float:
    sizes = []
    for ch in line_obj:
        if isinstance(ch, LTChar):
            sizes.append(getattr(ch, "size", 0.0))
    return float(sum(sizes)/len(sizes)) if sizes else 0.0

def _pdfminer_lines(pdf_bytes: bytes) -> Tuple[List[Dict], Dict[int, int], Dict[int, float]]:
    """
    Returns (lines, page_char_counts, page_heights)
    Normalizes y so sorting by y asc = top→bottom (PDF origin is bottom-left).
    """
    if not HAVE_PDFMINER:
        return [], {}, {}
    laparams = LAParams(line_overlap=0.3, char_margin=2.2, line_margin=0.45, word_margin=0.25, all_texts=True)
    f = io.BytesIO(pdf_bytes)
    lines: List[Dict] = []
    char_counts: Dict[int, int] = {}
    page_heights: Dict[int, float] = {}
    try:
        for page_num, layout in enumerate(extract_pages(f, laparams=laparams), start=1):
            page_lines: List[Tuple[float, float, float, float, str, float]] = []
            page_heights[page_num] = float(getattr(layout, "height", 1000.0) or 1000.0)
            for element in layout:
                if isinstance(element, LTTextContainer):
                    for obj in element:
                        if isinstance(obj, LTTextLine):
                            txt = _norm(obj.get_text() or "")
                            if not txt:
                                continue
                            x0, y0, x1, y1 = obj.bbox
                            y_top = -float(y0)  # flip so smaller = nearer to top
                            page_lines.append((float(x0), y_top, float(x1), float(y1), txt, _avg_font_size(obj)))
            page_lines.sort(key=lambda t: (t[1], t[0]))  # y asc, then x asc
            count = 0
            for x0, y_top, x1, y1, txt, fsz in page_lines:
                lines.append({"page": page_num, "x": x0, "y": y_top, "x1": x1, "y1": y1, "text": txt, "src": "pm", "fsz": fsz})
                count += len(txt)
            char_counts[page_num] = count
    except Exception as e:
        log.exception("pdfminer failed: %s", e)
        return [], {}, {}
    return lines, char_counts, page_heights

# ================= merge + reading order =================
def _merge_lines_per_page(pm_lines: List[Dict], rd_lines: List[Dict], dupe_px: float = 6.0) -> List[Dict]:
    """Merge lines from pdfminer and READ; drop near-duplicates by proximity + identical text."""
    by_page: Dict[int, List[Dict]] = {}
    for l in pm_lines:
        by_page.setdefault(l["page"], []).append(l)

    def is_dupe(existing: List[Dict], cand: Dict) -> bool:
        t = cand["text"]
        cx, cy = cand["x"], cand["y"]
        for e in existing:
            if e["text"] != t:
                continue
            if abs(e["x"] - cx) <= dupe_px and abs(e["y"] - cy) <= dupe_px:
                return True
        return False

    for cand in rd_lines:
        page = cand["page"]
        arr = by_page.setdefault(page, [])
        if not is_dupe(arr, cand):
            arr.append(cand)

    out: List[Dict] = []
    for p in sorted(by_page.keys()):
        out.extend(by_page[p])
    return out

def _sort_in_reading_order(page_lines: List[Dict]) -> List[Dict]:
    """Sort lines for human reading order with a lightweight 2-column heuristic."""
    if len(page_lines) < 12:
        return sorted(page_lines, key=lambda l: (l["y"], l["x"]))

    xs = sorted(l["x"] for l in page_lines)
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    if not gaps:
        return sorted(page_lines, key=lambda l: (l["y"], l["x"]))

    big_gap = max(gaps)
    median_gap = sorted(gaps)[len(gaps)//2]
    if big_gap > max(120.0, 4.0 * max(1.0, median_gap)):
        idx = gaps.index(big_gap)
        split_at = (xs[idx] + xs[idx+1]) / 2.0
        left  = [l for l in page_lines if l["x"] <= split_at]
        right = [l for l in page_lines if l["x"] >  split_at]
        left_sorted  = sorted(left,  key=lambda l: (l["y"], l["x"]))
        right_sorted = sorted(right, key=lambda l: (l["y"], l["x"]))
        return left_sorted + right_sorted

    return sorted(page_lines, key=lambda l: (l["y"], l["x"]))

# ================= header/footer & ToC bootstrapping =================
def _detect_repeating_headers_footers(lines: List[Dict], page_heights: Dict[int, float]) -> Tuple[Set[str], Set[str]]:
    """Find repeating header/footer lines that appear on most pages in the top/bottom band."""
    by_page = defaultdict(list)
    for ln in lines:
        by_page[ln["page"]].append(ln)
    total_pages = len(by_page) or 1

    header_counts = defaultdict(int)
    footer_counts = defaultdict(int)

    for p, arr in by_page.items():
        H = float(page_heights.get(p, 1000.0) or 1000.0)
        top_band = -H * HEADER_FOOTER_BAND
        bot_band = -H * (1.0 - HEADER_FOOTER_BAND)
        # remember y is already inverted (top ≈ smaller/more negative)
        for l in arr:
            txt = l["text"].strip()
            if not txt or len(txt) < 3:
                continue
            y = l["y"]
            if y <= top_band:
                header_counts[txt] += 1
            if y >= bot_band:
                footer_counts[txt] += 1

    header_set = {t for t, c in header_counts.items() if c/total_pages >= HEADER_FOOTER_SUPPORT}
    footer_set = {t for t, c in footer_counts.items() if c/total_pages >= HEADER_FOOTER_SUPPORT}
    return header_set, footer_set

def _extract_toc_candidates(lines: List[Dict]) -> Set[str]:
    """
    Heuristic ToC miner: scan first 2-3 pages for lines like "FACILITIES .... 6" or "Working at Plenum Technologies: ... 10"
    Captures the left text chunk before a page number or dot leader.
    """
    toc_titles: Set[str] = set()
    early = [l for l in lines if l["page"] <= 3]
    for l in early:
        t = l["text"].strip()
        # Look for a leader and trailing page number
        m = re.match(r'^(.{4,}?)(?:\.{2,}|\s)\s*([0-9]{1,3})\s*$', t)
        if m:
            title = m.group(1).strip()
            title = re.sub(r'\s{2,}', ' ', title).rstrip(':')
            if 2 <= len(title) <= 80:
                toc_titles.add(title.upper())
        else:
            if _is_all_caps_words(t) and 2 <= len(t) <= 80:
                toc_titles.add(t.upper())
    return toc_titles

# ================= page → items, chunking =================
def _items_by_page(lines: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Build items (heading | list | paragraph) per-page with column-aware ordering.
    Removes detected headers/footers.
    """
    by_page: Dict[int, List[Dict]] = {}
    for ln in lines:
        by_page.setdefault(ln["page"], []).append(ln)

    # compute page heights (fallback if missing)
    page_heights = {}
    for p, arr in by_page.items():
        ys = [abs(a["y"]) for a in arr]
        page_heights[p] = max(ys) if ys else 1000.0

    # strip repeating header/footer text
    header_set, footer_set = _detect_repeating_headers_footers([ln for arr in by_page.values() for ln in arr], page_heights)

    # ToC bootstrap
    toc = _extract_toc_candidates([ln for arr in by_page.values() for ln in arr])

    items_by_page: Dict[int, List[Dict]] = {}
    for pno, page_lines in by_page.items():
        # filter header/footer
        filtered = [ln for ln in page_lines if ln["text"].strip() not in header_set and ln["text"].strip() not in footer_set]
        page_lines = _sort_in_reading_order(filtered)

        # adaptive thresholds from y deltas
        y_vals = [l["y"] for l in page_lines]
        y_gaps = sorted([abs(y_vals[i+1]-y_vals[i]) for i in range(len(y_vals)-1)]) if len(y_vals) > 1 else [12.0]
        med_gap = y_gaps[len(y_gaps)//2] if y_gaps else 12.0
        MAX_Y_GAP = max(10.0, min(22.0, med_gap * 1.6))    # adaptive line gap threshold
        INDENT_TOL = 28.0                                  # indent tolerance for wraps

        items: List[Dict] = []
        classed = []
        for ln in page_lines:
            t = ln["text"].strip()
            # heading if in ToC or heuristics say so
            is_heading = _looks_like_heading_text(t) or (t.upper() in toc)
            if is_heading:
                k = "heading"
            elif _bullet_like.match(t) or _numbered_hdr.match(t):
                t = _norm(_bullet_like.sub("", t))
                t = _norm(_numbered_hdr.sub("", t))
                k = "listItem"
            else:
                k = "line"
            classed.append({**ln, "kind": k, "text": t})

        i = 0
        while i < len(classed):
            c = classed[i]
            if c["kind"] == "heading":
                items.append({"kind": "heading", "text": c["text"].rstrip(':'), "page": pno, "x": c["x"], "y": c["y"]})
                i += 1
                continue

            if c["kind"] == "listItem":
                bullets = [c["text"]]
                j = i + 1
                while j < len(classed) and classed[j]["kind"] == "listItem":
                    bullets.append(classed[j]["text"])
                    j += 1
                items.append({
                    "kind": "list",
                    "text": "\n".join(f"• {b}" for b in bullets),
                    "page": pno, "x": c["x"], "y": c["y"]
                })
                i = j
                continue

            # paragraph lines (stay on same page while merging wraps)
            para_lines = [c["text"]]
            x0, y0 = c["x"], c["y"]
            j = i + 1
            while j < len(classed):
                nxt = classed[j]
                if nxt["kind"] in {"heading", "listItem"}:
                    break
                dy = abs(nxt["y"] - y0)
                dx = abs(nxt["x"] - x0)
                # stop if a real block break (big vertical gap) after punctuation
                if dy > MAX_Y_GAP and para_lines[-1][-1:] in ".!?;:":
                    break
                # otherwise treat as wrap if indentation small or vertical gap tiny
                if dx <= INDENT_TOL or dy <= MAX_Y_GAP:
                    para_lines.append(nxt["text"])
                    x0, y0 = nxt["x"], nxt["y"]
                    j += 1
                else:
                    break
            para = _norm(" ".join(para_lines))
            if para:
                items.append({"kind": "paragraph", "text": para, "page": pno, "x": c["x"], "y": c["y"]})
            i = j

        items_by_page[pno] = items
    return items_by_page

def _build_chunks(items_by_page: Dict[int, List[Dict]], filename: str) -> List[Dict]:
    """
    Section-aware chunking with optional cross-page accumulation.
    Maintains a global heading stack; tries to keep bullets/paragraphs intact across page breaks.
    Uses character-overlap slicing to improve recall.
    """
    chunks: List[Dict] = []
    heading_stack: deque[tuple[int, str]] = deque(maxlen=24)

    def section_prefix() -> str:
        return "\n".join(h for _, h in list(heading_stack)[-HEADING_CONTEXT_DEPTH:])

    buf = ""
    cur_first_page = None
    cur_last_page = None

    def flush_buf():
        nonlocal buf, cur_first_page, cur_last_page
        if not buf.strip():
            return
        content = (section_prefix() + "\n" + buf).strip() if heading_stack else buf.strip()
        for piece in _split_chars_with_overlap(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            chunks.append({
                "page": cur_first_page or cur_last_page or 1,
                "page_first": cur_first_page or cur_last_page or 1,
                "page_last": cur_last_page or cur_first_page or 1,
                "content": piece,
                "source": filename,
                "section_path": " > ".join(h for _, h in heading_stack) or None,
            })
        buf = ""
        cur_first_page = None
        cur_last_page = None

    # iterate in page order
    pages = sorted(items_by_page.keys())
    carry_list = False  # track if a list flows across a page break

    for pno in pages:
        items = items_by_page[pno]

        for it in items:
            if it["kind"] == "heading":
                if buf:
                    flush_buf()
                lvl = _heading_level(it["text"])
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, it["text"]))
                carry_list = False
                continue

            unit = it["text"]
            if it["kind"] == "list":
                carry_list = True
            else:
                carry_list = False

            if not buf:
                cur_first_page = pno
                cur_last_page = pno
                buf = unit
            else:
                cand = (buf + "\n" + unit).strip()
                if len(cand) > CHUNK_MAX_CHARS:
                    flush_buf()
                    cur_first_page = pno
                    cur_last_page = pno
                    buf = unit
                else:
                    buf = cand
                    cur_last_page = pno

        # end-of-page: if cross-page disabled or unlikely to continue list, flush if large
        if not CROSS_PAGE_CHUNKING or not carry_list:
            next_p = next((pp for pp in pages if pp > pno), None)
            if (not next_p) or (buf and len(buf) >= int(CHUNK_MAX_CHARS * 0.9)):
                flush_buf()

    flush_buf()  # final
    return chunks

# ================= Hybrid parse pipeline =================
def parse_with_hybrid(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """pdfminer (true text) + Azure READ (OCR) → union → items → chunks"""
    pm_lines, _pm_counts, page_heights = _pdfminer_lines(pdf_bytes)
    rd_lines: List[Dict] = []
    if di_client:
        try:
            res = _analyze_read(pdf_bytes)
            rd_lines = _lines_from_read(res)
        except Exception as e:
            log.exception("Azure READ failed; continuing with pdfminer only: %s", e)

    if not pm_lines and not rd_lines:
        log.warning("No lines from pdfminer or READ; falling back to PyPDF2.")
        return parse_with_pypdf2(pdf_bytes, filename)

    selected_lines = _merge_lines_per_page(pm_lines, rd_lines, dupe_px=6.0)
    items_by_page = _items_by_page(selected_lines)
    return _build_chunks(items_by_page, filename)

# ================= Fallback (if everything else fails) =================
def parse_with_pypdf2(pdf_bytes: bytes, filename: str) -> List[Dict]:
    out: List[Dict] = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        log.exception("PyPDF2 failed to open: %s", e)
        return out
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        txt = _norm(raw)
        if not txt:
            continue
        for piece in _split_chars_with_overlap(txt, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            out.append({
                "page": page_no, "page_first": page_no, "page_last": page_no,
                "content": piece, "source": filename,
                "section_path": None,
            })
    return out

def parse_pdf_into_chunks(pdf_bytes: bytes, filename: str) -> List[Dict]:
    try:
        if HAVE_PDFMINER:
            return parse_with_hybrid(pdf_bytes, filename)
        if di_client:
            res = _analyze_read(pdf_bytes)
            lines = _lines_from_read(res)
            if lines:
                items = _items_by_page(lines)
                return _build_chunks(items, filename)
        return parse_with_pypdf2(pdf_bytes, filename)
    except Exception as e:
        log.exception("Parsing failed; using PyPDF2 fallback: %s", e)
        return parse_with_pypdf2(pdf_bytes, filename)

# ================= Optional: render images + OCR-ready PDF =================
def _render_pdf_to_pngs(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> List[Tuple[bytes, Tuple[int, int]]]:
    if not HAVE_PDFIUM:
        raise RuntimeError("pypdfium2 is not installed.")
    imgs: List[Tuple[bytes, Tuple[int, int]]] = []
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    n = len(doc)
    for i in range(n):
        page = doc[i]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        with io.BytesIO() as buf:
            pil_img.save(buf, format="PNG")
            png = buf.getvalue()
        imgs.append((png, (pil_img.width, pil_img.height)))
    return imgs

def _build_image_only_pdf(pngs: List[Tuple[bytes, Tuple[int, int]]], dpi: int = RENDER_DPI) -> bytes:
    if not HAVE_REPORTLAB:
        raise RuntimeError("reportlab is not installed.")
    out = io.BytesIO()
    for idx, (png, (w_px, h_px)) in enumerate(pngs):
        w_pt = (w_px / dpi) * 72.0
        h_pt = (h_px / dpi) * 72.0
        if idx == 0:
            c = canvas.Canvas(out, pagesize=(w_pt, h_pt))
        else:
            c.setPageSize((w_pt, h_pt))
        c.drawImage(ImageReader(io.BytesIO(png)), 0, 0, width=w_pt, height=h_pt, preserveAspectRatio=False, mask='auto')
        c.showPage()
    c.save()
    return out.getvalue()

# ================= Embeddings & scoring helpers =================
def _embed_batch(texts: List[str], batch: int = 64) -> List[List[float]]:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=part)
        for d in sorted(resp.data, key=lambda d: d.index):
            vecs.append(d.embedding)
    return vecs

def _upsert(docs: List[Dict]):
    """
    Upload in batches bounded by doc count AND request size.
    Retries once with tighter caps on payload errors.
    Raises RuntimeError if the Azure Search service is disabled/suspended.
    """
    # pre-flight: fail fast if service is disabled
    try:
        _ = search_client.get_document_count()
    except HttpResponseError as e:
        msg = (getattr(e, "message", "") or str(e)).lower()
        if "disabled" in msg or "suspended" in msg:
            raise RuntimeError("Azure AI Search service is disabled/suspended.") from e
        raise

    def _upload(md: int, mb: int):
        for batch in _iter_size_bounded_batches(docs, md, mb):
            search_client.upload_documents(documents=batch)

    try:
        _upload(MAX_BATCH_DOCS, MAX_BATCH_BYTES)
    except HttpResponseError as e:
        emsg = (getattr(e, "message", "") or str(e)).lower()
        if "disabled" in emsg or "suspended" in emsg:
            raise RuntimeError("Azure AI Search service is disabled/suspended.") from e
        if "request body is too large" in emsg or "payload" in emsg or "413" in emsg or "400" in emsg:
            time.sleep(RETRY_BACKOFF)
            _upload(max(50, MAX_BATCH_DOCS // 2), max(1024 * 1024, MAX_BATCH_BYTES // 2))
        else:
            raise

def _title_filter_or_none(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = title.strip().lower()
    if t in {"", "*", "string", "any", "all", "none", "null"}:
        return None
    safe = title.replace("'", "''")
    return f"title eq '{safe}'"

def _select_fields() -> List[str]:
    return ["title", "page", "page_first", "page_last", "source", "section_path", "content"]

def _exec_search(kwargs: Dict, semantic: bool = False):
    """
    Resilient wrapper: uses semantic ranker when requested and supported,
    gracefully falls back on older SDKs (no query_type) or non-semantic.
    """
    try:
        if semantic:
            try:
                if QueryType is not None:
                    return search_client.search(**kwargs,
                                                semantic_configuration_name=SEMANTIC_CONFIG,
                                                query_type=QueryType.SEMANTIC)
                else:
                    # Older SDKs accept the string "semantic" sometimes
                    return search_client.search(**kwargs,
                                                semantic_configuration_name=SEMANTIC_CONFIG,
                                                query_type="semantic")
            except TypeError:
                # Some older SDKs only expose semantic_configuration_name without query_type
                return search_client.search(**kwargs, semantic_configuration_name=SEMANTIC_CONFIG)
        return search_client.search(**kwargs)
    except TypeError:
        # transform vector_queries -> vector/vector_fields/k for older SDKs
        alt = dict(kwargs)
        vqs = alt.pop("vector_queries", None)
        if vqs:
            v = vqs[0]
            alt["vector"] = v.get("vector")
            alt["k"] = v.get("k")
            alt["vector_fields"] = v.get("fields")
        try:
            if semantic:
                if QueryType is not None:
                    return search_client.search(**alt,
                                                semantic_configuration_name=SEMANTIC_CONFIG,
                                                query_type=QueryType.SEMANTIC)
                else:
                    return search_client.search(**alt,
                                                semantic_configuration_name=SEMANTIC_CONFIG)
            return search_client.search(**alt)
        except Exception:
            return search_client.search(**alt)
    except HttpResponseError:
        # retry without semantic if server rejects
        return search_client.search(**kwargs)

def _dedupe_hits(hits: List[Dict]) -> List[Dict]:
    seen: Set[Tuple[str, int, str]] = set()
    out: List[Dict] = []
    for h in hits:
        key = (h.get("title") or "", int(h.get("page") or 0), (h.get("content") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

def _cosine(a: List[float], b: List[float]) -> float:
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0 or db == 0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))

def _calc_accuracy_scores(question_vec: List[float], docs: List[Dict]) -> List[Dict]:
    """
    Adds fields: azure_score (raw), azure_norm (0..1), cosine_sim (0..1), accuracy (0..1).
    Combines Azure ranker score with embedding cosine for robust ordering.
    """
    # Embed chunks (cap length for cost/latency)
    to_embed = [(d.get("content") or "")[:1200] for d in docs]
    chunk_vecs = _embed_batch(to_embed, batch=64) if to_embed else []

    # Azure score normalization
    raw_scores = []
    for d in docs:
        az = d.get("@search.score") or d.get("azure_score") or d.get("azureScore")
        if az is not None:
            raw_scores.append(float(az))
    mn = min(raw_scores) if raw_scores else 0.0
    mx = max(raw_scores) if raw_scores else 1.0
    span = (mx - mn) if (mx - mn) > 1e-9 else 1.0

    out = []
    for i, d in enumerate(docs):
        az_raw = d.get("@search.score") or d.get("azure_score") or d.get("azureScore")
        az_norm = ((float(az_raw) - mn) / span) if az_raw is not None else 0.0
        cos = _cosine(question_vec, chunk_vecs[i]) if i < len(chunk_vecs) else 0.0
        acc = 0.5 * az_norm + 0.5 * cos
        nd = dict(d)
        nd["azure_score"] = float(az_raw) if az_raw is not None else None
        nd["azure_norm"] = az_norm
        nd["cosine_sim"] = cos
        nd["accuracy"] = acc
        out.append(nd)
    return out


# ================= Models =================
class UploadResult(BaseModel):
    filename: str
    pages: int
    chunks_indexed: int

class UploadResponse(BaseModel):
    results: List[UploadResult]
    total_chunks: int

class ChatTurn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class AskIn(BaseModel):
    question: str
    hybrid: bool = True
    top: int = 10  # default 10 citations
    semantic: bool = True   # <—— semantic search ON by default
    title: Optional[str] = None
    history: Optional[List[ChatTurn]] = None  # optional chat history for style/continuity

class Citation(BaseModel):
    id: int
    title: Optional[str]
    page: Optional[int]
    source: Optional[str]
    section_path: Optional[str]
    snippet: str
    accuracy: float
    cosine_sim: float
    azure_score: Optional[float]

class AskOut(BaseModel):
    answer: str
    confidence: float
    citations: List[Citation]

# ================= Routes =================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "di": bool(di_client),
        "pdfminer": HAVE_PDFMINER,
        "pdfium": HAVE_PDFIUM,
        "reportlab": HAVE_REPORTLAB,
        "mode": "hybrid(pdfminer+read)" if HAVE_PDFMINER else ("prebuilt-read" if di_client else "pypdf2"),
        "cross_page_chunking": CROSS_PAGE_CHUNKING,
        "semantic_default": True,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS
    }

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="no files received")

    summary: List[UploadResult] = []
    grand_total = 0

    for f in files:
        if not (f.filename.lower().endswith(".pdf") or f.content_type in {"application/pdf", "application/octet-stream"}):
            raise HTTPException(status_code=415, detail=f"Only PDF supported. Got: {f.filename}")

        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")

        chunks = parse_pdf_into_chunks(data, f.filename)

        pages_seen: Set[int] = set()
        for c in chunks:
            p_first = c.get("page_first", c.get("page", 1))
            p_last = c.get("page_last",  c.get("page", 1))
            pages_seen.update(range(int(p_first), int(p_last) + 1))

        if not chunks:
            summary.append(UploadResult(filename=f.filename, pages=0, chunks_indexed=0))
            continue

        texts = [c.get("embed_text", c["content"]) for c in chunks]
        embeddings = _embed_batch(texts, batch=64)

        batch_docs: List[Dict] = []
        file_chunk_count = 0
        for c, vec in zip(chunks, embeddings):
            batch_docs.append({
                "id": str(uuid.uuid4()),
                "title": f.filename,
                "content": c["content"],
                "page": c["page"],
                "page_first": c.get("page_first", c["page"]),
                "page_last": c.get("page_last", c["page"]),
                "source": c["source"],
                "section_path": c.get("section_path"),
                "embedding": vec,
            })
            if len(batch_docs) >= 500:
                try:
                    _upsert(batch_docs)
                except RuntimeError as e:
                    if "disabled" in str(e).lower() or "suspended" in str(e).lower():
                        raise HTTPException(503, "Search service is disabled/suspended. Re-enable it or update AZURE_SEARCH_ENDPOINT.")
                    raise
                file_chunk_count += len(batch_docs)
                batch_docs.clear()

        if batch_docs:
            try:
                _upsert(batch_docs)
            except RuntimeError as e:
                if "disabled" in str(e).lower() or "suspended" in str(e).lower():
                    raise HTTPException(503, "Search service is disabled/suspended. Re-enable it or update AZURE_SEARCH_ENDPOINT.")
                raise
            file_chunk_count += len(batch_docs)

        summary.append(UploadResult(filename=f.filename, pages=len(pages_seen), chunks_indexed=file_chunk_count))
        grand_total += file_chunk_count

    return UploadResponse(results=summary, total_chunks=grand_total)

@app.post("/ask", response_model=AskOut)
async def ask(body: AskIn):
    # 1) Embed the question
    qvec = oai.embeddings.create(model=EMBED_MODEL, input=body.question).data[0].embedding

    # 2) Build search kwargs (hybrid semantic → vector-only → keyword semantic → browse)
    filter_expr = _title_filter_or_none(body.title)
    select = _select_fields()

    base_kwargs = dict(
        search_text=(body.question if body.hybrid else None),
        vector_queries=[{"kind": "vector", "vector": qvec, "k": max(10, body.top * 2), "fields": "embedding"}],
        top=body.top,
        select=select,
        search_fields=["content", "title"],
    )
    if filter_expr:
        base_kwargs["filter"] = filter_expr

    def _collect(res):
        out = []
        for r in res:
            az = None
            try:
                az = r.get("@search.score")
            except Exception:
                try:
                    az = r['@search.score']  # type: ignore
                except Exception:
                    az = None
            out.append({
                "title": r.get("title"),
                "page": r.get("page"),
                "page_first": r.get("page_first"),
                "page_last": r.get("page_last"),
                "source": r.get("source"),
                "section_path": r.get("section_path"),
                "content": r.get("content"),
                "@search.score": az,
            })
        return out

    docs: List[Dict] = []
    # Hybrid + semantic (preferred)
    try:
        results = _exec_search(base_kwargs, semantic=body.semantic and bool(base_kwargs.get("search_text")))
        docs = _collect(results)
    except Exception:
        docs = []

    # If filtered and empty → retry without filter
    if not docs and filter_expr:
        bk = dict(base_kwargs); bk.pop("filter", None)
        try:
            results = _exec_search(bk, semantic=body.semantic and bool(bk.get("search_text")))
            docs = _collect(results)
        except Exception:
            pass

    # Vector-only (no semantic ranker)
    if not docs:
        vkwargs = dict(base_kwargs)
        vkwargs["search_text"] = None
        vkwargs.pop("filter", None)
        try:
            results = _exec_search(vkwargs, semantic=False)
            docs = _collect(results)
        except Exception:
            pass

    # Keyword-only + semantic ranker
    if not docs:
        kkwargs = dict(
            search_text=(body.question if (body.question or "").strip() else "*"),
            top=body.top,
            select=select,
            search_fields=["content", "title"],
        )
        try:
            results = _exec_search(kkwargs, semantic=body.semantic and bool(kkwargs.get("search_text")))
            docs = _collect(results)
        except Exception:
            pass

    # Browse everything as last resort (no semantic)
    if not docs:
        skwargs = dict(search_text="*", top=body.top, select=select)
        try:
            results = _exec_search(skwargs, semantic=False)
            docs = _collect(results)
        except Exception:
            pass

    docs = _dedupe_hits(docs)
    if not docs:
        return AskOut(
            answer="I couldn't find any matching policy content. Try uploading documents first or rephrasing your question.",
            confidence=0.0,
            citations=[]
        )

    # 3) Score and re-rank with cosine + Azure score
    scored = _calc_accuracy_scores(qvec, docs)
    scored.sort(key=lambda d: d["accuracy"], reverse=True)

    # 4) Build stitched, diverse context blocks (MMR) and run grounded synthesis

    # ---- 4a) MMR diversify the top candidates so we don't send duplicates to the LLM
    def _mmr_select(docs: List[Dict], k: int = 8, lambda_: float = 0.7) -> List[int]:
        if not docs:
            return []
        k = min(k, len(docs))
        cand_texts = [(d.get("content") or "")[:1200] for d in docs]
        cand_vecs = _embed_batch(cand_texts, batch=64) if cand_texts else []
        if not cand_vecs:
            return list(range(k))

        # start with the most similar to the query
        sims_to_q = [_cosine(qvec, v) for v in cand_vecs]
        selected = [int(max(range(len(docs)), key=lambda i: sims_to_q[i]))]
        remaining = set(range(len(docs))) - set(selected)

        while len(selected) < k and remaining:
            best_i = None
            best_score = -1e9
            for i in remaining:
                max_sim_to_sel = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected) if selected else 0.0
                score = lambda_ * sims_to_q[i] - (1.0 - lambda_) * max_sim_to_sel
                if score > best_score:
                    best_score = score
                    best_i = i
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    mmr_idx = _mmr_select(scored, k=min(8, body.top), lambda_=0.7)
    mmr_docs = [scored[i] for i in mmr_idx]

    # ---- 4b) Stitch adjacent/related chunks (same title/section/source, contiguous pages)
    groups: List[Dict] = []
    for d in mmr_docs:
        t = d.get("title")
        sec = d.get("section_path") or ""
        src = d.get("source")
        p_first = d.get("page_first") or d.get("page") or 1
        p_last  = d.get("page_last")  or d.get("page") or p_first
        placed = False
        for g in groups:
            if g["title"] == t and (g["section_path"] or "") == sec and g["source"] == src:
                # if contiguous or overlapping pages, merge
                if not (p_first > g["page_last"] + 1 or p_last < g["page_first"] - 1):
                    g["content"] += "\n" + (d.get("content") or "")
                    g["page_first"] = min(g["page_first"], p_first)
                    g["page_last"]  = max(g["page_last"],  p_last)
                    # keep best metrics
                    g["azure_score"] = max(g["azure_score"], float(d.get("azure_score") or 0.0))
                    g["cosine_sim"]  = max(g["cosine_sim"],  float(d.get("cosine_sim")  or 0.0))
                    g["accuracy"]    = max(g["accuracy"],    float(d.get("accuracy")    or 0.0))
                    placed = True
                    break
        if not placed:
            groups.append({
                "title": t,
                "source": src,                      # keep the real filename/source
                "section_path": sec or None,
                "page_first": p_first,
                "page_last":  p_last,
                "content": d.get("content") or "",
                "azure_score": float(d.get("azure_score") or 0.0),
                "cosine_sim":  float(d.get("cosine_sim")  or 0.0),
                "accuracy":    float(d.get("accuracy")    or 0.0),
            })

    # keep the most useful stitched groups
    groups.sort(key=lambda g: g["accuracy"], reverse=True)
    groups = groups[:min(6, len(groups))]

    # ---- 4c) Build numbered context blocks
    ctx_blocks = []
    for i, g in enumerate(groups, start=1):
        sec = f" | {g['section_path']}" if g.get("section_path") else ""
        pg_str = f"p.{g['page_first']}" if g["page_first"] == g["page_last"] else f"pp.{g['page_first']}-{g['page_last']}"
        text = (g["content"] or "").strip()
        if len(text) > 2000:
            text = text[:2000]
        ctx_blocks.append(f"[{i}] {g['title']}{sec} | {pg_str}\n{text}")

    # also rebuild citations aligned to the stitched groups (so [n] matches)
    citations: List[Citation] = []
    for i, g in enumerate(groups, start=1):
        snippet = (g.get("content") or "").strip()[:800]
        citations.append(Citation(
            id=i,
            title=g.get("title"),
            page=g.get("page_first"),
            source=g.get("source"),              # ← fixed to use real source
            section_path=g.get("section_path"),
            snippet=snippet,
            accuracy=float(g.get("accuracy") or 0.0),
            cosine_sim=float(g.get("cosine_sim") or 0.0),
            azure_score=float(g.get("azure_score") or 0.0)
        ))

    # ---- 4d) Compose messages and ask the model to synthesize across blocks.
    system_prompt = (
        "You are an HR policy assistant. Answer using ONLY the provided context blocks. "
        "Synthesize across multiple blocks; if two blocks disagree, prefer the most specific or most recent text. "
        "Do NOT invent or assume. If something is not specified, say so explicitly. "
        "Cite sources inline using bracketed numbers like [1], [2] (matching the blocks below). "
        "Prefer bullet points when listing options. Keep it concise and precise.\n\n"
        "At the end of your answer, add a line like: <CONF=0.00..1.00> reflecting how strongly the context supports your answer."
    )

    user_prompt = (
        f"User question:\n{body.question}\n\n"
        "Use ONLY these numbered context blocks:\n\n" +
        "\n\n---\n\n".join(ctx_blocks)
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (body.history or []):
        if turn.role in {"user", "assistant", "system"} and turn.content:
            messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_prompt})

    chat = oai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages,
    )
    raw_answer = chat.choices[0].message.content or ""

    # ---- 4e) Parse trailing confidence tag if present, and combine with retrieval confidence
    model_conf = None
    m = re.search(r"<CONF\s*=\s*([0-1](?:\.\d+)?)\s*>", raw_answer)
    if m:
        try:
            model_conf = float(m.group(1))
            raw_answer = re.sub(r"\s*<CONF\s*=\s*[0-1](?:\.\d+)?\s*>\s*$", "", raw_answer).strip()
        except Exception:
            model_conf = None

    # retrieval confidence from stitched groups (rank-weighted)
    if citations:
        weights = [1.0 / (i + 0.0) for i in range(1, len(citations) + 1)]
        s = sum(w * c.accuracy for w, c in zip(weights, citations))
        retrieval_conf = s / sum(weights)
    else:
        retrieval_conf = 0.0

    # blend (favor the lower to be conservative if model_conf present)
    if model_conf is not None:
        confidence = round(float(0.5 * retrieval_conf + 0.5 * model_conf), 4)
    else:
        confidence = round(float(retrieval_conf), 4)

    return AskOut(
        answer=raw_answer,
        confidence=confidence,
        citations=citations
    )

@app.get("/chunks")
def list_chunks(
    q: str = Query("*", description="Search text. Use * for all."),
    title: Optional[str] = Query(None, description="Exact title filter"),
    page: Optional[int] = Query(None, description="Exact page filter"),
    top: int = Query(50, ge=1, le=1000, description="Items per page (max 1000)"),
    skip_token: Optional[str] = Query(None, description="Continuation token from previous call"),
    order: Optional[str] = Query(None, description="orderby expression, defaults to title/page when q='*'")
):
    flt = []
    if title:
        flt.append("title eq '{}'".format(title.replace("'", "''")))
    if page is not None:
        flt.append(f"page eq {page}")
    filter_expr = " and ".join(flt) if flt else None

    kwargs = dict(
        search_text=q or "*",
        top=top,
        include_total_count=True,
        select=["id", "title", "page", "page_first", "page_last", "source", "section_path", "content"],
    )
    if filter_expr:
        kwargs["filter"] = filter_expr

    if order:
        kwargs["order_by"] = [s.strip() for s in order.split(",") if s.strip()]
    elif q == "*" or q.strip() == "":
        kwargs["order_by"] = ["title asc", "page asc"]

    results = search_client.search(**kwargs)
    pager = results.by_page(continuation_token=skip_token)
    try:
        this_page = next(pager)
    except StopIteration:
        return {"items": [], "nextToken": None, "totalCount": 0}

    items = []
    for doc in this_page:
        items.append({
            "id": doc.get("id"),
            "title": doc.get("title"),
            "page": doc.get("page"),
            "page_first": doc.get("page_first"),
            "page_last": doc.get("page_last"),
            "source": doc.get("source"),
            "section_path": doc.get("section_path"),
            "contentPreview": (doc.get("content") or "")[:420]
        })

    next_token = getattr(pager, "continuation_token", None)
    total = getattr(results, "get_count", lambda: None)()

    return {"items": items, "nextToken": next_token, "totalCount": total}

# ---------- Optional: build an OCR-ready PDF & a ZIP of page PNGs ----------
@app.post("/prep-ocr")
async def prep_ocr(file: UploadFile = File(...), return_zip: bool = False):
    """
    Renders the PDF to 300-DPI PNGs and returns an image-only PDF (great for OCR engines).
    If return_zip=True, returns a ZIP with all PNGs + the OCR PDF inside.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF supported.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    if not HAVE_PDFIUM:
        raise HTTPException(500, "pypdfium2 not installed on server.")
    if not HAVE_REPORTLAB:
        raise HTTPException(500, "reportlab not installed on server.")

    try:
        pngs = _render_pdf_to_pngs(data, dpi=RENDER_DPI)  # [(bytes, (w,h)), ...]
        img_only_pdf = _build_image_only_pdf(pngs, dpi=RENDER_DPI)

        if not return_zip:
            fn = os.path.splitext(os.path.basename(file.filename))[0] + "_ocr_ready.pdf"
            return StreamingResponse(
                io.BytesIO(img_only_pdf),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{fn}"'}
            )
        # Return as ZIP
        tmp = io.BytesIO()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ocr_ready.pdf", img_only_pdf)
            for i, (png, _wh) in enumerate(pngs, start=1):
                z.writestr(f"page_{i:03d}.png", png)
        tmp.seek(0)
        zip_fn = os.path.splitext(os.path.basename(file.filename))[0] + "_pages_and_ocr_pdf.zip"
        return StreamingResponse(
            tmp,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_fn}"'}
        )

    except Exception as e:
        log.exception("prep-ocr failed: %s", e)
        raise HTTPException(500, f"prep-ocr failed: {e}")

from fastapi.responses import HTMLResponse

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    """Serve a lightweight HTML chat interface directly from FastAPI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Policy Chat</title>
<style>
  body { font-family: system-ui, sans-serif; background:#f6f7fb; margin:0; display:flex; flex-direction:column; height:100vh; }
  header { background:#004578; color:white; padding:12px 20px; font-size:1.2em; font-weight:600; }
  #chat { flex:1; overflow-y:auto; padding:16px; }
  .msg { margin-bottom:16px; max-width:80%; line-height:1.5; white-space:pre-wrap; }
  .user { background:#0078d4; color:white; align-self:flex-end; border-radius:12px 12px 0 12px; padding:10px 14px; margin-left:auto; }
  .assistant { background:white; border-radius:12px 12px 12px 0; padding:10px 14px; box-shadow:0 0 3px rgba(0,0,0,0.1); }
  #input-bar { display:flex; border-top:1px solid #ccc; padding:10px; background:white; }
  #q { flex:1; border:1px solid #ccc; border-radius:8px; padding:8px; font-size:1em; }
  button { background:#0078d4; color:white; border:none; border-radius:8px; margin-left:8px; padding:8px 14px; cursor:pointer; }
  button:hover { background:#005a9e; }
  .confidence { font-size:0.85em; color:#666; margin-top:4px; }
  .citations { font-size:0.85em; color:#888; margin-top:4px; }
</style>
</head>
<body>
  <header>HR Policy Assistant</header>
  <div id="chat"></div>
  <div id="input-bar">
    <input id="q" type="text" placeholder="Ask about company policies..." autocomplete="off">
    <button onclick="send()">Send</button>
  </div>
<script>
const chatDiv = document.getElementById('chat');
const qInput = document.getElementById('q');
let history = [];

function appendMessage(role, text, extra="") {
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  msg.innerHTML = text + (extra ? `<div class="confidence">${extra}</div>` : "");
  chatDiv.appendChild(msg);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

async function send() {
  const text = qInput.value.trim();
  if (!text) return;
  appendMessage('user', text);
  qInput.value = '';

  const body = {
    question: text,
    hybrid: true,
    top: 8,
    semantic: true,
    history: history
  };

  appendMessage('assistant', "Thinking...");

  try {
    const res = await fetch("/ask", {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    const last = chatDiv.querySelector('.assistant:last-child');
    last.remove();

    const citations = (data.citations || [])
      .map(c => `[${c.id}] ${c.title} p.${c.page}`)
      .join(', ');
    const extra = `Confidence: ${(data.confidence*100).toFixed(1)}%` +
                  (citations ? `<br><span class="citations">Sources: ${citations}</span>` : "");
    appendMessage('assistant', data.answer, extra);

    history.push({role: 'user', content: text});
    history.push({role: 'assistant', content: data.answer});
  } catch (err) {
    console.error(err);
    const last = chatDiv.querySelector('.assistant:last-child');
    if (last) last.remove();
    appendMessage('assistant', "❌ Error contacting the server.");
  }
}
qInput.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
</script>
</body>
</html>
"""
