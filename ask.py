# ask.py
from __future__ import annotations
import os, json, math, argparse
from typing import Optional, List, Dict, Tuple, Set

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from openai import OpenAI

load_dotenv()

# ---------- Env (aligned with ingest) ----------
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY      = os.environ["AZURE_SEARCH_KEY"]
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX", "hr-policies")

EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "gpt-4o-mini")
SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "semantic-hr")

# ---------- Clients ----------
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY))
oai           = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- Helpers ----------
def _safe_filter_for_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = title.strip().lower()
    if t in {"", "*", "string", "any", "all", "none", "null"}:
        return None
    safe = title.replace("'", "''")  # OData escape single quotes
    return f"title eq '{safe}'"

def _select_fields() -> List[str]:
    # Include optional fields if they exist in your index
    return ["title", "page", "page_first", "page_last", "source", "section_path", "content"]

def _fmt_citation(d: Dict) -> str:
    title = d.get("title") or "<untitled>"
    pg    = d.get("page") or d.get("page_first") or "?"
    return f"[{title} | p.{pg}]\n{d.get('content') or ''}"

def _is_browse_query(q: str) -> bool:
    return (q or "").strip() in {"", "*"}

def _looks_noisy(content: str, min_chars: int) -> bool:
    if not content:
        return True
    s = content.strip()
    if len(s) < min_chars:
        return True
    alnum = sum(ch.isalnum() for ch in s)
    non_alnum = max(1, len(s) - alnum)
    return (alnum / non_alnum) < 0.2  # mostly non-text

def _exec_search(kwargs: Dict, semantic: bool, debug: bool = False):
    """
    Handles semantic ranker fallback and SDK param differences
    (vector_queries vs vector/vector_fields/k).
    """
    try:
        if semantic:
            return search_client.search(**kwargs, semantic_configuration_name=SEMANTIC_CONFIG)
        return search_client.search(**kwargs)
    except TypeError as te:
        if debug:
            print("[debug] TypeError with vector_queries; retrying legacy vector args:", te)
        alt = dict(kwargs)
        vqs = alt.pop("vector_queries", None)
        if vqs:
            v = vqs[0]
            alt["vector"] = v.get("vector")
            alt["k"] = v.get("k")
            alt["vector_fields"] = v.get("fields")
        try:
            if semantic:
                return search_client.search(**alt, semantic_configuration_name=SEMANTIC_CONFIG)
            return search_client.search(**alt)
        except HttpResponseError:
            return search_client.search(**alt)
    except HttpResponseError as he:
        if debug:
            print("[debug] HttpResponseError; retrying without semantic:", he)
        try:
            return search_client.search(**kwargs)
        except TypeError:
            alt = dict(kwargs)
            vqs = alt.pop("vector_queries", None)
            if vqs:
                v = vqs[0]
                alt["vector"] = v.get("vector")
                alt["k"] = v.get("k")
                alt["vector_fields"] = v.get("fields")
            return search_client.search(**alt)

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

def _apply_noise_filter(hits: List[Dict], min_chars: int) -> List[Dict]:
    if min_chars <= 0:
        return hits
    return [h for h in hits if not _looks_noisy(h.get("content") or "", min_chars)]

def _collect_results(res) -> List[Dict]:
    out: List[Dict] = []
    for r in res:
        # Try both patterns for Azure score
        az = None
        try:
            az = r.get("@search.score")
        except Exception:
            pass
        if az is None:
            try:
                az = getattr(r, "score")  # some SDKs expose it this way
            except Exception:
                pass
        out.append({
            "title": r.get("title"),
            "page": r.get("page"),
            "page_first": r.get("page_first"),
            "page_last": r.get("page_last"),
            "source": r.get("source"),
            "section_path": r.get("section_path"),
            "content": r.get("content"),
            "azure_score": float(az) if az is not None else None,
        })
    return out

def _embed_batch(texts: List[str], batch: int = 64) -> List[List[float]]:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=part)
        # ensure original order
        for d in sorted(resp.data, key=lambda d: d.index):
            vecs.append(d.embedding)
    return vecs

def _cosine(a: List[float], b: List[float]) -> float:
    num = 0.0; da = 0.0; db = 0.0
    for x, y in zip(a, b):
        num += x * y; da += x * x; db += y * y
    if da == 0 or db == 0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))

def _score_and_rank(question_vec: List[float], hits: List[Dict], weight_cosine: float = 0.5) -> List[Dict]:
    """
    Add cosine_sim, azure_norm, accuracy to each hit and sort by accuracy desc.
    """
    # 1) cosine on re-embedded snippets (cap length for cost)
    snippets = [ (h.get("content") or "")[:1200] for h in hits ]
    chunk_vecs = _embed_batch(snippets) if hits else []

    # 2) normalize azure score to 0..1 across current hits
    raw = [h["azure_score"] for h in hits if h.get("azure_score") is not None]
    mn = min(raw) if raw else 0.0
    mx = max(raw) if raw else 1.0
    span = (mx - mn) if (mx - mn) > 1e-9 else 1.0

    ranked: List[Dict] = []
    for i, h in enumerate(hits):
        cos = _cosine(question_vec, chunk_vecs[i]) if i < len(chunk_vecs) else 0.0
        az_raw = h.get("azure_score")
        az_norm = ((az_raw - mn) / span) if az_raw is not None else 0.0
        wcos = max(0.0, min(1.0, weight_cosine))
        waz  = 1.0 - wcos
        acc = wcos * cos + waz * az_norm
        ranked.append({**h, "cosine_sim": cos, "azure_norm": az_norm, "accuracy": acc})
    ranked.sort(key=lambda d: d["accuracy"], reverse=True)
    return ranked

# ---------- Retrieval ----------
def retrieve(
    question: str,
    hybrid: bool,
    k_vec: int = 10,
    top: int = 10,
    semantic: bool = False,
    title: Optional[str] = None,
    debug: bool = False,
    min_chars: int = 0,
) -> List[Dict]:
    """
    Hybrid keyword + vector retrieval with optional semantic ranker and title filter.
    """
    if _is_browse_query(question):
        kw_kwargs = dict(
            search_text="*",
            top=top,
            select=_select_fields(),
            include_total_count=True,
        )
        flt = _safe_filter_for_title(title)
        if flt:
            kw_kwargs["filter"] = flt
        if debug:
            print("[debug] plan 0 (browse '*') kwargs:", kw_kwargs)
        res = _exec_search(kw_kwargs, semantic=False, debug=debug)
        items = _collect_results(res)
        if debug and hasattr(res, "get_count"):
            print("[debug] plan 0 count:", res.get_count())
        return _apply_noise_filter(_dedupe_hits(items), min_chars)

    k_vec = max(k_vec, top * 2)  # headroom on vector side
    qvec = oai.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

    base_kwargs = dict(
        search_text=(question if hybrid else None),
        vector_queries=[{"kind": "vector", "vector": qvec, "k": k_vec, "fields": "embedding"}],
        top=top,
        select=_select_fields(),
        include_total_count=True,
        search_fields=["content", "title"],
    )
    flt = _safe_filter_for_title(title)
    if flt:
        base_kwargs["filter"] = flt

    if debug:
        dbg = {**base_kwargs, "vector_queries": "[…]"}
        print("[debug] plan A (hybrid) kwargs:", dbg, "| semantic:", semantic)

    items: List[Dict] = []
    try:
        results = _exec_search(base_kwargs, semantic=semantic, debug=debug)
        if debug and hasattr(results, "get_count"):
            print("[debug] plan A count:", results.get_count())
        items = _collect_results(results)
    except Exception as e:
        if debug:
            print("[debug] plan A failed:", repr(e))

    if not items:
        vec_kwargs = dict(base_kwargs); vec_kwargs["search_text"] = None
        if debug:
            print("[debug] plan B (vector-only) kwargs:", {**vec_kwargs, "vector_queries": "[…]"})
        try:
            results = _exec_search(vec_kwargs, semantic=False, debug=debug)
            if debug and hasattr(results, "get_count"):
                print("[debug] plan B count:", results.get_count())
            items = _collect_results(results)
        except Exception as e:
            if debug:
                print("[debug] plan B failed:", repr(e))

    if not items:
        kw_kwargs = dict(
            search_text=(question if question.strip() else "*"),
            top=top,
            select=_select_fields(),
            include_total_count=True,
            search_fields=["content", "title"],
        )
        if flt:
            kw_kwargs["filter"] = flt
        if debug:
            print("[debug] plan C (keyword-only) kwargs:", kw_kwargs)
        try:
            results = _exec_search(kw_kwargs, semantic=semantic, debug=debug)
            if debug and hasattr(results, "get_count"):
                print("[debug] plan C count:", results.get_count())
            items = _collect_results(results)
        except Exception as e:
            if debug:
                print("[debug] plan C failed:", repr(e))

    if not items:
        star_kwargs = dict(
            search_text="*",
            top=top,
            select=_select_fields(),
            include_total_count=True,
        )
        if flt:
            star_kwargs["filter"] = flt
        if debug:
            print("[debug] plan D (browse '*') kwargs:", star_kwargs)
        try:
            results = _exec_search(star_kwargs, semantic=False, debug=debug)
            if debug and hasattr(results, "get_count"):
                print("[debug] plan D count:", results.get_count())
            items = _collect_results(results)
        except Exception as e:
            if debug:
                print("[debug] plan D failed:", repr(e))

    # Attach query vector so caller can score
    for it in items:
        it["_qvec"] = qvec
    return _apply_noise_filter(_dedupe_hits(items), min_chars)

# ---------- Neighbor expansion (pull adjacent pages for context) ----------
def expand_neighbors(
    hits: List[Dict],
    neighbors: int = 1,
    per_doc_cap: int = 12,
    debug: bool = False,
) -> List[Dict]:
    if neighbors <= 0 or not hits:
        return hits

    pages_by_title: Dict[str, Set[int]] = {}
    for h in hits:
        t = h.get("title") or ""
        p = int(h.get("page") or 0)
        if p:
            pages_by_title.setdefault(t, set()).add(p)

    expanded: List[Dict] = list(hits)
    for title, pages in pages_by_title.items():
        if not title or not pages:
            continue
        wanted: Set[int] = set()
        for p in pages:
            for d in range(-neighbors, neighbors + 1):
                if p + d >= 1:
                    wanted.add(p + d)

        flt_title = _safe_filter_for_title(title)
        page_exprs = [f"page eq {p}" for p in sorted(wanted)]
        page_filter = " or ".join(page_exprs)
        full_filter = f"{flt_title} and ({page_filter})" if flt_title else f"({page_filter})"

        kwargs = dict(search_text="*", filter=full_filter, top=min(per_doc_cap, len(wanted) * 2), select=_select_fields())
        try:
            kwargs_ordered = dict(kwargs); kwargs_ordered["order_by"] = ["page asc"]
            res = _exec_search(kwargs_ordered, semantic=False, debug=debug)
        except Exception:
            if debug:
                print(f"[debug] neighbor order_by unsupported; retrying without order_by for '{title}'")
            res = _exec_search(kwargs, semantic=False, debug=debug)

        for r in res:
            expanded.append({
                "title":   r.get("title"),
                "page":    r.get("page"),
                "source":  r.get("source"),
                "content": r.get("content"),
            })

    return _dedupe_hits(expanded)

# ---------- Summarization with numbered citations ----------
def summarize_with_citations(question: str, ranked_hits: List[Dict], max_context: int = 10, force_answer: bool = False) -> str:
    if not ranked_hits:
        return "I couldn’t find any matching policy content in the index."

    # Numbered blocks for clear [1], [2] citation style
    blocks = []
    for i, h in enumerate(ranked_hits[:max_context], 1):
        title = h.get("title") or "<untitled>"
        pg = h.get("page") or h.get("page_first") or "?"
        sec = f" | {h.get('section_path')}" if h.get("section_path") else ""
        snippet = (h.get("content") or "").strip()
        blocks.append(f"[{i}] {title}{sec} | p.{pg}\n{snippet}")

    system = (
        "You are an HR policy assistant. Answer conversationally but precisely using ONLY the provided context. "
        "If the exact topic is not specified, say what isn’t specified and summarize the nearest relevant guidance. "
        "Always cite inline using [1], [2], ... that match the numbered sources."
    )
    user = f"Question:\n{question}\n\nUse ONLY these numbered context blocks:\n\n" + "\n\n---\n\n".join(blocks)
    if force_answer:
        user = "Always provide a helpful summary from the context, even if the exact term is not present.\n\n" + user

    chat = oai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return chat.choices[0].message.content

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--hybrid", action="store_true", help="Use keyword + vector in one call")
    ap.add_argument("--semantic", action="store_true", help="Use semantic ranker if configured")
    ap.add_argument("--top", type=int, default=10, help="Number of chunks to retrieve")
    ap.add_argument("--kvec", type=int, default=10, help="Vector neighbors (auto-bumped to >= 2*top)")
    ap.add_argument("--title", type=str, default=None, help="Exact document title filter (OData eq)")
    ap.add_argument("--neighbors", type=int, default=1, help="Also pull this many adjacent pages per hit")
    ap.add_argument("--perdoccap", type=int, default=12, help="Cap neighbor results per document")
    ap.add_argument("--min-chars", type=int, default=30, help="Drop chunks with fewer than this many chars (noise filter)")
    ap.add_argument("--max-context", type=int, default=10, help="Max chunks to send to the LLM")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--noanswer", action="store_true", help="Skip the LLM step; only show retrieved chunks")
    ap.add_argument("--force-answer", action="store_true", help="Even if exact topic not present, summarize closest guidance")
    ap.add_argument("--json", action="store_true", help="Emit JSON with scores and confidence")
    ap.add_argument("--wcos", type=float, default=0.5, help="Weight for cosine in accuracy (0..1)")
    args = ap.parse_args()

    # 1) retrieve
    hits = retrieve(
        args.question,
        hybrid=args.hybrid,
        k_vec=args.kvec,
        top=args.top,
        semantic=args.semantic,
        title=args.title,
        debug=args.debug,
        min_chars=args.min_chars,
    )

    # 2) optional neighbor expansion
    hits = expand_neighbors(hits, neighbors=args.neighbors, per_doc_cap=args.perdoccap, debug=args.debug)

    # 3) score & rank
    if hits and "_qvec" in hits[0]:
        qvec = hits[0]["_qvec"]
    else:
        qvec = oai.embeddings.create(model=EMBED_MODEL, input=args.question).data[0].embedding
    ranked = _score_and_rank(qvec, hits, weight_cosine=args.wcos)
    topk = ranked[:args.top]

    # confidence = weighted average of accuracy by 1/rank
    if topk:
        weights = [1.0/(i+0.0) for i in range(1, len(topk)+1)]
        s = sum(w*h["accuracy"] for w, h in zip(weights, topk))
        confidence = s / sum(weights)
    else:
        confidence = 0.0

    # 4) output
    if args.json:
        print(json.dumps({
            "question": args.question,
            "confidence": round(confidence, 4),
            "citations": [
                {
                    "rank": i+1,
                    "title": h.get("title"),
                    "page": h.get("page") or h.get("page_first"),
                    "source": h.get("source"),
                    "section_path": h.get("section_path"),
                    "accuracy": round(h.get("accuracy", 0.0), 4),
                    "cosine_sim": round(h.get("cosine_sim", 0.0), 4),
                    "azure_score": h.get("azure_score"),
                    "snippet": (h.get("content") or "")[:600],
                } for i, h in enumerate(topk)
            ]
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\nTop {min(len(topk), args.top)} chunks (re-ranked by accuracy):")
        for i, h in enumerate(topk, 1):
            title = h.get("title") or "<untitled>"
            page  = h.get("page") or h.get("page_first") or "?"
            preview = (h.get("content") or "").replace("\n", " ")[:160]
            acc = h.get("accuracy", 0.0)
            cos = h.get("cosine_sim", 0.0)
            az  = h.get("azure_score")
            print(f"{i}. {title} (p.{page})  acc={acc:.3f}  cos={cos:.3f}  az={az if az is not None else '—'}")
            print(f"   {preview}...")

        print("\nAnswer:\n--------")
        if args.noanswer or _is_browse_query(args.question):
            print("(skipped LLM in browse/noanswer mode)")
        else:
            send = topk[: max(args.max_context, args.top)]
            ans = summarize_with_citations(args.question, send, max_context=args.max_context, force_answer=args.force_answer)
            print(ans)
            print(f"\n[confidence: {confidence:.3f}]")
