#!/usr/bin/env python3
"""
Section-aware PDF -> ChromaDB ingestion (content only).

Stored per chunk:  "Paper: <title> / Section: <path>" + body text.
Metadata:          source, title, section, chunk_type  (no page numbers, no authors).

Removed on purpose: figure captions, authors/affiliations/emails, received/accepted
lines, keywords, licence text, references, acknowledgements, running headers/footers,
and inline citation markers like [12] or [3-5].
Kept: body text, abstract, highlights, and real tables (with their caption).

Install:  pip install pymupdf chromadb sentence-transformers
Usage:
  python ingest_papers.py --pdf_dir ./papers
  python ingest_papers.py --pdf_dir ./papers --dump chunks.jsonl --no_store
  python ingest_papers.py --query "thermal vacuum test temperature range"
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

DEFAULT_DB = str(Path(__file__).resolve().parent / "chroma_db")
MAX_WORDS = 300
MIN_WORDS = 60
HARD_CAP = 1.3
MERGE_SMALL = True
TOP_MARGIN, BOTTOM_MARGIN = 0.07, 0.935

NUM_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z].{2,160})$", re.S)
NUM_START = re.compile(r"^\d+(?:\.\d+){0,3}\s+[A-Z]")
KNOWN_HEADINGS = {"abstract", "highlights", "introduction", "conclusion", "conclusions",
                  "abbreviations", "discussion", "methods", "results"}
SKIP_SECTIONS = {"abbreviations"}
STOP_PREFIXES = ("references", "acknowledg", "funding", "declarations", "author contributions",
                 "conflict of interest", "competing interest", "data availability")
CAPTION_RE = re.compile(r"^(Fig\.?|Figure|Table)\s*\d+", re.I)
NOISE_RE = re.compile(r"^(Vol\.:\(0123456789\)|1\s*3|\d{1,4}|Springer|Publisher.s Note.*)$", re.I)
DROP_PARA = re.compile(r"^(keywords?\b|received:|accepted:|published online|revised:|©|"
                       r"this article is licensed|open access|publisher.s note|"
                       r"\*?\s*correspondence|check for updates|article highlights$)", re.I)
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
CITE = re.compile(r"\s*\[\s*\d+(?:\s*[\u2013\u2014\-,]\s*\d+)*\s*\]")


# ----------------------------- helpers ----------------------------------------
def clean(s):
    return CTRL.sub(" ", s).replace("\u00ad", "")


def is_bold(span):
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def join_lines(lines):
    out = ""
    for l in lines:
        if not out:
            out = l
        elif out.endswith("-") and l[:1].islower():
            out = out[:-1] + l
        else:
            out += " " + l
    return re.sub(r"\s+", " ", out).strip()


def strip_cites(s):
    return re.sub(r"\s+", " ", CITE.sub("", s)).strip()


def block_lines(block):
    res = []
    for ln in block["lines"]:
        spans = [s for s in ln["spans"] if s["text"].strip()]
        if not spans:
            continue
        res.append({
            "text": clean("".join(s["text"] for s in ln["spans"])).strip(),
            "x0": ln["bbox"][0],
            "size": max(s["size"] for s in spans),
            "bold_chars": sum(len(s["text"]) for s in spans if is_bold(s)),
            "chars": sum(len(s["text"]) for s in spans),
            "first_span_bold_text": clean(spans[0]["text"]).strip() if is_bold(spans[0]) else "",
        })
    return res


def split_paragraphs(lines):
    base = min(l["x0"] for l in lines)
    paras, cur = [], []
    for i, l in enumerate(lines):
        if i > 0 and cur and (l["x0"] > base + 4 or l["text"].startswith("•")):
            paras.append(join_lines(cur))
            cur = []
        cur.append(l["text"])
    if cur:
        paras.append(join_lines(cur))
    return [p for p in paras if p]


def is_known(t):
    return t[:1].isupper() and t.lower().rstrip(":") in KNOWN_HEADINGS


def segment_block(lines):
    segs, buf, i = [], [], 0
    frac = lambda l: l["bold_chars"] / max(1, l["chars"])
    while i < len(lines):
        l = lines[i]
        t = l["text"]
        if frac(l) >= 0.9 and (NUM_START.match(t) or is_known(t)):
            parts, j = [t], i + 1
            while (j < len(lines) and len(parts) < 3 and frac(lines[j]) >= 0.9
                   and abs(lines[j]["size"] - l["size"]) < 0.6 and not NUM_START.match(lines[j]["text"])):
                parts.append(lines[j]["text"])
                j += 1
            if buf:
                segs.append(("text", buf))
                buf = []
            segs.append(("heading", join_lines(parts)))
            i = j
        else:
            buf.append(l)
            i += 1
    if buf:
        segs.append(("text", buf))
    return segs


def body_font_size(doc):
    c = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    c[round(s["size"], 1)] += len(s["text"])
    return c.most_common(1)[0][0] if c else 10.0


def guess_title(page):
    spans = [s for b in page.get_text("dict")["blocks"] for ln in b.get("lines", [])
             for s in ln["spans"] if s["text"].strip()]
    if not spans:
        return "Untitled"
    mx = max(s["size"] for s in spans)
    return re.sub(r"\s+", " ", clean(" ".join(s["text"].strip() for s in spans if s["size"] >= mx - 0.5)))


def table_markdown(t):
    try:
        md = t.to_markdown()
        if md.strip():
            return md.strip()
    except Exception:
        pass
    return "\n".join("| " + " | ".join((c or "").replace("\n", " ") for c in r) + " |" for r in t.extract())


def good_table(t):
    try:
        rows = t.extract()
    except Exception:
        return False
    if len(rows) < 3 or not (2 <= len(rows[0]) <= 15):
        return False
    cells = [c for r in rows for c in r]
    return sum(bool(c and c.strip()) for c in cells) / max(1, len(cells)) > 0.4


def find_valid_tables(page):
    for kw in ({}, {"vertical_strategy": "text", "horizontal_strategy": "lines"}):
        try:
            ts = [t for t in page.find_tables(**kw).tables if good_table(t)]
        except Exception:
            continue
        if ts:
            return ts
    return []


def overlap_ratio(bbox, rect):
    r = fitz.Rect(bbox)
    inter = r & rect
    return 0 if r.is_empty or inter.is_empty else inter.get_area() / r.get_area()


# ----------------------------- PDF -> sections --------------------------------
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    body = body_font_size(doc)
    title = guess_title(doc[0])

    sections, stack, cur = [], [], None
    started = stopped = False

    for pno, page in enumerate(doc, 1):
        if stopped:
            break
        H = page.rect.height
        tabs = [{"rect": fitz.Rect(t.bbox), "md": table_markdown(t), "el": None}
                for t in find_valid_tables(page)]
        tcaps = []   # table captions seen on this page

        def emit(kind, text):
            if cur is not None and not cur["skip"] and text.strip():
                cur["elements"].append({"type": kind, "text": text, "p0": pno, "p1": pno})

        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            x0, y0, x1, y1 = b["bbox"]
            if y1 < H * TOP_MARGIN or y0 > H * BOTTOM_MARGIN:
                continue
            lines = block_lines(b)
            if pno == 1:   # drop affiliation / footnote lines even when glued to body text
                lines = [l for l in lines if l["size"] >= body - 1.5 or CAPTION_RE.match(l["text"])]
            if not lines:
                continue
            text = join_lines([l["text"] for l in lines])
            if NOISE_RE.match(text) or ("@" in text and len(text) < 400):
                continue

            # table body text -> emit the whole table once
            hit = next((t for t in tabs if overlap_ratio(b["bbox"], t["rect"]) > 0.6), None)
            if hit is not None:
                if hit["el"] is None and cur is not None and not cur["skip"]:
                    hit["el"] = {"type": "table", "text": hit["md"], "p0": pno, "p1": pno}
                    cur["elements"].append(hit["el"])
                continue

            first_bold = lines[0]["first_span_bold_text"]
            plain = re.sub(r"^\d+(\.\d+)*\s+", "", text.lower())
            if started and pno > 1 and plain.startswith(STOP_PREFIXES) and (first_bold or len(text) < 60):
                stopped = True
                break

            for kind, payload in segment_block(lines):
                if kind == "heading":
                    m = NUM_HEADING.match(payload)
                    if not (m or is_known(payload)):
                        continue
                    level = m.group(1).count(".") + 1 if m else 1
                    name = payload if m else payload.rstrip(":")
                    started = True
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, name))
                    cur = {"path": [t for _, t in stack], "elements": [],
                           "skip": re.sub(r"^\d+(\.\d+)*\s+", "", name).lower() in SKIP_SECTIONS}
                    sections.append(cur)
                    continue

                if not started or cur is None:
                    continue
                seg_text = join_lines([l["text"] for l in payload])

                # real captions start with a bold "Fig. N" / "Table N" label
                if CAPTION_RE.match(seg_text) and payload[0]["first_span_bold_text"]:
                    if seg_text.lower().startswith("table") and not cur["skip"]:
                        el = {"type": "tcap", "text": seg_text, "p0": pno, "p1": pno}
                        cur["elements"].append(el)
                        tcaps.append({"y": y0, "el": el, "used": False})
                    continue                      # figure captions are dropped

                for para in split_paragraphs(payload):
                    if not DROP_PARA.match(para):
                        emit("para", para)

        # pair tables with their captions; tables whose text was never seen are added here
        for t in tabs:
            if t["el"] is None:
                if cur is not None and not cur["skip"]:
                    t["el"] = {"type": "table", "text": t["md"], "p0": pno, "p1": pno}
                    cur["elements"].append(t["el"])
                else:
                    continue
            best = None
            for c in tcaps:
                d = min(abs(c["y"] - t["rect"].y0), abs(c["y"] - t["rect"].y1))
                if not c["used"] and d < 150 and (best is None or d < best[0]):
                    best = (d, c)
            if best:
                best[1]["used"] = True
                best[1]["el"]["type"] = "drop"
                t["el"]["text"] = best[1]["el"]["text"] + "\n\n" + t["md"]
        for c in tcaps:
            if not c["used"]:
                c["el"]["type"] = "para"          # no table found: keep caption text as info
    return title, sections


# ----------------------------- sections -> chunks -----------------------------
def wc(s):
    return len(s.split())


def stitch_paragraphs(elements):
    out = []
    for e in (dict(e) for e in elements if e["type"] == "para"):
        if out and not re.search(r"[.!?:;][\)\]\"”’]?$", out[-1]["text"]) and e["text"][:1].islower():
            out[-1]["text"] += " " + e["text"]
        else:
            out.append(e)
    for e in out:
        e["text"] = strip_cites(e["text"])
    return [e for e in out if e["text"]]


def split_long(para):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", para["text"])
    pieces, cur = [], []
    for s in sents:
        if cur and wc(" ".join(cur + [s])) > MAX_WORDS:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return [{**para, "text": p} for p in pieces]


def pack(paras):
    units = []
    for p in paras:
        units.extend(split_long(p) if wc(p["text"]) > MAX_WORDS else [p])
    groups, cur, n = [], [], 0
    for u in units:
        w = wc(u["text"])
        if cur and n + w > MAX_WORDS:
            groups.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        groups.append(cur)
    if len(groups) > 1 and sum(wc(u["text"]) for u in groups[-1]) < MIN_WORDS:
        merged = groups[-2] + groups[-1]
        if sum(wc(u["text"]) for u in merged) <= MAX_WORDS * HARD_CAP:
            groups[-2:] = [merged]
    return groups


def split_table(text):
    if wc(text) <= MAX_WORDS * HARD_CAP:
        return [text]
    lines = text.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith("|")), 0)
    head, rows = lines[:idx + 2], lines[idx + 2:]
    parts, cur = [], []
    for r in rows:
        if cur and wc("\n".join(head + cur + [r])) > MAX_WORDS:
            parts.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def build_chunks(title, sections, source):
    units = []
    for s in sections:
        if s["skip"]:
            continue
        paras = stitch_paragraphs(s["elements"])
        tables = [e["text"] for e in s["elements"] if e["type"] == "table" and e["text"].strip()]
        if paras or tables:
            units.append({"path": s["path"], "paras": paras, "tables": tables})

    if MERGE_SMALL:   # merge whole tiny text-only sections into the next sibling
        merged, i = [], 0
        while i < len(units):
            u = units[i]
            w = sum(wc(p["text"]) for p in u["paras"])
            if (u["paras"] and not u["tables"] and w < MIN_WORDS and i + 1 < len(units)
                    and units[i + 1]["path"][0] == u["path"][0]
                    and w + sum(wc(p["text"]) for p in units[i + 1]["paras"]) <= MAX_WORDS):
                nxt = units[i + 1]
                head = {"type": "para", "text": nxt["path"][-1]}
                units[i + 1] = {"path": u["path"], "paras": u["paras"] + [head] + nxt["paras"],
                                "tables": nxt["tables"]}
                i += 1
                continue
            merged.append(u)
            i += 1
        units = merged

    chunks = []

    def add(kind, path, text):
        chunks.append({
            "text": f"Paper: {title}\nSection: {' > '.join(path)}\n\n{text}",
            "meta": {"source": source, "title": title, "section": " > ".join(path),
                     "chunk_type": kind},
        })

    for u in units:
        for g in pack(u["paras"]):
            add("text", u["path"], "\n\n".join(p["text"] for p in g))
        for tb in u["tables"]:
            for part in split_table(tb):
                add("table", u["path"], part)

    for i, c in enumerate(chunks):
        c["id"] = f"{Path(source).stem}::{i:04d}"
    return chunks


# ----------------------------- ChromaDB ---------------------------------------
def get_collection(db_path, name, model):
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    return client.get_or_create_collection(name=name, embedding_function=ef,
                                           metadata={"hnsw:space": "cosine"})


def store(chunks, col, batch=64):
    for src in {c["meta"]["source"] for c in chunks}:   # remove stale chunks of re-ingested files
        col.delete(where={"source": src})
    for i in range(0, len(chunks), batch):
        b = chunks[i:i + batch]
        col.upsert(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                   metadatas=[c["meta"] for c in b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default=None)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--no_store", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    if a.query:
        col = get_collection(a.db, a.collection, a.model)
        r = col.query(query_texts=[a.query], n_results=a.k)
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            print(f"\n--- {meta['source']} | {meta['section']} | {meta['chunk_type']} | dist={dist:.3f}")
            print(doc[:600])
        return

    all_chunks = []
    for pdf in sorted(Path(a.pdf_dir).glob("*.pdf")):
        title, secs = extract_sections(str(pdf))
        ch = build_chunks(title, secs, pdf.name)
        tops = Counter(c["meta"]["section"].split(" > ")[0] for c in ch)
        print(f"{pdf.name}: {len(ch)} chunks | sections: {dict(tops)}")
        all_chunks += ch

    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps({"id": c["id"], **c["meta"], "text": c["text"]}, ensure_ascii=False) + "\n")
    if not a.no_store:
        store(all_chunks, get_collection(a.db, a.collection, a.model))
        print(f"Stored {len(all_chunks)} chunks in '{a.collection}' at {a.db}")


if __name__ == "__main__":
    main()
