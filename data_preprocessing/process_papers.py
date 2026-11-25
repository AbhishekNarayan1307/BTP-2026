import json
from pathlib import Path
import re
import fitz  # pip install pymupdf

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ECE_DATA_DIR = PROJECT_ROOT / "Data" / "rawdata" / "ece_data"

PAPERS = [
    {
        "pdf_path": ECE_DATA_DIR / "s40964-025-01386-w.pdf",
        "paper_id": "s40964-025-01386-w",
        "title": (
            "Increasing isotropy and predictability through electroplating of"
            " tensile properties of polyamide 12 parts manufactured by selective"
            " laser sintering"
        ),
        "domain": "Additive Manufacturing / Materials",
        "output_json": BASE_DIR / "paper_sls_pa12_chunks.json",
    },
    {
        "pdf_path": ECE_DATA_DIR / "s12567-023-00526-8.pdf",
        "paper_id": "s12567-023-00526-8",
        "title": (
            "Analysis of the thermal design of a COTS-based modular battery"
            " system for satellites by thermal vacuum testing"
        ),
        "domain": "Spacecraft / Battery Thermal Management",
        "output_json": BASE_DIR / "paper_satellite_bms_chunks.json",
    },
    {
        "pdf_path": ECE_DATA_DIR / "s12200-025-00158-2.pdf",
        "paper_id": "s12200-025-00158-2",
        "title": (
            "Recent advances in monolithic-integrated lead-based optoelectronic"
            " devices"
        ),
        "domain": "Optoelectronics / Perovskites & CQDs",
        "output_json": BASE_DIR / "paper_optoelectronics_chunks.json",
    },
    {
        "pdf_path": ECE_DATA_DIR / "s42452-022-05216-4.pdf",
        "paper_id": "s42452-022-05216-4",
        "title": (
            "Smooth frequency change effects in electrical circuits: explained by"
            " the Fresnel integral"
        ),
        "domain": "Electrical Engineering / Circuit Dynamics",
        "output_json": BASE_DIR / "paper_fresnel_circuits_chunks.json",
    },
]

RUNNING_NOISE_PATTERNS = [
    r"Vol\.\:\(0123456789\)",
    r"Progress in Additive Manufacturing\s*\(\d{4}\)\s*\d+:\d+[–-]\d+",
    r"CEAS Space Journal\s*\(\d{4}\)\s*\d+:\d+[–-]\d+",
    r"Frontiers of Optoelectronics\s*\(\d{4}\)\s*\d+:\d+",
    r"SN Applied Sciences\s*\(\d{4}\)\s*\d+:\d+",
    r"Agneter et al\.\s*Photonix\s*\(\d{4}\)\s*\d+:\d+",
    r"M\.\s*Eilenberger et al\.",
    r"Analysis of the thermal design of a COTS-based modular battery system for satellites by thermal\.{0,3}",
    r"Page\s+\d+\s+of\s+\d+",
    r"https?://doi\.org/\S+",
    r"SN Applied Sciences\s*A SPRINGER NATURE journal",
    r"\bResearch Article\b",
    r"\bSpringer(?:\s+Open|\s+Nature)?\b",
]

MATH_GLYPH_MAP = {
    "훥": "Delta_",
    "휏": "tau",
    "휔": "omega",
    "휌": "rho",
    "훺": "Omega",
    "휋": "pi",
    "휐": "phi",
    "퐯": "v",
    "퐫": "r",
    "퐜": "c",
    "\u0007": " ",
    "\u00a0": " ",
    "\u2003": " ",
}

# Matches numbered section/subsection headings or standalone major headings
HEADING_PATTERN = re.compile(
    r"(?:^|\n)[^\S\r\n]*"
    r"(?:"
    r"(?:[1-9]|1[0-5])(?:\.[1-9][0-9]?){0,2}\s+[A-Z][A-Za-z0-9,\-\s\(\)/:μΩ]{2,80}|"
    r"(?:Abstract|Article Highlights|Highlights|Keywords|Abbreviations|"
    r"Results|Discussion|Conclusions?|Conclusion|"
    r"Methods|Materials and methods|Acknowledgements?|"
    r"Funding(?:\s+sources)?|Declarations?|References)"
    r")(?=\s*\n)",
    re.MULTILINE,
)


def clean_paper_text(raw_text: str) -> str:
  text = raw_text
  for bad, good in MATH_GLYPH_MAP.items():
    text = text.replace(bad, good)

  # Strip non-printable control characters except \n and \t
  text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

  for pat in RUNNING_NOISE_PATTERNS:
    text = re.sub(pat, "", text, flags=re.IGNORECASE)

  text = re.sub(r"[^\S\r\n]+", " ", text)
  text = re.sub(r"\n\d{1,3}(?=\n|$)", "\n", text)
  text = re.sub(r"\n\|\s*(?=\n|$)", "\n", text)
  text = re.sub(r"([A-Za-z0-9])-\s*\n\s*([A-Za-z0-9])", r"\1\2", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def split_non_overlapping_paragraphs(text: str, max_chars: int = 3500) -> list[str]:
  """Splits text strictly into non-overlapping chunks at paragraph/sentence boundaries."""
  if len(text) <= max_chars:
    return [text.strip()]

  paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
  chunks = []
  current = ""

  for para in paragraphs:
    if len(para) > max_chars:
      if current:
        chunks.append(current.strip())
        current = ""
      sentences = re.split(r"(?<=[.!?])\s+", para)
      sent_buf = ""
      for s in sentences:
        if len(sent_buf) + len(s) + 1 > max_chars and sent_buf:
          chunks.append(sent_buf.strip())
          sent_buf = s
        else:
          sent_buf = f"{sent_buf} {s}".strip()
      if sent_buf:
        current = sent_buf
    else:
      candidate = f"{current}\n\n{para}".strip() if current else para
      if len(candidate) > max_chars and current:
        chunks.append(current.strip())
        current = para
      else:
        current = candidate

  if current.strip():
    chunks.append(current.strip())

  return chunks


def extract_paper_chunks(paper_cfg: dict) -> list[dict]:
  doc = fitz.open(str(paper_cfg["pdf_path"]))
  page_texts = []

  for page_num, page in enumerate(doc):
    rect = page.rect
    clip = fitz.Rect(
        rect.width * 0.03,
        rect.height * 0.04,
        rect.width * 0.97,
        rect.height * 0.96,
    )
    txt = page.get_text("text", clip=clip)
    page_texts.append((page_num + 1, clean_paper_text(txt)))

  merged_parts = [f"[[PAGE_{pno}]]\n{ptxt}" for pno, ptxt in page_texts if ptxt]
  full_doc = "\n\n".join(merged_parts)

  ref_cut = re.search(
      r"\n(?:References|Declarations)\s*\n", full_doc, flags=re.IGNORECASE
  )
  if ref_cut:
    full_doc = full_doc[: ref_cut.start()].strip()

  raw_matches = list(HEADING_PATTERN.finditer(full_doc))
  matches = []
  for m in raw_matches:
    h = re.sub(r"\s+", " ", m.group(0).strip())
    if re.match(r"^(?:Table|Fig\.|Figure)\s+\d+", h, flags=re.IGNORECASE):
      continue
    if any(sym in h for sym in ("=", "≈", "≅", "≤", "≥", "∫")) or h.startswith(
        "Delta_"
    ):
      continue
    matches.append((m, h))

  sections = []
  if not matches:
    sections.append(("Body", full_doc))
  else:
    front_matter = full_doc[: matches[0][0].start()].strip()
    if len(front_matter) > 100:
      sections.append(("Front Matter / Title", front_matter))

    for idx, (m, heading) in enumerate(matches):
      start = m.start()
      end = (
          matches[idx + 1][0].start() if idx + 1 < len(matches) else len(full_doc)
      )
      sec_block = full_doc[start:end].strip()
      if sec_block:
        sections.append((heading, sec_block))

  final_chunks = []
  for sec_label, sec_text in sections:
    pages_found = sorted(
        {int(x) for x in re.findall(r"\[\[PAGE_(\d+)\]\]", sec_text)}
    )
    page_str = (
        f"p.{pages_found[0]}"
        if len(pages_found) == 1
        else (f"pp.{pages_found[0]}–{pages_found[-1]}" if pages_found else "N/A")
    )
    clean_sec_text = re.sub(r"\n?\[\[PAGE_\d+\]\]\n?", "\n", sec_text).strip()

    sub_blocks = split_non_overlapping_paragraphs(clean_sec_text, max_chars=3500)
    for part_idx, block in enumerate(sub_blocks, 1):
      part_suffix = (
          f" (Part {part_idx}/{len(sub_blocks)})" if len(sub_blocks) > 1 else ""
      )
      sec_name = f"{sec_label}{part_suffix}"
      header_meta = (
          f"[{paper_cfg['paper_id']} | {paper_cfg['domain']} | "
          f"{sec_name} | {page_str}]"
      )
      final_chunks.append({
          "paper_id": paper_cfg["paper_id"],
          "paper_title": paper_cfg["title"],
          "domain": paper_cfg["domain"],
          "section": sec_name,
          "pages": page_str,
          "text_for_embedding": f"{header_meta}\n{block}",
          "content": block,
      })

  return final_chunks


def main():
  all_paper_chunks = []
  for cfg in PAPERS:
    if not cfg["pdf_path"].exists():
      print(f"Skipping missing file: {cfg['pdf_path']}")
      continue
    chunks = extract_paper_chunks(cfg)
    with open(cfg["output_json"], "w", encoding="utf-8") as f:
      json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks):>2} chunks -> {cfg['output_json'].name}")
    all_paper_chunks.extend(chunks)

  combined_path = BASE_DIR / "all_research_papers_chunks.json"
  with open(combined_path, "w", encoding="utf-8") as f:
    json.dump(all_paper_chunks, f, indent=2, ensure_ascii=False)
  print(
      f"\nTotal research chunks across all papers: {len(all_paper_chunks)} ->"
      f" {combined_path.name}"
  )


if __name__ == "__main__":
  main()