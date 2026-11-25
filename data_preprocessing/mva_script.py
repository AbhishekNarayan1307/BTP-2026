import json
import re
import fitz  # pip install pymupdf


def clean_mva_text(text):
  # 1. Remove standalone page numbers (e.g. "\n 12 \n")
  text = re.sub(r"\n\s*\d{1,3}\s*(?=\n|$)", "", text)
  # 2. Remove omitted-clause stars ("1* * * * *")
  text = re.sub(r"\n\s*\d?\*\s+(?:\*\s*)+", "", text)
  # 3. Fix hyphenated line-breaks and collapse extra whitespace/newlines
  text = re.sub(r"-\n\s*", "", text)
  text = re.sub(r"[ \t]+\n", "\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def split_large_section(content, max_chars=2500):
  if len(content) <= max_chars:
    return [content]

  parts = re.split(r"\n(?=\d?\[?\((?:[ivx]+|[a-z]{1,2}|\d{1,2})\)\s)", content)
  header = parts[0]
  sub_chunks = []
  current = header

  for part in parts[1:]:
    if len(current) + len(part) > max_chars and current != header:
      sub_chunks.append(current.strip())
      current = f"{header} [contd.]\n{part}"
    else:
      current += "\n" + part

  if current.strip():
    sub_chunks.append(current.strip())
  return sub_chunks


def parse_motor_vehicles_act(pdf_path, act_name="Motor Vehicles Act, 1988"):
  doc = fitz.open(pdf_path)
  full_text = []

  # Skip pages 1-7 (Arrangement of Sections TOC); stop at page 99 (before First Schedule traffic sign images)
  for page_num in range(7, 99):
    page = doc[page_num]
    rect = page.rect
    body_rect = fitz.Rect(0, rect.height * 0.05, rect.width, rect.height * 0.93)
    text = page.get_text("text", clip=body_rect)

    # Strip bottom amendment footnotes (handles both "1. Subs." and "1.Subs." and multi-line footnotes)
    text = re.sub(
        r"\n\s*\d+\.\s*(?:1st\s+July|Subs\.|Ins\.|Added|Omitted|The\s+(?:word|proviso|Schedule)|Sub-section|Clause|Section\s+\d+\s+renumbered)[\s\S]*$",
        "",
        text,
    )
    full_text.append(text)

  combined = clean_mva_text("\n".join(full_text))

  # Strip inline footnote line left in Section 71 on Page 45
  combined = re.sub(r"\n1\.\s+Proviso omitted by Act[^\n]+", "", combined)

  # Normalize inline CHAPTER VIII header so trailing/leading chapter regex catches it cleanly
  combined = re.sub(
      r"\s*CHAPTER VIII\s*\n\s*CONTROL OF TRAFFIC",
      "\nCHAPTER VIII\nCONTROL OF TRAFFIC",
      combined,
  )

  chunks = []
  current_chapter = "CHAPTER I - PRELIMINARY"

  section_splits = re.split(
      r"\n(?=(?:\d+\[)?\d{1,3}[A-Z]?\.\s+[A-Z])", combined
  )

  for block in section_splits:
    block = block.strip()
    if not block:
      continue

    next_chapter = current_chapter
    trailing_chap = re.search(r"\n(CHAPTER\s+[IVX]+\s*\n[^\n]+)\s*$", block)
    if trailing_chap:
      next_chapter = (
          re.sub(r"\s+", " ", trailing_chap.group(1).replace("\n", " - "))
          .strip()
      )
      block = block[: trailing_chap.start()].strip()

    leading_chap = re.match(r"^(CHAPTER\s+[IVX]+\s*\n[^\n]+)\n", block)
    if leading_chap:
      current_chapter = (
          re.sub(r"\s+", " ", leading_chap.group(1).replace("\n", " - "))
          .strip()
      )
      block = block[leading_chap.end() :].strip()

    if re.match(r"^2\.\s+Definitions", block):
      definitions = re.split(r'\n(?=(?:\d+\[)?\(\d{1,2}[A-Z]?\)\s+[“"])', block)
      for d in definitions:
        d = d.strip()
        if d.startswith("2."):
          continue
        def_num = re.match(r"^(?:\d+\[)?\((\d{1,2}[A-Z]?)\)", d)
        label = f"Section 2({def_num.group(1)})" if def_num else "Section 2"
        for sub in split_large_section(d):
          chunks.append({
              "act": act_name,
              "chapter": current_chapter,
              "section": label,
              "text_for_embedding": (
                  f"[{act_name} | {current_chapter} | {label}]\n{sub}"
              ),
              "content": sub,
          })
    else:
      sec_match = re.match(r"^(?:\d+\[)?(\d{1,3}[A-Z]?)\.", block)
      if not sec_match:
        continue
      sec_label = f"Section {sec_match.group(1)}"
      for sub in split_large_section(block):
        chunks.append({
            "act": act_name,
            "chapter": current_chapter,
            "section": sec_label,
            "text_for_embedding": (
                f"[{act_name} | {current_chapter} | {sec_label}]\n{sub}"
            ),
            "content": sub,
        })

    current_chapter = next_chapter

  # Extract Pages 110-111 (THE SECOND SCHEDULE - Structured Compensation Formula) cleanly
  sched_pages = []
  for p in range(109, len(doc)):
    t = doc[p].get_text("text")
    t = re.sub(
        r"\n1\.\s+Ins\.\s+by\s+Act[\s\S]*?\(w\.e\.f\.\s+14-11-1994\)\.", "", t
    )
    sched_pages.append(clean_mva_text(t))
  second_sched = "\n".join(sched_pages).strip()
  second_sched = re.sub(
      r"^1\s*\n\[THE SECOND SCHEDULE", "THE SECOND SCHEDULE", second_sched
  )

  chunks.append({
      "act": act_name,
      "chapter": "THE SECOND SCHEDULE",
      "section": "Second Schedule (Compensation)",
      "text_for_embedding": (
          f"[{act_name} | THE SECOND SCHEDULE]\n{second_sched}"
      ),
      "content": second_sched,
  })

  return chunks


if __name__ == "__main__":
  mva_chunks = parse_motor_vehicles_act(
      "../Data/rawdata/Legal_data/motor-vehicle-act.pdf"
  )
  with open("mva_1988_rag_final.json", "w", encoding="utf-8") as f:
    json.dump(mva_chunks, f, indent=2, ensure_ascii=False)

  print(f"Saved {len(mva_chunks)} clean MVA chunks -> mva_1988_rag_final.json")