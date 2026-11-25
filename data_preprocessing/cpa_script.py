import json
import re
import fitz  # pip install pymupdf


def extract_clean_text(pdf_path):
  doc = fitz.open(pdf_path)
  full_text = []

  for page_num, page in enumerate(doc):
    rect = page.rect
    main_body = fitz.Rect(
        rect.width * 0.18,
        rect.height * 0.09,
        rect.width * 0.82,
        rect.height * 0.92,
    )
    text = page.get_text("text", clip=main_body)

    if page_num == 0:
      text = re.split(r"MINISTRY OF LAW AND JUSTICE", text)[0]

    full_text.append(text)

  combined = "\n".join(full_text)
  combined = re.sub(
      r"\n?\s*(?:\d+\s+)?THE GAZETTE OF INDIA EXTRAORDINARY\s*(?:\[PART II—|SEC\.\s*1\])?\s*(?:\d+)?\s*\n?",
      "\n",
      combined,
  )
  combined = re.sub(r"\n\s*(?:\[PART II—|SEC\.\s*1\]|\[|\])\s*(?=\n|$)", "", combined)
  # Remove vertical runs of 3+ single-character lines (clipped margin letters)
  combined = re.sub(r"(?:\n[A-Za-z0-9,\.]){3,}(?=\n|$)", "", combined)
  # Fix line-break hyphenations
  combined = re.sub(r"-\n\s*", "", combined)
  combined = re.sub(r"\n{3,}", "\n\n", combined)
  return combined.strip()


def split_large_section(content, max_chars=2500):
  if len(content) <= max_chars:
    return [content]

  parts = re.split(r"\n(?=\((?:[ivx]+|[a-z]{1,2})\)\s)", content)
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


def chunk_gazette_act(text, act_name="Consumer Protection Act, 2019"):
  chunks = []
  current_chapter = "CHAPTER I - PRELIMINARY"

  section_splits = re.split(r"\n(?=\d{1,3}\.\s+(?:\(1\)|[A-Z]))", text)

  for block in section_splits:
    block = block.strip()
    if not block:
      continue

    next_chapter = current_chapter
    trailing_chap = re.search(r"\n(CHAPTER\s+[IVX]+\s*\n[^\n]+)\s*$", block)
    if trailing_chap:
      next_chapter = trailing_chap.group(1).replace("\n", " - ").strip()
      block = block[: trailing_chap.start()].strip()

    leading_chap = re.match(r"^(CHAPTER\s+[IVX]+\s*\n[^\n]+)\n", block)
    if leading_chap:
      current_chapter = leading_chap.group(1).replace("\n", " - ").strip()
      block = block[leading_chap.end() :].strip()

    if re.match(r"^2\.\s+In this Act", block):
      definitions = re.split(r'\n(?=\(\d{1,2}\)\s+[“"])', block)
      for d in definitions:
        d = d.strip()
        if d.startswith("2."):
          continue
        def_num = re.match(r"^\((\d{1,2})\)", d)
        label = f"Section 2({def_num.group(1)})" if def_num else "Section 2"
        for sub in split_large_section(d):
          chunks.append({
              "act": act_name,
              "chapter": current_chapter,
              "section": label,
              "text_for_embedding": f"[{act_name} | {current_chapter} | {label}]\n{sub}",
              "content": sub,
          })
    else:
      sec_match = re.match(r"^(\d{1,3})\.", block)
      sec_label = f"Section {sec_match.group(1)}" if sec_match else "Preamble"

      if sec_label == "Preamble":
        continue
      block = re.sub(r"\n————[\s\S]*$", "", block).strip()

      for sub in split_large_section(block):
        chunks.append({
            "act": act_name,
            "chapter": current_chapter,
            "section": sec_label,
            "text_for_embedding": f"[{act_name} | {current_chapter} | {sec_label}]\n{sub}",
            "content": sub,
        })

    current_chapter = next_chapter

  return chunks


if __name__ == "__main__":
  raw_text = extract_clean_text(
      "../Data/rawdata/Legal_data/consumer_protection_act_2019_gazette.pdf"
  )
  chunks = chunk_gazette_act(raw_text)

  with open("cpa_2019_rag_final.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

  print(f"Saved {len(chunks)} clean CPA chunks -> cpa_2019_rag_final.json")