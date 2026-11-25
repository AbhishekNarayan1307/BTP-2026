import json
import re
import fitz  # pip install pymupdf


def clean_dpdp_noise(text):
  # Remove Gazette running headers & SEC. 1 / SEC. 11 artifacts
  text = re.sub(
      r"\n?\s*(?:\d+\s+)?THE GAZETTE OF INDIA EXTRAORDINARY\s*(?:\[PART II—|SEC\.\s*11?\])?\s*(?:\d+)?\s*\n?",
      "\n",
      text,
  )
  text = re.sub(r"\n\s*(?:\[PART II—|SEC\.\s*11?|\[|\])\s*(?=\n|$)", "", text)
  # Remove Act citation margin numbers like "24 of 1997.", "21 of 200", or "of 1997."
  text = re.sub(r"\n(?:of\s+)?\d{2,4}\.?\s*(?=\n|$)", "", text)
  text = re.sub(r"\n\d{1,3}\s+of\s+\d{2,4}\.?\s*(?=\n|$)", "", text)
  # Remove vertical runs of clipped margin-note lines (2+ consecutive lines <= 12 chars)
  text = re.sub(r"(?:\n[A-Za-z][A-Za-z0-9,\.\s]{0,11}){2,}(?=\n|$)", "", text)
  # Fix hyphenated line-breaks
  text = re.sub(r"-\n\s*", "", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def split_large_section(content, max_chars=2500):
  if len(content) <= max_chars:
    return [content]

  parts = re.split(r"\n(?=\((?:[ivx]+|[a-z]{1,2}|\d{1,2})\)\s)", content)
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


def parse_dpdp_act(
    pdf_path, act_name="Digital Personal Data Protection Act, 2023"
):
  doc = fitz.open(pdf_path)
  full_text = []

  for page_num in range(len(doc) - 1):
    page = doc[page_num]
    rect = page.rect
    clip_rect = fitz.Rect(
        rect.width * 0.12,
        rect.height * 0.08,
        rect.width * 0.88,
        rect.height * 0.92,
    )
    text = page.get_text("text", clip=clip_rect)

    if page_num == 0:
      text = re.split(r"MINISTRY OF LAW AND JUSTICE", text)[0]

    full_text.append(text)

  combined = clean_dpdp_noise("\n".join(full_text))

  # Restore any section numbers whose digits were clipped at the margin edge
  combined = re.sub(r"\n\.\s+In this Act,", "\n2. In this Act,", combined)
  combined = re.sub(
      r"\n\.\s+The Chairperson shall exercise",
      "\n26. The Chairperson shall exercise",
      combined,
  )
  combined = re.sub(
      r"\n\.\s+\(1\)\s+The provisions of this Act shall be in addition",
      "\n38. (1) The provisions of this Act shall be in addition",
      combined,
  )

  chunks = []
  current_chapter = "CHAPTER I - PRELIMINARY"
  section_splits = re.split(r"\n(?=\d{1,3}\.\s+(?:\(1\)|[A-Z]))", combined)

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
      definitions = re.split(r'\n(?=\([a-z]{1,2}\)\s+[“"])', block)
      for d in definitions:
        d = d.strip()
        if d.startswith("2."):
          continue
        def_id = re.match(r"^\(([a-z]{1,2})\)", d)
        label = f"Section 2({def_id.group(1)})" if def_id else "Section 2"
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
      sec_match = re.match(r"^(\d{1,3})\.", block)
      if not sec_match:
        continue
      sec_label = f"Section {sec_match.group(1)}"

      # Strip Gazette masthead lines at the bottom of Section 1
      if sec_label == "Section 1":
        block = re.split(r"\nShort title", block)[0].strip()

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

  schedule_content = """THE SCHEDULE [See section 33(1)] — Monetary Penalties:
1. Breach in observing the obligation of Data Fiduciary to take reasonable security safeguards to prevent personal data breach under sub-section (5) of section 8: May extend to two hundred and fifty crore rupees.
2. Breach in observing the obligation to give the Board or affected Data Principal notice of a personal data breach under sub-section (6) of section 8: May extend to two hundred crore rupees.
3. Breach in observance of additional obligations in relation to children under section 9: May extend to two hundred crore rupees.
4. Breach in observance of additional obligations of Significant Data Fiduciary under section 10: May extend to one hundred and fifty crore rupees.
5. Breach in observance of the duties under section 15: May extend to ten thousand rupees.
6. Breach of any term of voluntary undertaking accepted by the Board under section 32: Up to the extent applicable for the breach in respect of which the proceedings under section 28 were instituted.
7. Breach of any other provision of this Act or the rules made thereunder: May extend to fifty crore rupees."""

  chunks.append({
      "act": act_name,
      "chapter": "THE SCHEDULE",
      "section": "Schedule (Penalties)",
      "text_for_embedding": f"[{act_name} | THE SCHEDULE]\n{schedule_content}",
      "content": schedule_content,
  })

  return chunks


if __name__ == "__main__":
  dpdp_chunks = parse_dpdp_act(
      "../Data/rawdata/Legal_data/digital_personal_data.pdf"
  )
  with open("dpdp_2023_rag_final.json", "w", encoding="utf-8") as f:
    json.dump(dpdp_chunks, f, indent=2, ensure_ascii=False)

  print(f"Saved {len(dpdp_chunks)} clean DPDP chunks -> dpdp_2023_rag_final.json")