
import re
import fitz


def normalize_line(line):
    line = line.replace("\u00ad", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def extract_lines(pdf_path):

    pdf = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(pdf, start=1):

        # Preserve reading order as much as possible.
        text = page.get_text("text", sort=True)

        lines = [
            normalize_line(line)
            for line in text.splitlines()
        ]

        cleaned = []

        for line in lines:

            if not line:
                continue

            # Common Gazette page furniture.
            if re.match(
                r"^(SEC\.|SEC|THE GAZETTE OF INDIA)",
                line,
                re.IGNORECASE
            ):
                continue

            if re.fullmatch(r"\d{1,4}", line):
                continue

            # Ignore Hindi transliteration fragments.
            if re.search(r"[\u0900-\u097F]", line):
                continue

            if re.match(
                r"^(REGISTERED NO\.|PUBLISHED BY AUTHORITY)",
                line,
                re.IGNORECASE
            ):
                continue

            cleaned.append(line)

        pages.append({
            "page": page_number,
            "lines": cleaned
        })

    pdf.close()

    return pages
