
import re

from .config import MAX_CHUNK_CHARS


SECTION_RE = re.compile(
    r"^\s*(\d+[A-Z]?)\.\s+(.*)$",
    re.IGNORECASE
)

CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+)\b",
    re.IGNORECASE
)

SUBSECTION_RE = re.compile(
    r"^\s*\((\d+)\)\s*(.*)$"
)

SUPPLEMENT_RE = re.compile(
    r"^\s*(Explanation|Illustration|Provided that|"
    r"Provided further that)\b",
    re.IGNORECASE
)


def prepare_lines(pages, strategy):

    lines = []

    # The Motor Vehicles Act has an arrangement of
    # sections before the actual statutory text.
    started = strategy != "motor"

    for page_data in pages:

        page = page_data["page"]

        for line in page_data["lines"]:

            if not started:

                if re.search(
                    r"\bBE it enacted\b",
                    line,
                    re.IGNORECASE
                ):
                    started = True

                continue

            # Ignore table-of-contents entries.
            if strategy == "motor" and re.match(
                r"^SECTIONS$",
                line,
                re.IGNORECASE
            ):
                continue

            lines.append({
                "text": line,
                "page": page
            })

    return lines


def identify_units(lines, strategy):

    sections = []

    current_chapter = "Unknown"
    current_section = None

    for item in lines:

        text = item["text"]

        chapter_match = CHAPTER_RE.match(text)

        if chapter_match:
            current_chapter = (
                "CHAPTER " + chapter_match.group(1).upper()
            )
            continue

        section_match = SECTION_RE.match(text)

        if section_match:

            section_number = section_match.group(1)
            remainder = section_match.group(2).strip()

            # Start a new section.
            current_section = {
                "section": section_number,
                "chapter": current_chapter,
                "pages": [item["page"]],
                "lines": [],
                "heading": ""
            }

            if remainder:
                current_section["lines"].append({
                    "text": remainder,
                    "page": item["page"]
                })

            sections.append(current_section)

            continue

        if current_section is not None:

            current_section["lines"].append(item)

            if item["page"] not in current_section["pages"]:
                current_section["pages"].append(item["page"])

    return sections


def split_into_subsections(section, strategy):

    lines = section["lines"]

    units = []
    current_unit = None

    for item in lines:

        text = item["text"]

        match = SUBSECTION_RE.match(text)

        # DPDP illustrations and explanations belong
        # to the preceding subsection.
        # Consumer and Motor use the same attachment rule.
        if match:

            if current_unit:
                units.append(current_unit)

            current_unit = {
                "subsection": match.group(1),
                "lines": [item],
                "pages": [item["page"]]
            }

        elif current_unit is not None:

            current_unit["lines"].append(item)

            if item["page"] not in current_unit["pages"]:
                current_unit["pages"].append(item["page"])

        else:

            # Text before the first numbered subsection.
            # Preserve it as a section-level unit.
            if not units:

                current_unit = {
                    "subsection": "General",
                    "lines": [item],
                    "pages": [item["page"]]
                }

            else:
                units[-1]["lines"].append(item)

    if current_unit:
        units.append(current_unit)

    return units


def split_large_unit(text, max_size=MAX_CHUNK_CHARS):

    if len(text) <= max_size:
        return [text]

    # Split at legal clause boundaries first.
    # This is a fallback for oversized subsections.
    pattern = r"(?m)(?=^\s*\([a-z]\)\s*)"

    parts = re.split(pattern, text)

    chunks = []
    current = ""

    for part in parts:

        if not part.strip():
            continue

        if current and len(current) + len(part) > max_size:
            chunks.append(current.strip())
            current = ""

        if len(part) > max_size:

            if current:
                chunks.append(current.strip())
                current = ""

            # Last resort: sentence-level splitting.
            sentences = re.split(
                r"(?<=[.!?])\s+",
                part
            )

            for sentence in sentences:

                if current and len(current) + len(sentence) > max_size:
                    chunks.append(current.strip())
                    current = ""

                current = (current + " " + sentence).strip()

        else:
            current += "\n" + part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_section_chunks(sections, strategy):

    final_chunks = []

    for section in sections:

        units = split_into_subsections(
            section,
            strategy
        )

        for unit in units:

            content = "\n".join(
                item["text"]
                for item in unit["lines"]
            ).strip()

            if not content:
                continue

            parts = split_large_unit(content)

            for part_index, part in enumerate(parts, start=1):

                final_chunks.append({
                    "chapter": section["chapter"],
                    "section": section["section"],
                    "subsection": unit["subsection"],
                    "page_start": min(unit["pages"]),
                    "page_end": max(unit["pages"]),
                    "part": part_index,
                    "total_parts": len(parts),
                    "text": part
                })

    return final_chunks


def chunk_consumer_protection(pages):

    lines = prepare_lines(pages, "consumer")
    sections = identify_units(lines, "consumer")

    return build_section_chunks(
        sections,
        "consumer"
    )


def chunk_dpdp(pages):

    lines = prepare_lines(pages, "dpdp")
    sections = identify_units(lines, "dpdp")

    return build_section_chunks(
        sections,
        "dpdp"
    )


def chunk_motor_vehicles(pages):

    lines = prepare_lines(pages, "motor")
    sections = identify_units(lines, "motor")

    return build_section_chunks(
        sections,
        "motor"
    )


CHUNKING_STRATEGIES = {
    "consumer": chunk_consumer_protection,
    "dpdp": chunk_dpdp,
    "motor": chunk_motor_vehicles
}
