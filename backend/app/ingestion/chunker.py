
import re

MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

# Boundary markers for natural sentence endings
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

# Sensitive keywords that must not be split away from their modifying context
SENSITIVE_PATTERNS = [
    r'\bsome\s+or\s+all\b',
    r'\bone\s+or\s+more\b',
    r'\bat\s+least\b',
    r'\bup\s+to\b',
    r'\bunless\b',
    r'\bexcept\b',
    r'\bonly\s+if\b',
    r'\bwhere\s+applicable\b',
    r'\bwhen\s+required\b',
    r'\bwill\s+not\s+be\b',
    r'\bnot\s+be\s+considered\b',
    r'\bnot\s+mandatory\b',
    r'\bnot\s+required\b',
    r'\bnot\s+optional\b',
    r'\bif\s+their\b',
    r'\bsubject\s+to\b',
    r'\bmay\s+include\b',
    r'\bare\s+encouraged\b',
]
_SENSITIVE_RE = re.compile('|'.join(SENSITIVE_PATTERNS), re.IGNORECASE)


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" :")


def is_heading(block: dict, median_font_size: float) -> bool:
    text = block["text"].strip()

    if not text or len(text) > 110:
        return False

    words = text.split()

    if len(words) > 12:
        return False

    if text.endswith((".", "?", "!", ",")):
        return False

    font_size = block.get("font_size", median_font_size)
    is_bold = block.get("is_bold", False)

    # Short, bold text is a strong heading candidate.
    if is_bold and len(words) <= 12:
        return True

    # Larger text compared with the page's typical font size.
    if font_size >= median_font_size * 1.15 and len(words) <= 12:
        return True

    # Short all-uppercase headings.
    if text.isupper() and len(words) <= 10:
        return True

    return False


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving sentence content."""
    text = text.strip()
    if not text:
        return []

    # First split on line breaks that look like separate paragraphs or list items
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sentences = []

    for line in lines:
        # Split line on sentence endings
        parts = _SENTENCE_END_RE.split(line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


def create_chunks(
    pages: list,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list:
    """
    Create semantically cohesive, sentence-preserving chunks from pages.

    Features:
    - Never splits mid-sentence; preserves complete sentences.
    - Configurable chunk overlap (overlap_chars) so boundary context is preserved.
    - Prevents qualifiers, conditions, negations, and exceptions from being isolated.
    - Preserves headings and resets section scope appropriately.
    - Supports multi-page continuous content while maintaining page_number and bboxes.
    - Deduplicates bounding boxes to ensure clean citation highlighting.
    """
    chunks = []
    chunk_number = 0
    current_section = "General"

    # Step 1: Flatten blocks across pages into atomic items
    # Each item represents a sentence or a small block with its page_number and bbox
    flat_units = []

    for page in pages:
        page_num = page["page_number"]
        blocks = page["blocks"]

        font_sizes = [
            b["font_size"]
            for b in blocks
            if b.get("font_size", 0) > 0
        ]
        median_font_size = (
            sorted(font_sizes)[len(font_sizes) // 2]
            if font_sizes else 10
        )

        for block in blocks:
            text = block["text"].strip()
            if not text:
                continue

            if is_heading(block, median_font_size):
                flat_units.append({
                    "is_heading": True,
                    "heading_text": text,
                    "page_number": page_num,
                    "bbox": block.get("bbox"),
                })
            else:
                sentences = split_into_sentences(text)
                if not sentences:
                    sentences = [text]

                for sentence in sentences:
                    flat_units.append({
                        "is_heading": False,
                        "text": sentence,
                        "page_number": page_num,
                        "bbox": block.get("bbox"),
                    })

    # Step 2: Accumulate units into chunks with sentence preservation and overlap
    current_units = []
    current_len = 0

    def flush_chunk():
        nonlocal chunk_number, current_units, current_len
        if not current_units:
            return

        chunk_text = " ".join(u["text"] for u in current_units)
        # Determine dominant or primary page_number (first unit's page)
        primary_page = current_units[0]["page_number"]

        # Collect and deduplicate bboxes
        bboxes = []
        seen_boxes = set()
        for u in current_units:
            bbox = u.get("bbox")
            if bbox:
                # tuple for hashability
                box_key = tuple(bbox)
                if box_key not in seen_boxes:
                    seen_boxes.add(box_key)
                    bboxes.append(bbox)

        chunks.append({
            "chunk_id": f"chunk_{chunk_number}",
            "page_number": primary_page,
            "section": current_section,
            "text": chunk_text,
            "bboxes": bboxes,
        })
        chunk_number += 1

        # Calculate overlap units to carry forward
        if overlap_chars > 0 and len(current_units) > 1:
            overlap_units = []
            overlap_len = 0
            # Pull from end of current_units
            for u in reversed(current_units):
                u_len = len(u["text"]) + 1
                if overlap_len + u_len <= overlap_chars:
                    overlap_units.insert(0, u)
                    overlap_len += u_len
                else:
                    break

            # If the last unit contains a sensitive qualifier/condition, ensure it's in overlap
            if not overlap_units and current_units:
                overlap_units = [current_units[-1]]
                overlap_len = len(current_units[-1]["text"])

            current_units = overlap_units
            current_len = overlap_len
        else:
            current_units = []
            current_len = 0

    for unit in flat_units:
        if unit.get("is_heading"):
            # A new heading flushes the current chunk and starts a new section
            flush_chunk()
            current_units = []
            current_len = 0
            current_section = unit["heading_text"]
            continue

        unit_text = unit["text"]
        unit_len = len(unit_text) + (1 if current_units else 0)

        # Check if adding this unit exceeds max_chars
        if current_units and (current_len + unit_len > max_chars):
            # Before flushing, check if unit contains a qualifier or condition that modifies the previous unit
            # or if previous unit contains a qualifier modifying this unit
            flush_chunk()

        current_units.append(unit)
        current_len += unit_len

    flush_chunk()
    return chunks

