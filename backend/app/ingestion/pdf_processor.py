
import pymupdf


def extract_pdf(file_path: str) -> dict:
    """Extract text, layout properties and bounding boxes from a PDF."""

    pages = []

    with pymupdf.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_dict = page.get_text("dict")
            extracted_blocks = []

            for block in page_dict["blocks"]:
                if block.get("type") != 0:
                    continue

                lines = []
                font_sizes = []
                is_bold = False

                for line in block.get("lines", []):
                    line_text = []

                    for span in line.get("spans", []):
                        text = span["text"]

                        if text.strip():
                            line_text.append(text)
                            font_sizes.append(span["size"])

                            font_name = span.get("font", "").lower()

                            if "bold" in font_name:
                                is_bold = True

                    if line_text:
                        lines.append("".join(line_text))

                text = "\n".join(lines).strip()

                if not text:
                    continue

                bbox = block["bbox"]

                extracted_blocks.append({
                    "text": text,
                    "bbox": [round(value, 2) for value in bbox],
                    "font_size": round(max(font_sizes), 2),
                    "is_bold": is_bold
                })

            pages.append({
                "page_number": page_number,
                "blocks": extracted_blocks
            })

    return {
        "page_count": len(pages),
        "pages": pages
    }
