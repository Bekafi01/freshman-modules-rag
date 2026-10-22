"""PDF Ingestion Module for Freshman Modules RAG.

Extracts text, strips running headers/footers via edge frequency analysis,
normalizes character typography before NFKC, detects chapters and sections
before joining lines, splits text into heading-aligned segments, and saves
structured records to JSON Lines (JSONL) files.
"""

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Any
import unicodedata

import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Regular Expressions & Constants ---

QUOTE_FIXES = (
    ("‗‗", '"'),
    ("‘‘", '"'),
    ("‖", '"'),
    ("‗", "'"),
    ("―", "—"),
)

SUPERSCRIPT_MAP = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "ⁿ": "^n",
}

FIGURE_RE = re.compile(
    r"^(Figure|Fig\.)\s*(\d+(?:\.\d+)?)\s*[:\-]\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

EXERCISE_RE = re.compile(
    r"(Activity\s*#?\s*\d+|Self[\s\-]Check\s*Exercise|Review\s*Questions|Exercise\s*\d+)",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    r"^(?:CHAPTER|UNIT|Chapter|Unit)\s+(?:\d+|(?i:one|two|three|four|five|six|seven|eight|nine|ten))\b(?!\s+(?i:review|summary|summery|exercise|questions|is\b|deals\b|discusses\b|explained\b|describes\b|presents\b|shows\b|stated\b|defined\b))(?:\s*[:\-–]\s*|\s+)?(.*)$"
)

# Supports 1.1, 1.1.1, 1.1.3.1, Lesson 1:, and titles ending with question marks
SECTION_RE = re.compile(
    r"^(?:Lesson\s+\d+\b|[1-9]\d*(?:\.[1-9]\d*)+\.?)\s*[:.\-]?\s+[A-Z][^.!\n]{2,90}\??$"
)

STANDALONE_SEC_NUM = re.compile(r"^[1-9]\d*(?:\.[1-9]\d*)+$")
CONTINUATION_END = re.compile(r"(\band|\bor|\bof|\bthe|,|:)$", re.IGNORECASE)

DISALLOWED_SEC_START = re.compile(
    r"^(?:where|hence|therefore|since|because|then|if|when|let|here|for example|in this|as shown|note that)\b",
    re.IGNORECASE,
)
DISALLOWED_SEC_END = re.compile(r"\b(?:is|are|was|were|the|a|an|that|this)\s*$", re.IGNORECASE)

# Strict Roman numeral regex to avoid matching words like "civil", "mid", "dim", or "I"
ROMAN_RE = re.compile(
    r"^(?:Page\s*)?(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)


def find_running_lines(doc: pymupdf.Document, edge: int = 3, min_ratio: float = 0.25) -> set[str]:
    """Finds repeating header/footer lines across pages by masking digits with '#'."""
    counts = Counter()
    total_pages = len(doc)
    if total_pages == 0:
        return set()

    for page in doc:
        lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
        edge_lines = lines[:edge] + lines[-edge:]
        counts.update({re.sub(r"\d+", "#", l) for l in edge_lines})

    return {line for line, count in counts.items() if count / total_pages >= min_ratio}


def strip_page_headers_footers(
    lines: list[str],
    running_lines: set[str],
    edge: int = 3,
) -> tuple[list[str], int | str | None]:
    """Strips repeating header/footer lines and captures the printed page number."""
    printed_page: int | str | None = None

    # Identify top edge lines to drop
    top_drop_indices = set()
    top_count = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if top_count >= edge:
            break
        top_count += 1

        masked = re.sub(r"\d+", "#", s)
        is_running = masked in running_lines
        is_number = bool(re.match(r"^(?:Page\s*)?(\d+)$", s, re.IGNORECASE))
        is_roman = bool(ROMAN_RE.match(s))

        if is_running or is_number or is_roman:
            top_drop_indices.add(idx)
            if printed_page is None:
                m = re.search(r"\b(\d+)\b", s)
                if m:
                    printed_page = int(m.group(1))
                elif is_roman:
                    printed_page = s.lower().replace("page", "").strip()

    # Identify bottom edge lines to drop
    bottom_drop_indices = set()
    bottom_count = 0
    for idx in range(len(lines) - 1, -1, -1):
        s = lines[idx].strip()
        if not s:
            continue
        if bottom_count >= edge:
            break
        bottom_count += 1

        masked = re.sub(r"\d+", "#", s)
        is_running = masked in running_lines
        is_number = bool(re.match(r"^(?:Page\s*)?(\d+)$", s, re.IGNORECASE))
        is_roman = bool(ROMAN_RE.match(s))

        if is_running or is_number or is_roman:
            bottom_drop_indices.add(idx)
            if printed_page is None:
                m = re.search(r"\b(\d+)\b", s)
                if m:
                    printed_page = int(m.group(1))
                elif is_roman:
                    printed_page = s.lower().replace("page", "").strip()

    all_drop = top_drop_indices | bottom_drop_indices
    cleaned_lines = [line for i, line in enumerate(lines) if i not in all_drop]
    return cleaned_lines, printed_page


def normalize_math(text: str) -> str:
    """Converts unicode superscripts to standard '^n' syntax before normalization."""
    for char, replacement in SUPERSCRIPT_MAP.items():
        text = text.replace(char, replacement)
    return text


def normalize_chars(text: str) -> str:
    """Performs character-level normalization without collapsing line breaks."""
    # 1. Custom quote fixes must run BEFORE NFKC
    for old, new in QUOTE_FIXES:
        text = text.replace(old, new)

    # 2. Unicode normalization & invisible character removal
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\xad", "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 3. Bullets to markdown dashes
    text = re.sub(r"[\uf0d8\uf076\uf0b7•■◆►]\s*", "- ", text)

    # 4. Collapse inline tabs and spaces (leaves newlines intact)
    return re.sub(r"[ \t]+", " ", text)


def format_captions(text: str) -> str:
    """Formats figure captions into bold Markdown blocks."""
    return FIGURE_RE.sub(r"\n\n**\1 \2: \3**\n\n", text)


def join_lines(text: str) -> str:
    """Joins hard-wrapped lines while preserving paragraphs and bullet lists."""
    text = re.sub(r"\.{3,}", "...", text)
    # Rejoin hyphenated words only when the trailing part starts with a lowercase letter
    text = re.sub(r"(\w+)-\s*\n\s*([a-z]\w*)", r"\1\2", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Keep bullets on their own line; join single newlines into spaces
    return re.sub(r"(?<!\n)\n(?!\n|- )", " ", text).strip()


def detect_exercise(text: str) -> bool:
    """Detects whether the segment text contains activity or self-check exercise markers."""
    return bool(EXERCISE_RE.search(text))


def extract_tables_from_page(page: pymupdf.Page) -> list[str]:
    """Extracts tables from a PyMuPDF page as Markdown-formatted strings."""
    tables_md = []
    try:
        tabs = page.find_tables()
        for tab in tabs.tables:
            data = tab.extract()
            if len(data) >= 2 and len(data[0]) >= 2:
                tables_md.append(tab.to_markdown())
    except Exception as exc:
        logger.debug("Table extraction error on page %d: %s", page.number, exc, exc_info=True)
    return tables_md


def split_by_headings(
    raw: str,
    chapter: str,
    section: str,
    page_num: int,
    printed_page: int | str | None,
    source: str,
    module: str,
    is_toc: bool = False,
) -> tuple[list[dict[str, Any]], str, str]:
    """Splits text on detected chapter and section lines into distinct segment records."""
    segments: list[dict[str, Any]] = []
    buf: list[str] = []
    lines = raw.split("\n")
    skip_indices: set[int] = set()

    def flush() -> None:
        text = join_lines("\n".join(buf))
        if text:
            segments.append(
                {
                    "module": module,
                    "source": source,
                    "page": page_num,
                    "printed_page": printed_page,
                    "chapter": chapter,
                    "section": section,
                    "has_exercise": detect_exercise(text),
                    "text": text,
                }
            )
        buf.clear()

    for i, line in enumerate(lines):
        if i in skip_indices:
            continue

        s = line.strip()
        if not s:
            buf.append(line)
            continue

        # Skip Table of Contents leader lines
        if "..." in s or re.search(r"\.{3,}", s):
            buf.append(line)
            continue

        # Standalone section number followed by title on next line (e.g. "1.1" \n "Evolution of Technologies")
        if not is_toc and STANDALONE_SEC_NUM.match(s) and i + 1 < len(lines):
            next_s = lines[i + 1].strip()
            if (
                re.match(r"^[A-Z][^.!\n]{2,90}\??$", next_s)
                and not next_s.isdigit()
                and not DISALLOWED_SEC_START.search(next_s)
                and not DISALLOWED_SEC_END.search(next_s)
                and not ("..." in next_s or re.search(r"\.{3,}", next_s))
            ):
                s = f"{s} {next_s}"
                line = f"{line} {next_s}"
                skip_indices.add(i + 1)

        m_chap = CHAPTER_RE.match(s) if not is_toc else None
        if m_chap:
            title_part = m_chap.group(1).strip()
            # If chapter heading has no title on this line, peek at next line
            # OR if title ends with CONTINUATION_END (e.g. 'OF', 'AND', ',')
            if (not title_part or CONTINUATION_END.search(s)) and i + 1 < len(lines):
                next_s = lines[i + 1].strip()
                if (
                    len(next_s) < 80
                    and not next_s.isdigit()
                    and not next_s.lower().startswith(("chapter", "module", "unit", "lesson", "part"))
                    and not ("..." in next_s or re.search(r"\.{3,}", next_s))
                ):
                    if title_part:
                        s = f"{s} {next_s}"
                        line = f"{line} {next_s}"
                    else:
                        s = f"{s}: {next_s}"
                        line = f"{line}: {next_s}"
                    skip_indices.add(i + 1)
            flush()
            chapter, section = s, ""
            buf.append(line)
            continue

        if not is_toc and SECTION_RE.match(s):
            # If section title wraps onto next line (ends with 'and', 'or', ',', ':')
            if CONTINUATION_END.search(s) and i + 1 < len(lines):
                next_s = lines[i + 1].strip()
                if (
                    len(next_s) < 80
                    and not next_s.isdigit()
                    and not next_s.lower().startswith(("chapter", "lesson", "unit", "activity", "exercise"))
                    and not ("..." in next_s or re.search(r"\.{3,}", next_s))
                ):
                    s = f"{s} {next_s}"
                    line = f"{line} {next_s}"
                    skip_indices.add(i + 1)
            flush()
            section = s
            buf.append(line)
            continue

        buf.append(line)

    flush()
    return segments, chapter, section


def extract_structured_pages(
    path: Path,
    include_tables: bool = False,
) -> list[dict[str, Any]]:
    """Extracts, cleans, and structures text into heading-aligned segments from a PDF textbook."""
    doc = pymupdf.open(path)
    all_segments: list[dict[str, Any]] = []

    # 1. Frequency-based header/footer detection across the whole document
    running_lines = find_running_lines(doc, edge=3, min_ratio=0.25)
    logger.debug("Discovered running edge lines for %s: %s", path.name, running_lines)

    current_chapter = "Introduction / Front Matter"
    current_section = ""
    module_name = path.stem
    in_toc = False

    for page_num in range(doc.page_count):
        page = doc[page_num]
        raw_lines = page.get_text("text").split("\n")

        # 2. Strip edge noise and extract printed page number
        cleaned_lines, printed_page = strip_page_headers_footers(raw_lines, running_lines, edge=3)

        # 3. Track Table of Contents pages
        has_toc_header = any("table of contents" in l.lower() for l in cleaned_lines[:5])
        digit_ratio = (
            sum(1 for l in cleaned_lines if l.strip().isdigit()) / len(cleaned_lines)
            if cleaned_lines
            else 0
        )
        dot_ratio = (
            sum(1 for l in cleaned_lines if "..." in l or "....." in l) / len(cleaned_lines)
            if cleaned_lines
            else 0
        )
        if has_toc_header:
            in_toc = True
        elif in_toc and (digit_ratio < 0.2 and dot_ratio < 0.2):
            in_toc = False

        page_text = "\n".join(cleaned_lines)

        # 4. Math superscript and character normalization
        page_text = normalize_math(page_text)
        page_text = normalize_chars(page_text)
        page_text = format_captions(page_text)

        # 5. Optional table extraction
        if include_tables:
            tables = extract_tables_from_page(page)
            if tables:
                page_text += "\n\n" + "\n\n".join(tables)

        # 6. Heading detection and segmentation
        if page_text.strip():
            page_segments, current_chapter, current_section = split_by_headings(
                raw=page_text,
                chapter=current_chapter,
                section=current_section,
                page_num=page_num + 1,
                printed_page=printed_page,
                source=path.name,
                module=module_name,
                is_toc=in_toc,
            )
            all_segments.extend(page_segments)

    doc.close()
    return all_segments


def save_processed_pages(pages: list[dict[str, Any]], output_path: Path) -> None:
    """Saves structured records to a JSON Lines (JSONL) file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_data in pages:
            f.write(json.dumps(page_data, ensure_ascii=False) + "\n")


def ingest_all(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    include_tables: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Ingests all PDFs from raw_dir and saves processed JSONL files to processed_dir."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", raw_dir)
        return {}

    all_modules: dict[str, list[dict[str, Any]]] = {}

    for pdf_path in pdf_paths:
        logger.info("Processing %s...", pdf_path.name)
        segments = extract_structured_pages(pdf_path, include_tables=include_tables)
        all_modules[pdf_path.name] = segments

        output_file = processed_dir / f"{pdf_path.stem}.jsonl"
        save_processed_pages(segments, output_file)
        logger.info("  -> Saved %d segments to %s", len(segments), output_file)

    logger.info("Done! Ingested %d modules into %s", len(all_modules), processed_dir)
    return all_modules


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed"

    ingest_all(raw_path, processed_path, include_tables=False)