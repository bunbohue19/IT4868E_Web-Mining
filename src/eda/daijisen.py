from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Sequence

__all__ = [
    "DaijisenEntry",
    "load_term_bank",
    "flatten_structured_node",
    "flatten_entry_blocks",
    "search_word_metadata",
    "search_word_metadata_all",
    "search_word_details",
    "search_word_details_all",
    "format_entry_details",
    "run_terminal_search",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAIJISEN_DIR = PROJECT_ROOT / "data" / "[JA-JA] 大辞泉 第二版[2025-04-29]"
DEFAULT_TERM_BANK_PATH = DEFAULT_DAIJISEN_DIR / "term_bank_1.json"

BLOCK_LEVEL_TAGS = {"div", "p", "section", "article", "ul", "ol", "li"}
INLINE_BREAK_TAGS = {"br"}
WHITESPACE_RE = re.compile(r"[ \t\f\r]+")
NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class DaijisenEntry:
    """Small helper to reason about the 8-field Daijisen array payload."""

    headword: str
    reading: str
    variation: str
    part_of_speech: str
    priority: int
    blocks: Sequence[Any]
    index: int
    note: str

    @classmethod
    def from_raw(cls, payload: Sequence[Any]) -> "DaijisenEntry":
        if len(payload) != 8:
            raise ValueError(f"Expected 8 values per entry, got {len(payload)}")
        headword, reading, variation, pos, priority, blocks, index, note = payload
        return cls(
            headword=str(headword or ""),
            reading=str(reading or ""),
            variation=str(variation or ""),
            part_of_speech=str(pos or ""),
            priority=int(priority or 0),
            blocks=tuple(blocks or ()),
            index=int(index or 0),
            note=str(note or ""),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the entry into a JSON-friendly metadata dict."""
        return {
            "headword": self.headword,
            "reading": self.reading,
            "variation": self.variation,
            "part_of_speech": self.part_of_speech,
            "priority": self.priority,
            "index": self.index,
            "note": self.note,
        }


def load_term_bank(path: Path) -> List[DaijisenEntry]:
    """
    Load a Daijisen term_bank JSON file and normalize it into entries.

    Parameters
    ----------
    path:
        Path to a `term_bank_*.json` file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Daijisen term bank not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw_entries = json.load(handle)
    return [DaijisenEntry.from_raw(entry) for entry in raw_entries]


def _collect_term_bank_paths(location: Path | None = None) -> List[Path]:
    """
    Resolve one or multiple term bank paths.

    If `location` is a file, return it; if it's a directory, return every
    `term_bank_*.json` found under it (sorted by filename). Defaults to the
    Daijisen data directory.
    """
    target = location or DEFAULT_DAIJISEN_DIR
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(target.glob("term_bank_*.json"))
        if not files:
            raise FileNotFoundError(f"No term_bank files found in {target}")
        return files
    raise FileNotFoundError(f"Term bank path not found: {target}")


def _iter_children(node: Any) -> Iterator[Any]:
    """Yield every child node regardless of whether content is a list or single item."""
    content = None
    if isinstance(node, dict):
        content = node.get("content")
    if isinstance(content, list):
        yield from content
    elif content is not None:
        yield content


def flatten_structured_node(node: Any) -> str:
    """
    Convert a Daijisen structured-content node into plain text.

    This strips every tag and keeps lightweight block/newline hints so the
    resulting text remains readable while avoiding HTML/XHTML payloads.
    """
    pieces: List[str] = []

    def _walk(current: Any) -> None:
        if current is None:
            return
        if isinstance(current, str):
            pieces.append(current)
            return
        if isinstance(current, (int, float)):
            pieces.append(str(current))
            return
        if isinstance(current, list):
            for child in current:
                _walk(child)
            return
        if isinstance(current, dict):
            tag = current.get("tag")
            if tag in INLINE_BREAK_TAGS:
                pieces.append("\n")
                return
            before_len = len(pieces)
            for child in _iter_children(current):
                _walk(child)
            if tag in BLOCK_LEVEL_TAGS and len(pieces) > before_len:
                pieces.append("\n")
            return
        pieces.append(str(current))

    _walk(node)
    text = "".join(pieces)
    text = WHITESPACE_RE.sub(" ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def flatten_entry_blocks(entry: DaijisenEntry) -> str:
    """Flatten every structured-content block stored in `entry` to readable text."""
    paragraphs = [flatten_structured_node(block) for block in entry.blocks]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return "\n\n".join(paragraphs).strip()


def search_word_metadata(word: str, term_bank_path: Path | None = None) -> List[dict[str, Any]]:
    """
    Find every entry whose headword or reading matches `word` and return metadata.
    """
    if not isinstance(word, str):
        raise TypeError("word must be a string")
    if not word:
        raise ValueError("word must be a non-empty string")

    path = term_bank_path or DEFAULT_TERM_BANK_PATH
    entries = load_term_bank(path)
    matches = [
        entry.to_metadata()
        for entry in entries
        if entry.headword == word or entry.reading == word
    ]
    return matches


def search_word_metadata_all(
    word: str, term_bank_dir: Path | None = None
) -> List[dict[str, Any]]:
    """
    Scan every term bank file for `word` (matching headword or reading).

    The response mirrors `search_word_metadata` but adds a `term_bank` field so
    callers can identify which JSON file provided the entry.
    """
    paths = _collect_term_bank_paths(term_bank_dir)
    aggregated: List[dict[str, Any]] = []
    for path in paths:
        entries = load_term_bank(path)
        for entry in entries:
            if entry.headword == word or entry.reading == word:
                metadata = entry.to_metadata()
                metadata["term_bank"] = path.name
                aggregated.append(metadata)
    return aggregated


def _entry_to_detail(entry: DaijisenEntry, term_bank_name: str | None = None) -> dict[str, Any]:
    """Return metadata plus raw/flattened definitions for an entry."""
    data = entry.to_metadata()
    if term_bank_name:
        data["term_bank"] = term_bank_name
    data["definition"] = flatten_entry_blocks(entry)
    data["blocks"] = json.loads(json.dumps(entry.blocks))
    return data


def search_word_details(
    word: str, term_bank_path: Path | None = None
) -> List[dict[str, Any]]:
    """Return metadata + definitions for every match within a single term bank."""
    path = term_bank_path or DEFAULT_TERM_BANK_PATH
    entries = load_term_bank(path)
    details = [
        _entry_to_detail(entry, term_bank_name=path.name)
        for entry in entries
        if entry.headword == word or entry.reading == word
    ]
    return details


def search_word_details_all(word: str, term_bank_dir: Path | None = None) -> List[dict[str, Any]]:
    """Return metadata + definitions for every match across all term banks."""
    paths = _collect_term_bank_paths(term_bank_dir)
    aggregated: List[dict[str, Any]] = []
    for path in paths:
        entries = load_term_bank(path)
        for entry in entries:
            if entry.headword == word or entry.reading == word:
                aggregated.append(_entry_to_detail(entry, term_bank_name=path.name))
    return aggregated


def format_entry_details(entries: Sequence[dict[str, Any]]) -> str:         
    """Render a list of entry metadata dicts."""                                
    chunks: List[str] = []                                                          
    for entry in entries:                                                                             
        headword = entry.get("headword") or "?"                                     
        reading = entry.get("reading") or "?"                                               
        pos = entry.get("part_of_speech") or "—"                                        
        variation = entry.get("variation")                                              
        note = entry.get("note")                                                                
        definition = entry.get("definition")                                            
                                                                                    
        chunks.append({                                                                         
            "headword": headword,                                                   
            "reading": reading,                                                         
            "part_of_speech": pos,                                                              
            "variation": variation if variation else "",                                    
            "note": note if note else "",                                                           
            "definition": definition if definition else "",                                         
        })
        
    return json.dumps(chunks, ensure_ascii=False, indent=3)    


def describe_daijisen_entry(word: str) -> str:
    """
    Prompt the user for search terms and print readable definitions.

    The loop exits on blank input, EOF, or Ctrl-C.
    """
    try:
        results = search_word_details_all(word, term_bank_dir=DEFAULT_DAIJISEN_DIR)
    except FileNotFoundError as exc:
        return f"Error: {exc}"

    if not results:
        return "No entries found."

    return format_entry_details(results)


if __name__ == "__main__":
    # print(describe_daijisen_entry("空く"))
    print(DEFAULT_DAIJISEN_DIR)