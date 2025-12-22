from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from daijisen import (
    PROJECT_ROOT,
    DEFAULT_DAIJISEN_DIR,
    _collect_term_bank_paths,
    format_entry_details,
    load_term_bank,
    flatten_entry_blocks,
)

def iter_entry_details(term_bank_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield metadata + flattened definition for every Daijisen entry."""
    for path in _collect_term_bank_paths(term_bank_dir):
        for entry in load_term_bank(path):
            detail = entry.to_metadata()
            detail["term_bank"] = path.name
            detail["definition"] = flatten_entry_blocks(entry)
            yield detail


def build_lookup(term_bank_dir: Path) -> Dict[str, List[dict[str, Any]]]:
    """Aggregate entries by both headword and reading for quick lookups."""
    lookup: Dict[str, List[dict[str, Any]]] = defaultdict(list)
    for detail in iter_entry_details(term_bank_dir):
        keys = set()
        headword = detail.get("headword")
        if headword:
            keys.add(headword)
        reading = detail.get("reading")
        if reading:
            keys.add(reading)
        for key in keys:
            lookup[key].append(detail)
    return lookup


def write_jsonl(records: Iterable[dict[str, Any]], destination: Path) -> None:
    """Write dataset records to a UTF-8 JSONL file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def make_records(lookup: Dict[str, List[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Convert lookup dict into dataset-ready JSON-serializable records."""
    for word in sorted(lookup):
        entries = lookup[word]
        formatted_entries = json.loads(format_entry_details(entries))
        output_lines = []
        for entry in formatted_entries:
            # Extract fields
            headword = entry.get('headword', '')
            reading = entry.get('reading', '')
            pos = entry.get('part_of_speech', '')
            definition = entry.get('definition', '').strip()
            
            # 1. Header Line: ■ Headword (Reading)
            header = f"■ {headword} ({reading})"
            output_lines.append(header)
            
            # 2. Metadata Line (if available)
            if pos:
                output_lines.append(f"   [Part of Speech: {pos}]")
            
            # 3. Separator
            output_lines.append("   " + "-" * 10)
            
            # 4. Definition Body (Indented for readability)
            # Split by newline to preserve original formatting but add indent
            def_lines = definition.split('\n')
            for line in def_lines:
                # Only add lines that aren't empty to keep it tight
                if line.strip():
                    output_lines.append(f"   {line}")
            
            # Add a blank line between entries
            output_lines.append("")
        formatted = "\n".join(output_lines).strip()
        
        yield {
            "word": word,
            "entries": entries,
            "formatted": formatted,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Daijisen dataset suitable for describe_daijisen_entry lookups.",
    )
    parser.add_argument(
        "--term-bank-dir",
        type=Path,
        default=DEFAULT_DAIJISEN_DIR,
        help="Directory containing term_bank_*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Processed Data" / "[JA-JA]_大辞泉_第二版.jsonl",
        help="Destination JSON Lines file for the dataset.",
    )
    args = parser.parse_args()

    lookup = build_lookup(args.term_bank_dir)
    records = make_records(lookup)
    write_jsonl(records, args.output)
    print(f"Wrote {len(lookup)} entries to {args.output}")


if __name__ == "__main__":
    main()
