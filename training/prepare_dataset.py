"""Validate a simple JSONL dataset before a LoRA experiment."""

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {"instruction", "input", "output"}


def validate_dataset(path: Path) -> int:
    """Return validated row count or raise ValueError with the offending line."""
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Line {line_number} is missing: {', '.join(sorted(missing))}")
        if not all(isinstance(row[field], str) and row[field].strip() for field in REQUIRED_FIELDS):
            raise ValueError(f"Line {line_number} must contain non-empty strings.")
        count += 1
    if count == 0:
        raise ValueError("Dataset contains no examples.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LoRA instruction-tuning JSONL.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(f"Validated {validate_dataset(args.dataset)} example(s): {args.dataset}")


if __name__ == "__main__":
    main()
