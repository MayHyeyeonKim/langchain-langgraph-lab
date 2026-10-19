"""Run the reviewer locally before introducing an HTTP API."""

import argparse

from .documents import read_document
from .graph import ReviewWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a technical PDF, Markdown, or text document.")
    parser.add_argument("path", help="Path to a .pdf, .md, or .txt document")
    args = parser.parse_args()

    result = ReviewWorkflow().run(read_document(args.path))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
