"""Run the reviewer or just its Analyzer locally before introducing an HTTP API."""

import argparse

from .agents import OllamaDocumentAgents
from .documents import read_document
from .graph import ReviewWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a technical PDF, Markdown, or text document.")
    parser.add_argument("path", help="Path to a .pdf, .md, or .txt document")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only the Analyzer agent and print its validated AnalysisReport as JSON.",
    )
    args = parser.parse_args()

    document = read_document(args.path)
    if args.analyze_only:
        result = OllamaDocumentAgents().analyze(document)
    else:
        result = ReviewWorkflow().run(document)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
