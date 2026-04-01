"""CLI entry point for FetchArxivPapersTool."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=10, help="Max papers (default: 10)")
    parser.add_argument("--output-dir", default=None, help="Directory to save PDFs")
    args = parser.parse_args()

    try:
        from consortium.toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool

        tool = FetchArxivPapersTool(working_dir=args.output_dir)
        result = tool._run(search_query=args.query, max_results=args.max_results)
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
