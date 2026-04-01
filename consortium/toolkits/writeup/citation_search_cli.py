"""CLI entry point for CitationSearchTool — search papers and generate BibTeX."""
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Search academic papers and generate BibTeX citations")
    parser.add_argument("--query", required=True, help="Search query (keywords, title, author, topic)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--source", choices=["arxiv", "semantic_scholar", "both"], default="both",
                        help="Search database (default: both)")
    args = parser.parse_args()

    try:
        from .citation_search_tool import CitationSearchTool
        tool = CitationSearchTool()
        result = tool.run(args.query, max_results=args.limit, search_source=args.source)
        print(result)
    except Exception as e:
        json.dump({"error": str(e)}, sys.stdout, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    main()
