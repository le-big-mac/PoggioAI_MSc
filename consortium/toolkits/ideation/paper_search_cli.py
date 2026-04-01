"""CLI entry point for PaperSearchTool."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Search academic papers via Semantic Scholar")
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    args = parser.parse_args()

    try:
        from consortium.toolkits.ideation.paper_search_tool import PaperSearchTool

        tool = PaperSearchTool()
        result = tool._run(query=args.query, result_limit=args.limit)
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
