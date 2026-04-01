"""CLI entry point for MathClaimGraphTool."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Manage math claim graphs")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--action", required=True, help="Action to perform")
    parser.add_argument("--claim-id", default=None)
    parser.add_argument("--statement", default=None)
    parser.add_argument("--assumptions", default=None, help="JSON array of assumptions")
    parser.add_argument("--depends-on", default=None, help="JSON array of dependency claim ids")
    parser.add_argument("--tags", default=None, help="JSON array of tags")
    parser.add_argument("--status", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--must-accept", type=bool, default=None)
    parser.add_argument("--lemma-id", default=None)
    parser.add_argument("--lemma-tier", default=None)
    parser.add_argument("--lemma-statement", default=None)
    parser.add_argument("--lemma-conditions", default=None)
    parser.add_argument("--lemma-source", default=None)
    parser.add_argument("--lemma-usage-notes", default=None)
    parser.add_argument("--lemma-tags", default=None, help="JSON array of lemma tags")
    parser.add_argument("--lemma-status", default=None)
    parser.add_argument("--lemma-limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from consortium.toolkits.math.claim_graph_tool import MathClaimGraphTool

        tool = MathClaimGraphTool(working_dir=args.workspace)
        result = tool.run(
            action=args.action,
            claim_id=args.claim_id,
            statement=args.statement,
            assumptions_json=args.assumptions,
            depends_on_json=args.depends_on,
            tags_json=args.tags,
            status=args.status,
            notes=args.notes,
            must_accept=args.must_accept,
            workspace_subdir=".",
            lemma_id=args.lemma_id,
            lemma_tier=args.lemma_tier,
            lemma_statement=args.lemma_statement,
            lemma_conditions=args.lemma_conditions,
            lemma_source=args.lemma_source,
            lemma_usage_notes=args.lemma_usage_notes,
            lemma_tags_json=args.lemma_tags,
            lemma_status=args.lemma_status,
            lemma_limit=args.lemma_limit,
        )
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
