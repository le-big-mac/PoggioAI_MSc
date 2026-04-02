"""CLI entry point for MathProofRigorCheckerTool."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Check proof rigor")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--claim-id", default=None, help="Claim to check")
    parser.add_argument("--proof-text", default=None, help="Inline proof text")
    parser.add_argument("--check-level", default="strict", choices=["basic", "strict"],
                        help="Check level (default: strict)")
    args = parser.parse_args()

    try:
        from consortium.toolkits.math.proof_rigor_checker_tool import MathProofRigorCheckerTool

        tool = MathProofRigorCheckerTool(working_dir=args.workspace)
        result = tool.run(
            claim_id=args.claim_id,
            proof_text=args.proof_text,
            check_level=args.check_level,
            workspace_subdir="math_workspace",
        )
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
