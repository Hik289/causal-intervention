import argparse
import json
from pathlib import Path

from civic.checkpoint_selection import select_table_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Choose one measured checkpoint for a complete result table")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--selection-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = select_table_checkpoint(payload["candidates"], args.primary, payload["directions"],
                                     args.selection_split, args.verify_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
