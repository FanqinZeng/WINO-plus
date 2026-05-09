import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_trainingdata.common.eval_utils import count_operations
from prepare_trainingdata.common.jsonl import read_jsonl, write_jsonl


def _sort_key(row: dict):
    temperature = row.get("used_temperature", 0)
    steps = row.get("decoding_steps", 10**9)
    equation = row.get("metadata", {}).get("extracted_equation", row.get("extracted_equation", ""))
    return (temperature, steps, -count_operations(equation))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and sort WINO trajectory JSONL records.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--correct-only", action="store_true", default=True)
    parser.add_argument("--allow-incorrect", action="store_false", dest="correct_only")
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--keep-step-ratio", type=float, default=None, help="Keep earliest rows until cumulative steps reach this ratio.")
    parser.add_argument("--dedupe-key", default="unique_id", help="Keep one best row per key before final sorting.")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input_file):
        if args.correct_only and not row.get("correct", False):
            continue
        if args.max_prompt_length is not None and row.get("prompt_length", 0) > args.max_prompt_length:
            continue
        rows.append(row)

    if args.dedupe_key:
        grouped = {}
        for row in rows:
            key = row.get(args.dedupe_key)
            if key is None:
                key = row.get("question")
            if key not in grouped or _sort_key(row) < _sort_key(grouped[key]):
                grouped[key] = row
        rows = list(grouped.values())

    rows = sorted(rows, key=_sort_key)

    if args.keep_step_ratio is not None:
        if not 0 < args.keep_step_ratio <= 1:
            raise ValueError("--keep-step-ratio must be in (0, 1].")
        total_steps = sum(max(row.get("trajectory_accepted", [-1]) or [-1]) + 1 for row in rows)
        cutoff = int(total_steps * args.keep_step_ratio)
        kept = []
        cumulative = 0
        for row in rows:
            step_count = max(row.get("trajectory_accepted", [-1]) or [-1]) + 1
            if cumulative + step_count > cutoff:
                break
            kept.append(row)
            cumulative += step_count
        rows = kept

    write_jsonl(args.output_file, rows)
    print(f"Wrote {len(rows)} filtered trajectories to {args.output_file}")


if __name__ == "__main__":
    main()

