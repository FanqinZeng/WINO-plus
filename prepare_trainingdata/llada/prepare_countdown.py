import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_trainingdata.common.jsonl import write_jsonl


CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target "
    "number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think "
    "step-by-step. After reasoning, provide only your final expression inside \\boxed{} tags without including an "
    "equals sign or the target number. For example: \\boxed{a + b * c}"
    """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)


def _load_rows(args):
    if args.input_jsonl:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if args.input_parquet_dir:
        import pandas as pd

        rows = []
        for parquet_file in sorted(Path(args.input_parquet_dir).glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            for _, item in df.iterrows():
                rows.append({"input": item["nums"], "output": item["target"]})
        return rows

    raise ValueError("Provide either --input-jsonl or --input-parquet-dir.")


def _parse_numbers(raw_input):
    if isinstance(raw_input, str):
        return [int(i) for i in raw_input.split(",") if i.strip()]
    return [int(i) for i in raw_input]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Countdown prompts for WINO trajectory collection.")
    parser.add_argument("--output-file", required=True, help="Path to write processed JSONL.")
    parser.add_argument("--model-path", required=True, help="Tokenizer path, usually LLaDA-8B-Instruct.")
    parser.add_argument("--input-jsonl", default=None, help="Optional local JSONL with input/output fields.")
    parser.add_argument("--input-parquet-dir", default=None, help="Directory containing Countdown parquet files.")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = _load_rows(args)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    processed = []
    for index, row in enumerate(rows):
        target = int(row.get("output", row.get("target")))
        numbers = _parse_numbers(row.get("input", row.get("numbers")))
        question = f"Numbers: {numbers}\nTarget: {target}"
        context = [{"role": "user", "content": CTD_SYSTEM_PROMPT + "\n\n" + question}]
        prompt_text = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()

        processed.append(
            {
                "unique_id": row.get("unique_id", f"countdown_{index}"),
                "source": "countdown",
                "prompt_ids": prompt_ids,
                "prompt_length": len(prompt_ids),
                "numbers": numbers,
                "digital_answer": target,
                "question": question,
                "context": context,
            }
        )

    write_jsonl(args.output_file, processed)
    print(f"Wrote {len(processed)} Countdown prompts to {args.output_file}")


if __name__ == "__main__":
    main()

