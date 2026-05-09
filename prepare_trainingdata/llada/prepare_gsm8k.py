import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_trainingdata.common.eval_utils import extract_gsm8k_answer
from prepare_trainingdata.common.jsonl import write_jsonl


GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


def _load_rows(args):
    if args.input_jsonl:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    return list(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K prompts for WINO trajectory collection.")
    parser.add_argument("--output-file", required=True, help="Path to write processed JSONL.")
    parser.add_argument("--model-path", required=True, help="Tokenizer path, usually LLaDA-8B-Instruct.")
    parser.add_argument("--input-jsonl", default=None, help="Optional local JSONL with question/answer fields.")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = _load_rows(args)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    processed = []
    for index, row in enumerate(rows):
        question = row.get("question", "")
        answer = row.get("answer", "")
        context = [{"role": "user", "content": GSM_SYSTEM_PROMPT + "\n\n" + question}]
        prompt_text = GSM_SYSTEM_PROMPT + "\n\n" + question
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()

        processed.append(
            {
                "unique_id": row.get("unique_id", f"gsm8k_{index}"),
                "source": "gsm8k",
                "question": question,
                "answer": answer,
                "digital_answer": extract_gsm8k_answer(answer),
                "context": context,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_length": len(prompt_ids),
            }
        )

    write_jsonl(args.output_file, processed)
    print(f"Wrote {len(processed)} GSM8K prompts to {args.output_file}")


if __name__ == "__main__":
    main()
