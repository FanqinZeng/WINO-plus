import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_trainingdata.common.eval_utils import format_iconqa_prompt, parse_iconqa_problem
from prepare_trainingdata.common.jsonl import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IconQA prompts for MMaDA WINO trajectory collection.")
    parser.add_argument("--input-file", required=True, help="IconQA JSONL file.")
    parser.add_argument("--image-root", required=True, help="Root directory containing IconQA images.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-path", required=True, help="Tokenizer path, usually MMaDA-8B-MixCoT.")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    rows = []
    with Path(args.input_file).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    image_root = Path(args.image_root)
    processed = []
    for index, row in enumerate(rows):
        images = row.get("images", [])
        if not images:
            continue
        rel_image_path = images[0].get("path", "")
        if rel_image_path.startswith("images/"):
            rel_image_path = rel_image_path[len("images/") :]
        if not (image_root / rel_image_path).exists():
            continue

        question, options = parse_iconqa_problem(row.get("problem", ""))
        formatted_question = format_iconqa_prompt(question, options)
        context = [{"role": "user", "content": formatted_question}]
        prompt_text = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()

        processed.append(
            {
                "unique_id": row.get("id", f"iconqa_{index}"),
                "source": "iconqa",
                "rel_image_path": rel_image_path,
                "question": question,
                "options": options,
                "answer": row.get("answer", ""),
                "question_formatted": formatted_question,
                "context": context,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_length": len(prompt_ids),
            }
        )

    write_jsonl(args.output_file, processed)
    print(f"Wrote {len(processed)} IconQA prompts to {args.output_file}")


if __name__ == "__main__":
    main()

