import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LLaDA LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True, help="Path or Hugging Face id for the base LLaDA model.")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the merged model and tokenizer.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda", help="Device used while merging, for example cuda or cpu.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    dtype = parse_dtype(args.torch_dtype)

    print(f"Loading base LLaDA model from: {args.base_model}")
    base_model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(args.device)

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    print(f"Loading LoRA adapter from: {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter weights into the base model.")
    merged_model = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
