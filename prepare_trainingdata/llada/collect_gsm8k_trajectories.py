import argparse
import json
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_ROOT = REPO_ROOT / "LLaDA"
for path in (REPO_ROOT, LLADA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_trainingdata.common.eval_utils import gsm8k_is_correct
from prepare_trainingdata.common.jsonl import merge_jsonl_files, read_jsonl
from prepare_trainingdata.common.llada_wino import decoding_wino_with_trajectory
from prepare_trainingdata.common.schema import normalize_trajectory_record, validate_trajectory_record
from prepare_trainingdata.common.sharding import cleanup_files, split_evenly


def _temperatures(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _worker(rank, rows, output_file, args_dict, start_index):
    from transformers import AutoTokenizer
    from modeling_llada import LLaDAModelLM

    device = torch.device("cpu")
    if args_dict["device"] == "cuda":
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    model = LLaDAModelLM.from_pretrained(args_dict["model_path"], torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model_path"], trust_remote_code=True)

    with Path(output_file).open("w", encoding="utf-8") as f:
        for local_index, row in enumerate(tqdm(rows, position=rank, desc=f"worker {rank}", leave=False)):
            prompt_ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = prompt_ids.shape[1]
            final_record = None

            for temperature in args_dict["temperatures"]:
                output_ids, steps, traj_accept, traj_proposal = decoding_wino_with_trajectory(
                    model=model,
                    prompt=prompt_ids,
                    gen_length=args_dict["gen_length"],
                    block_length=args_dict["block_length"],
                    temperature=temperature,
                    mask_id=args_dict["mask_id"],
                    threshold=args_dict["threshold"],
                    threshold_back=args_dict["threshold_back"],
                )
                generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                correct = gsm8k_is_correct(generated_text, row.get("digital_answer"))

                final_record = normalize_trajectory_record(
                    {
                        "unique_id": row.get("unique_id", f"gsm8k_{start_index + local_index}"),
                        "source": "gsm8k",
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "prompt_ids": row["prompt_ids"],
                        "prompt_length": prompt_len,
                        "generated_ids": generated_ids,
                        "trajectory_accepted": traj_accept,
                        "trajectory_proposed": traj_proposal,
                        "correct": correct,
                        "generated_text": generated_text,
                        "decoding_steps": steps,
                        "used_temperature": temperature,
                        "digital_answer": row.get("digital_answer"),
                    },
                    source="gsm8k",
                )
                validate_trajectory_record(final_record)
                if correct:
                    break

            if final_record is not None:
                f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GSM8K WINO trajectories with LLaDA.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-file", required=True, help="Processed GSM8K JSONL from prepare_gsm8k.py.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--temperatures", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-back", type=float, default=0.8)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input_file))
    if args.num_samples is not None:
        rows = rows[: args.num_samples]
    if not rows:
        raise ValueError("No input rows found.")

    if args.device == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("CUDA requested but no GPUs were found.")
        num_workers = args.num_workers or gpu_count
    else:
        num_workers = 1

    chunks = split_evenly(rows, num_workers)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_files = [output_path.with_name(f"{output_path.stem}.worker{rank}.jsonl") for rank in range(len(chunks))]

    args_dict = vars(args)
    args_dict["temperatures"] = _temperatures(args.temperatures)

    mp.set_start_method("spawn", force=True)
    processes = []
    start_index = 0
    for rank, chunk in enumerate(chunks):
        process = mp.Process(target=_worker, args=(rank, chunk, temp_files[rank], args_dict, start_index))
        process.start()
        processes.append(process)
        start_index += len(chunk)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {process.exitcode}")

    total, correct = merge_jsonl_files(temp_files, output_path)
    cleanup_files(temp_files)
    print(f"Saved {total} GSM8K trajectories to {output_path}; correct={correct}; accuracy={correct / total if total else 0:.4f}")


if __name__ == "__main__":
    main()

