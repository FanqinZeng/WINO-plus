import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
MMADA_MODEL_ROOT = REPO_ROOT / "MMaDA" / "lmms_eval" / "lmms_eval" / "models"
for path in (REPO_ROOT, MMADA_MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prepare_trainingdata.common.eval_utils import iconqa_rule_is_correct
from prepare_trainingdata.common.jsonl import merge_jsonl_files, read_jsonl
from prepare_trainingdata.common.schema import normalize_trajectory_record, validate_trajectory_record
from prepare_trainingdata.common.sharding import cleanup_files, split_evenly
from prepare_trainingdata.mmada.mmada_wino import mmu_generate_wino_with_trajectory


MASK_TOKEN_ID = 126336
REASONING_PROMPT = (
    "You should first think about the reasoning process in the mind and then provide the user with the answer. "
    "The reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here\n"
)


def _temperatures(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _build_multimodal_input(image_tokens, text_ids, uni_prompting, device):
    batch_size = image_tokens.shape[0]
    return torch.cat(
        [
            (torch.ones(batch_size, 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(device),
            (torch.ones(batch_size, 1) * uni_prompting.sptids_dict["<|soi|>"]).to(device),
            image_tokens,
            (torch.ones(batch_size, 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(device),
            (torch.ones(batch_size, 1) * uni_prompting.sptids_dict["<|sot|>"]).to(device),
            text_ids,
        ],
        dim=1,
    ).long()


def _image_transform(image, resolution=512, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


def _gpt_judge(generated_text: str, row: dict) -> bool:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_URL"),
    )
    options = "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(row.get("options", [])))
    prompt = (
        "Return only 1 if the model prediction answers the question correctly, otherwise return 0.\n\n"
        f"Question:\n{row.get('question', '')}\n{options}\n\n"
        f"Ground truth:\n{row.get('answer', '')}\n\n"
        f"Prediction:\n{generated_text}"
    )
    response = client.chat.completions.create(
        model=os.environ.get("MODEL_VERSION", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    return response.choices[0].message.content.strip().startswith("1")


def _worker(rank, device_map_list, rows, output_file, args_dict, start_index):
    from transformers import AutoTokenizer
    from model_mmada.modeling_magvitv2 import MAGVITv2
    from model_mmada.modeling_mmada import MMadaModelLM
    from model_mmada.prompting_utils import UniversalPrompting

    first_gpu = f"cuda:{device_map_list[0]}"
    input_device = torch.device(first_gpu)
    max_memory = {gpu_id: args_dict["max_memory"] for gpu_id in device_map_list}

    tokenizer = AutoTokenizer.from_pretrained(args_dict["mmada_model_path"], trust_remote_code=True, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=args_dict["max_text_len"],
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True,
    )
    vq_model = MAGVITv2.from_pretrained(args_dict["vq_model_path"]).to(input_device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = MMadaModelLM.from_pretrained(
        args_dict["mmada_model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    ).eval()

    image_root = Path(args_dict["image_root"])
    with Path(output_file).open("w", encoding="utf-8") as f:
        for local_index, row in enumerate(tqdm(rows, position=rank, desc=f"worker {rank}", leave=False)):
            image_path = image_root / row.get("rel_image_path", "")
            if not image_path.exists():
                fallback = image_root / Path(row.get("rel_image_path", "")).name
                if fallback.exists():
                    image_path = fallback
                else:
                    continue

            image = Image.open(image_path).convert("RGB")
            image_tensor = _image_transform(image, resolution=args_dict["resolution"]).to(input_device).unsqueeze(0)
            image_tokens = vq_model.get_code(image_tensor) + len(uni_prompting.text_tokenizer)

            context = row.get("context") or [{"role": "user", "content": row.get("question_formatted", row.get("question", ""))}]
            if args_dict["reasoning"]:
                context = [dict(item) for item in context]
                for message in context:
                    if message.get("role") == "user" and REASONING_PROMPT not in message.get("content", ""):
                        message["content"] = REASONING_PROMPT + message["content"]
                        break
                text_ids = tokenizer.apply_chat_template(context, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(input_device)
            else:
                text_ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=input_device).unsqueeze(0)

            input_ids = _build_multimodal_input(image_tokens, text_ids, uni_prompting, input_device)
            prompt_len = input_ids.shape[1]
            final_record = None

            for temperature in args_dict["temperatures"]:
                torch.cuda.empty_cache()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids, steps, traj_accept, traj_proposal = mmu_generate_wino_with_trajectory(
                        model=model,
                        prompt=input_ids,
                        gen_length=args_dict["gen_length"],
                        block_length=args_dict["block_length"],
                        temperature=temperature,
                        mask_id=MASK_TOKEN_ID,
                        threshold=args_dict["threshold"],
                        threshold_back=args_dict["threshold_back"],
                    )

                generated_ids_tensor = output_ids[:, prompt_len:][0].detach().cpu()
                generated_ids = generated_ids_tensor.tolist()
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id in generated_ids:
                    eos_index = generated_ids.index(eos_token_id) + 1
                    generated_ids = generated_ids[:eos_index]
                    traj_accept = traj_accept[:eos_index]
                    traj_proposal = traj_proposal[:eos_index]

                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                if args_dict["judge"] == "gpt":
                    correct = _gpt_judge(generated_text, row)
                else:
                    correct = iconqa_rule_is_correct(generated_text, row.get("answer", ""), row.get("options", []))

                final_record = normalize_trajectory_record(
                    {
                        "unique_id": row.get("unique_id", f"iconqa_{start_index + local_index}"),
                        "source": "iconqa",
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "prompt_ids": input_ids[0].detach().cpu().tolist(),
                        "prompt_length": prompt_len,
                        "generated_ids": generated_ids,
                        "trajectory_accepted": traj_accept,
                        "trajectory_proposed": traj_proposal,
                        "correct": correct,
                        "generated_text": generated_text,
                        "decoding_steps": steps,
                        "used_temperature": temperature,
                        "image_path": str(image_path),
                        "metric_name": args_dict["judge"],
                        "metric_score": 1.0 if correct else 0.0,
                        "multimodal": True,
                    },
                    source="iconqa",
                )
                validate_trajectory_record(final_record)
                if correct:
                    break

            if final_record is not None:
                f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect IconQA WINO trajectories with MMaDA.")
    parser.add_argument("--mmada-model-path", required=True)
    parser.add_argument("--vq-model-path", default="showlab/magvitv2")
    parser.add_argument("--input-file", required=True, help="Processed IconQA JSONL from prepare_iconqa.py.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--temperatures", default="0,0.1")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--gpus-per-model", type=int, default=1)
    parser.add_argument("--max-memory", default="23GiB")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-back", type=float, default=0.9)
    parser.add_argument("--judge", choices=["rule", "gpt"], default="rule")
    parser.add_argument("--reasoning", action="store_true", default=True)
    parser.add_argument("--no-reasoning", action="store_false", dest="reasoning")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input_file))
    if args.num_samples is not None:
        rows = rows[: args.num_samples]
    if not rows:
        raise ValueError("No input rows found.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < args.gpus_per_model:
        raise RuntimeError(f"Need at least {args.gpus_per_model} GPU(s), found {gpu_count}.")

    num_workers = gpu_count // args.gpus_per_model
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
        gpu_start = rank * args.gpus_per_model
        device_map_list = list(range(gpu_start, gpu_start + args.gpus_per_model))
        process = mp.Process(target=_worker, args=(rank, device_map_list, chunk, temp_files[rank], args_dict, start_index))
        process.start()
        processes.append(process)
        start_index += len(chunk)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {process.exitcode}")

    total, correct = merge_jsonl_files(temp_files, output_path)
    cleanup_files(temp_files)
    print(f"Saved {total} IconQA trajectories to {output_path}; correct={correct}; accuracy={correct / total if total else 0:.4f}")


if __name__ == "__main__":
    main()
