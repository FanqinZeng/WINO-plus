import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
LLADA_ROOT = REPO_ROOT / "LLaDA"
if str(LLADA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLADA_ROOT))

from modeling_llada import LLaDAModelLM  # noqa: E402


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path | None) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def wait_for_path(path: Path, timeout_seconds: int = 7200) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for path: {path}")
        time.sleep(5)


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get(
        "deepspeed",
        {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "zero_allow_untested_optimizer": True,
            "bf16": {"enabled": "auto"},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        },
    )


def prepare_model(config: Dict[str, Any], init_adapter: Optional[Path] = None):
    model_cfg = config["model"]
    torch_dtype = getattr(torch, model_cfg["torch_dtype"])

    base_model = LLaDAModelLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if init_adapter is not None:
        print(f"Loading LoRA adapter from: {init_adapter}")
        model = PeftModel.from_pretrained(base_model, str(init_adapter), is_trainable=True)
    else:
        lora_cfg = config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


class DLMTrainer(Trainer):
    def __init__(
        self,
        mask_token_id: int = 126336,
        loss_log_file: Optional[Path] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        loss_config = loss_config or {}
        self.mask_token_id = mask_token_id
        self.temperature = loss_config.get("temperature", 1.0)
        self.block_size = loss_config.get("block_size", 128)
        self.w_ce_loss = loss_config.get("w_ce_loss", 1.0)
        self.w_unmask_loss = loss_config.get("w_unmask_loss", 0.1)
        self.w_remask_loss = loss_config.get("w_remask_loss", 1.0)
        self.threshold = loss_config.get("threshold", 0.6)
        self.threshold_back = loss_config.get("threshold_back", 0.9)
        self.legacy_block_mask_scope = loss_config.get("legacy_block_mask_scope", True)

        self.args.logging_steps = min(self.args.logging_steps, 5)
        self.loss_log_file = Path(loss_log_file) if loss_log_file is not None else None
        if self.is_world_process_zero() and self.loss_log_file is not None:
            self.loss_log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.loss_log_file.open("w", encoding="utf-8") as f:
                f.write(
                    "global_step,total_loss,ce_loss,unmask_loss,remask_loss,"
                    "num_masked,num_ce_loss,num_unmask_loss,num_remask_loss\n"
                )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_lengths = inputs["prompt_lengths"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        block_nums = inputs["block_num"]
        target_mask = inputs["target_mask"]

        masked_indices = input_ids == self.mask_token_id
        rs_indices = masked_indices & (~target_mask)

        if self.legacy_block_mask_scope:
            # Legacy dParallel behavior. This preserves the original single-sample
            # training semantics and is intentionally not generalized here.
            block_masked_indices = masked_indices.clone()
            block_masked_indices[:, : prompt_lengths + block_nums * self.block_size] = False
            block_masked_indices[:, prompt_lengths + (block_nums + 1) * self.block_size :] = False

            block_rs_indices = rs_indices.clone()
            block_rs_indices[:, : prompt_lengths + block_nums * self.block_size] = False
            block_rs_indices[:, prompt_lengths + (block_nums + 1) * self.block_size :] = False
        else:
            batch_size, seq_len = input_ids.shape
            seq_range = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
            start_pos = (prompt_lengths + block_nums * self.block_size).unsqueeze(1)
            end_pos = (prompt_lengths + (block_nums + 1) * self.block_size).unsqueeze(1)
            block_scope_mask = (seq_range >= start_pos) & (seq_range < end_pos)
            block_masked_indices = masked_indices & block_scope_mask
            block_rs_indices = rs_indices & block_scope_mask

        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        ce_loss = torch.tensor(0.0, device=input_ids.device)
        remask_loss = torch.tensor(0.0, device=input_ids.device)

        num_mask = masked_indices.sum().item()
        num_remask = block_rs_indices.sum().item()
        num_ce_loss = torch.tensor(0, device=input_ids.device)
        num_remask_loss = torch.tensor(0, device=input_ids.device)
        num_unmask_loss = torch.tensor(0, device=input_ids.device)

        if self.w_ce_loss > 0.0 and target_mask.sum() > 0:
            masked_logits = logits[target_mask]
            masked_labels = labels[target_mask]
            token_loss = F.cross_entropy(masked_logits, masked_labels, reduction="none")
            num_ce_loss = target_mask.sum()
            ce_loss = torch.sum(token_loss) / target_mask.sum()
        else:
            ce_loss = 0.0 * logits.sum()

        if self.w_unmask_loss > 0.0 and target_mask.sum() > 0:
            target_logits = logits[target_mask]
            with torch.no_grad():
                confidences, predictions = F.softmax(target_logits, dim=-1).max(dim=-1)
                target_labels = labels[target_mask]
                is_true = predictions == target_labels
                is_low_conf = confidences < self.threshold_back
                unmask_loss_indices = is_true & is_low_conf

            num_unmask_loss = unmask_loss_indices.sum()
            if unmask_loss_indices.sum() > 0:
                selected_logits = target_logits[unmask_loss_indices]
                selected_probs = F.softmax(selected_logits, dim=-1)
                selected_log_probs = F.log_softmax(selected_logits, dim=-1)
                entropy = -torch.sum(selected_probs * selected_log_probs, dim=-1)
                unmask_loss = entropy.mean()
            else:
                unmask_loss = 0.0 * logits.sum()
        else:
            unmask_loss = 0.0 * logits.sum()

        if self.w_remask_loss > 0.0 and rs_indices.sum() > 0:
            rs_logits_all = logits[block_rs_indices]
            rs_labels_all = labels[block_rs_indices]

            with torch.no_grad():
                rs_probs_all = F.softmax(rs_logits_all, dim=-1)
                confidences, predictions = rs_probs_all.max(dim=-1)
                is_wrong = predictions != rs_labels_all
                is_high_conf = confidences > self.threshold
                target_error_indices = is_wrong & is_high_conf

            num_remask_loss = target_error_indices.sum()
            if target_error_indices.sum() > 0:
                selected_logits = rs_logits_all[target_error_indices]
                selected_probs = F.softmax(selected_logits, dim=-1)
                selected_log_probs = F.log_softmax(selected_logits, dim=-1)
                entropy = -torch.sum(selected_probs * selected_log_probs, dim=-1)
                remask_loss = -entropy.mean()
            else:
                remask_loss = 0.0 * logits.sum()
        else:
            remask_loss = 0.0 * logits.sum()

        total_loss = (
            self.w_ce_loss * ce_loss
            + self.w_unmask_loss * unmask_loss
            + self.w_remask_loss * remask_loss
        )

        if self.state.global_step <= 10 and self.is_world_process_zero():
            print(
                f"[Step {self.state.global_step}] Loss Breakdown:",
                {
                    "total": total_loss,
                    "ce": ce_loss.item(),
                    "unmask": unmask_loss.item(),
                    "remask": remask_loss.item(),
                },
            )

        if self.is_world_process_zero():
            total_val = _to_number(total_loss)
            ce_val = _to_number(ce_loss)
            unmask_val = _to_number(unmask_loss)
            remask_val = _to_number(remask_loss)
            num_masked_val = _to_number(num_mask)
            num_ce_loss_val = _to_number(num_ce_loss)
            num_unmask_loss_val = _to_number(num_unmask_loss)
            num_remask_loss_val = _to_number(num_remask_loss)

            if self.loss_log_file is not None:
                with self.loss_log_file.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{self.state.global_step},{total_val},{ce_val},{unmask_val},"
                        f"{remask_val},{num_masked_val},{num_ce_loss_val},"
                        f"{num_unmask_loss_val},{num_remask_loss_val}\n"
                    )

            if self.state.global_step % self.args.logging_steps == 0:
                print(
                    f"[Step {self.state.global_step}] Loss Breakdown:",
                    {
                        "total": f"{total_val:.4f}",
                        "ce": f"{ce_val:.4f}",
                        "unmask": f"{unmask_val:.4f}",
                        "remask": f"{remask_val:.4f}",
                        "num_masked": int(num_masked_val),
                        "num_remasked": int(num_remask),
                        "num_ce_loss": int(num_ce_loss_val),
                        "num_unmask_loss": int(num_unmask_loss_val),
                        "num_remask_loss": int(num_remask_loss_val),
                    },
                )

        return (total_loss, outputs) if return_outputs else total_loss


def _to_number(value: Any) -> float:
    if torch.is_tensor(value):
        return value.item()
    return float(value)


def select_trajectory(examples: Dict[str, Any], idx: int, trajectory_field: Optional[str]):
    if trajectory_field is not None:
        return examples[trajectory_field][idx]
    for key in ("trajectory_accepted", "wino_trajectory", "trajectory_proposed"):
        if key in examples:
            return examples[key][idx]
    raise KeyError("No trajectory field found. Expected trajectory_accepted, wino_trajectory, or trajectory_proposed.")


def process_wino_step_batch(
    examples,
    indices,
    mask_token_id: int = 126336,
    block_length: int = 128,
    trajectory_field: Optional[str] = None,
):
    new_batch = {
        "trajectory_id": [],
        "input_ids": [],
        "labels": [],
        "prompt_lengths": [],
        "target_mask": [],
        "block_num": [],
    }

    for idx in range(len(examples["prompt_ids"])):
        if "correct" in examples and not examples["correct"][idx]:
            continue

        prompt_ids = examples["prompt_ids"][idx]
        generated_ids = examples["generated_ids"][idx]
        trajectory = select_trajectory(examples, idx, trajectory_field)
        current_traj_id = indices[idx]

        prompt_len = len(prompt_ids)
        min_len = min(len(generated_ids), len(trajectory))
        if min_len == 0:
            continue

        gen_arr = np.array(generated_ids[:min_len])
        traj_arr = np.array(trajectory[:min_len])
        max_step = traj_arr.max()
        full_target_ids = prompt_ids + gen_arr.tolist()
        prompt_mask = np.zeros(prompt_len, dtype=bool)

        for step in range(max_step + 1):
            keep_mask = traj_arr < step
            masked_gen_arr = np.full_like(gen_arr, mask_token_id)
            masked_gen_arr[keep_mask] = gen_arr[keep_mask]
            input_ids = prompt_ids + masked_gen_arr.tolist()

            current_step_mask_gen = traj_arr == step
            if not np.any(current_step_mask_gen):
                continue

            full_target_mask = np.concatenate([prompt_mask, current_step_mask_gen])
            relative_indices = np.where(current_step_mask_gen)[0]
            current_block_num = int(relative_indices[0] // block_length)

            new_batch["input_ids"].append(input_ids)
            new_batch["labels"].append(full_target_ids)
            new_batch["prompt_lengths"].append(prompt_len)
            new_batch["target_mask"].append(full_target_mask.tolist())
            new_batch["block_num"].append(current_block_num)
            new_batch["trajectory_id"].append(current_traj_id)

    return new_batch


@dataclass
class MaskDiffusionDataCollator:
    tokenizer: Any
    max_length: int = 448

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.eos_token_id
        target_length = self.max_length

        padded_input_ids = []
        padded_labels = []
        padded_target_masks = []

        for feature in features:
            input_ids = feature["input_ids"][:target_length]
            labels = feature["labels"][:target_length]
            target_mask = feature["target_mask"][:target_length]

            pad_len = target_length - len(input_ids)
            padded_input_ids.append(torch.tensor(input_ids + [pad_token_id] * pad_len, dtype=torch.long))
            padded_labels.append(torch.tensor(labels + [pad_token_id] * pad_len, dtype=torch.long))
            padded_target_masks.append(torch.tensor(target_mask + [False] * pad_len, dtype=torch.bool))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "target_mask": torch.stack(padded_target_masks),
            "prompt_lengths": torch.tensor([f["prompt_lengths"] for f in features], dtype=torch.long),
            "block_num": torch.tensor([f["block_num"] for f in features], dtype=torch.long),
            "trajectory_id": torch.tensor([f["trajectory_id"] for f in features], dtype=torch.long),
        }


def build_stage_dataset(stage: Dict[str, Any], config: Dict[str, Any]):
    data_cfg = config.get("data", {})
    cache_dir = resolve_path(
        stage.get("processed_cache_dir")
        or Path(data_cfg.get("cache_root", "outputs/llada_wino_plus/cache")) / stage["name"]
    )
    assert cache_dir is not None

    if cache_dir.exists():
        print(f"Loading processed dataset from: {cache_dir}")
        formatted_dataset = load_from_disk(str(cache_dir))
    elif not is_main_process():
        print(f"Rank {get_rank()} waiting for processed dataset cache: {cache_dir}")
        wait_for_path(cache_dir / "dataset_info.json")
        formatted_dataset = load_from_disk(str(cache_dir))
    else:
        trajectory_file = resolve_path(stage["trajectory_file"])
        if trajectory_file is None:
            raise ValueError(f"Stage {stage['name']} is missing trajectory_file")
        print(f"Loading trajectory JSONL from: {trajectory_file}")
        dataset = load_dataset("json", data_files=str(trajectory_file), split="train")
        print(f"Raw rows: {len(dataset)}")

        if "correct" in dataset.column_names:
            dataset = dataset.filter(lambda row: row["correct"] is True)
            print(f"Rows after correct=True filtering: {len(dataset)}")

        formatted_dataset = dataset.map(
            process_wino_step_batch,
            batched=True,
            batch_size=data_cfg.get("map_batch_size", 100),
            with_indices=True,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "mask_token_id": data_cfg.get("mask_token_id", 126336),
                "block_length": data_cfg.get("block_length", 128),
                "trajectory_field": stage.get("trajectory_field"),
            },
        )
        print(f"Expanded rows: {len(formatted_dataset)}")

        max_prompt_length = data_cfg.get("max_prompt_length")
        if max_prompt_length is not None:
            formatted_dataset = formatted_dataset.filter(lambda row: row["prompt_lengths"] <= max_prompt_length)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        formatted_dataset.save_to_disk(str(cache_dir))

    distributed_barrier()

    max_prompt_length = data_cfg.get("max_prompt_length")
    if max_prompt_length is not None:
        formatted_dataset = formatted_dataset.filter(lambda row: row["prompt_lengths"] <= max_prompt_length)

    train_size = data_cfg.get("train_size")
    if train_size is not None and len(formatted_dataset) > 1:
        split = formatted_dataset.train_test_split(
            train_size=train_size,
            shuffle=data_cfg.get("shuffle_train_split", False),
        )
        formatted_dataset = split["train"]

    print(f"Training rows for stage '{stage['name']}': {len(formatted_dataset)}")
    return formatted_dataset


def build_training_args(stage: Dict[str, Any], config: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    training_cfg = dict(config["training_common"])
    training_cfg.update(stage.get("training", {}))
    training_cfg["output_dir"] = str(output_dir)
    return TrainingArguments(
        **training_cfg,
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def resolve_stage_init_adapter(stage: Dict[str, Any], previous_adapter: Optional[Path]) -> Optional[Path]:
    init_adapter = stage.get("init_adapter")
    if init_adapter == "previous":
        if previous_adapter is None:
            raise ValueError(f"Stage {stage['name']} requested init_adapter=previous, but no previous adapter exists")
        return previous_adapter
    if init_adapter:
        return resolve_path(init_adapter)
    return None


def run_stage(stage: Dict[str, Any], config: Dict[str, Any], previous_adapter: Optional[Path]) -> Path:
    print("\n" + "=" * 80)
    print(f"Starting LLaDA WINO+ stage: {stage['name']}")
    print("=" * 80)

    output_root = resolve_path(config.get("output_root", "outputs/llada_wino_plus"))
    output_dir = resolve_path(stage.get("output_dir")) or output_root / stage["name"]
    loss_log_file = resolve_path(stage.get("loss_log_file")) or output_dir / "loss_log.csv"
    assert output_dir is not None

    init_adapter = resolve_stage_init_adapter(stage, previous_adapter)
    model, tokenizer = prepare_model(config, init_adapter=init_adapter)
    dataset = build_stage_dataset(stage, config)
    training_args = build_training_args(stage, config, output_dir)

    collator = MaskDiffusionDataCollator(
        tokenizer=tokenizer,
        max_length=config.get("data", {}).get("collator_max_length", 448),
    )

    if len(dataset) >= 2:
        debug_batch = collator([dataset[i] for i in range(2)])
        for key, value in debug_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Debug batch {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    trainer = DLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        mask_token_id=config.get("loss", {}).get("mask_token_id", 126336),
        loss_config=config.get("loss", {}),
        loss_log_file=loss_log_file,
    )

    trainer.train()

    final_adapter_dir = output_dir / "final_adapter"
    if trainer.is_world_process_zero():
        trainer.save_model(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))
    distributed_barrier()
    wait_for_path(final_adapter_dir)

    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if last_checkpoint:
        print(f"Last trainer checkpoint for stage '{stage['name']}': {last_checkpoint}")
    print(f"Final adapter for stage '{stage['name']}': {final_adapter_dir}")

    del trainer, model, tokenizer, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_adapter_dir


def validate_config(config: Dict[str, Any]) -> None:
    for key in ("model", "lora", "training_common", "stages"):
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    if not config["stages"]:
        raise ValueError("Config must define at least one stage")
    for stage in config["stages"]:
        for key in ("name", "trajectory_file"):
            if key not in stage:
                raise ValueError(f"Stage is missing required field: {key}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage LLaDA WINO+ LoRA training")
    parser.add_argument(
        "--config",
        type=str,
        default="training/llada/config/llada_wino_plus_two_stage.yaml",
        help="Path to the two-stage training YAML config",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config) or args.config)
    validate_config(config)

    previous_adapter = None
    for stage in config["stages"]:
        previous_adapter = run_stage(stage, config, previous_adapter)

    print(f"All stages complete. Final adapter: {previous_adapter}")


if __name__ == "__main__":
    main()
