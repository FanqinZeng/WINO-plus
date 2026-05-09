import argparse
import gc
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from MMaDA.models import MMadaModelLM
from MMaDA.models.lr_schedulers import get_scheduler
from MMaDA.models.utils import AverageMeter, flatten_omega_conf

from .trajectory_collator import TrajectoryDataCollator
from .trajectory_dataset import TrajectoryDataset
from .trajectory_trainer import TrajectoryTrainer


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MMaDA WINO+ LoRA adapters from trajectory JSONL.")
    parser.add_argument(
        "--config",
        default="training/mmada/config/mmada_wino_plus_lora.yaml",
        help="Path to the YAML config file.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_config(config_path: str, overrides: list[str]):
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def resolve_path(path_value: str | Path | None) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def prepare_accelerator(config):
    logging_dir = resolve_path(config.experiment.output_dir) / "logs"
    log_with = "wandb" if config.experiment.get("enable_wandb", False) else None
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=log_with,
        project_dir=str(logging_dir),
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.batch_size_trajectory
        )
    return accelerator, logging_dir


def setup_logging(config, accelerator, logging_dir: Path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    output_dir = resolve_path(config.experiment.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")

        if config.experiment.get("enable_wandb", False):
            run_id = config.wandb.get("run_id", wandb.util.generate_id())
            wandb_init_kwargs = {
                "name": config.experiment.name,
                "id": run_id,
                "resume": config.wandb.resume,
                "entity": config.wandb.get("entity", None),
                "config_exclude_keys": [],
            }
            wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
            accelerator.init_trackers(
                config.experiment.project,
                config=wandb_config,
                init_kwargs={"wandb": wandb_init_kwargs},
            )


def prepare_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    if config.training.get("gradient_checkpointing", True):
        if not getattr(model, "supports_gradient_checkpointing", False):
            model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for MMaDA.")

    lora_cfg = config.get("lora", {})
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if hasattr(target_modules, "__iter__") and not isinstance(target_modules, str):
        target_modules = list(target_modules)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 128),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def build_optimizer(model, config):
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": config.optimizer.params.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(
        optimizer_grouped_parameters,
        lr=config.optimizer.params.learning_rate,
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2),
        weight_decay=config.optimizer.params.weight_decay,
        eps=config.optimizer.params.epsilon,
    )


def build_dataloader(config, tokenizer, mask_id: int):
    dataset = TrajectoryDataset(
        data_path=str(resolve_path(config.dataset.params.train_trajectory_path)),
        mask_token_id=mask_id,
        block_length=config.trajectory.block_size,
        trajectory_mode=config.trajectory.get("mode", "original"),
    )
    collator = TrajectoryDataCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        ignore_id=config.dataset.preprocessing.get("ignore_id", -100),
        max_length=config.dataset.preprocessing.max_seq_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_trajectory,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.dataset.params.get("num_workers", 4),
        pin_memory=config.dataset.params.get("pin_memory", True),
    )
    return dataset, dataloader


def _number(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def log_step(accelerator, config, lr_scheduler, loss_dict: Dict[str, Any], batch_time_m: AverageMeter, step: int):
    samples_per_second_per_gpu = (
        config.training.gradient_accumulation_steps * config.training.batch_size_trajectory / max(batch_time_m.val, 1e-12)
    )
    log_values = {
        "loss/total": _number(loss_dict["total_loss"]),
        "loss/tok": _number(loss_dict["ce_loss"]),
        "loss/sharp": _number(loss_dict["unmask_loss"]),
        "loss/defer": _number(loss_dict["remask_loss"]),
        "stats/num_masked": loss_dict["loss_dict"]["num_masked"],
        "stats/num_deferred": loss_dict["loss_dict"]["num_remasked"],
        "lr": lr_scheduler.get_last_lr()[0],
        "samples/sec/gpu": samples_per_second_per_gpu,
    }
    if config.experiment.get("enable_wandb", False):
        accelerator.log(log_values, step=step)
    if accelerator.is_main_process:
        logger.info(
            "Step: %s Loss: %.4f CE: %.4f Sharp: %.4f Defer: %.4f LR: %.6f",
            step,
            log_values["loss/total"],
            log_values["loss/tok"],
            log_values["loss/sharp"],
            log_values["loss/defer"],
            log_values["lr"],
        )


def save_lora_adapter(accelerator, model, tokenizer, output_dir: Path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        import deepspeed

        trainable_params = [p for _, p in unwrapped_model.named_parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0):
            if accelerator.is_main_process:
                lora_state_dict = {
                    name: param.data.cpu().clone()
                    for name, param in unwrapped_model.named_parameters()
                    if param.requires_grad and "lora" in name.lower()
                }
                peft_state_dict = get_peft_model_state_dict(unwrapped_model, state_dict=lora_state_dict)
                save_file(peft_state_dict, output_dir / "adapter_model.safetensors")
                unwrapped_model.peft_config["default"].save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
    elif accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def main():
    args, overrides = parse_args()
    config = load_config(args.config, overrides)

    if config.training.get("enable_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator, logging_dir = prepare_accelerator(config)
    setup_logging(config, accelerator, logging_dir)

    if config.training.get("seed", None) is not None:
        set_seed(config.training.seed)

    model, tokenizer = prepare_model(config)
    mask_id = model.base_model.model.config.mask_token_id if hasattr(model, "base_model") else model.config.mask_token_id
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = build_optimizer(model, config)
    dataset, dataloader = build_dataloader(config, tokenizer, mask_id)
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.get("warmup_steps", 0),
        min_lr_scale=config.lr_scheduler.params.get("min_lr_scale", 0.0),
    )

    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)
    trajectory_trainer = TrajectoryTrainer(
        mask_token_id=mask_id,
        block_size=config.trajectory.block_size,
        w_ce_loss=config.trajectory.w_ce_loss,
        w_unmask_loss=config.trajectory.w_unmask_loss,
        w_remask_loss=config.trajectory.w_remask_loss,
        threshold=config.trajectory.threshold,
        threshold_back=config.trajectory.threshold_back,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / max(num_update_steps_per_epoch, 1))
    total_batch_size = (
        config.training.batch_size_trajectory * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    logger.info("***** Running MMaDA WINO+ training *****")
    logger.info("  Num examples = %s", len(dataset))
    logger.info("  Num epochs = %s", num_train_epochs)
    logger.info("  Total train batch size = %s", total_batch_size)

    model.train()
    progress_bar = tqdm(total=config.training.max_train_steps, disable=not accelerator.is_local_main_process)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    global_step = 0

    for _ in range(num_train_epochs):
        for batch in dataloader:
            data_time_m.update(time.time() - end)
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["input_ids"])
                loss_inputs = {
                    "input_ids": batch["input_ids"],
                    "labels": batch["labels"],
                    "target_mask": batch["target_mask"],
                    "block_num": batch["block_num"],
                    "prompt_lengths": batch["prompt_lengths"],
                }
                loss_dict = trajectory_trainer.compute_loss(inputs=loss_inputs, logits=outputs.logits)
                loss = loss_dict["total_loss"]
                accelerator.backward(loss)

                if config.training.get("max_grad_norm", None) is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                batch_time_m.update(time.time() - end)
                end = time.time()

                if global_step % config.experiment.log_every == 0:
                    log_step(accelerator, config, lr_scheduler, loss_dict, batch_time_m, global_step)
                    batch_time_m.reset()
                    data_time_m.reset()

                if config.experiment.save_every and global_step % config.experiment.save_every == 0:
                    save_lora_adapter(
                        accelerator,
                        model,
                        tokenizer,
                        resolve_path(config.experiment.output_dir) / f"checkpoint-{global_step}",
                    )

            if global_step >= config.training.max_train_steps:
                break
        if global_step >= config.training.max_train_steps:
            break

    save_lora_adapter(accelerator, model, tokenizer, resolve_path(config.experiment.output_dir))
    accelerator.end_training()

    del model, tokenizer, dataloader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
