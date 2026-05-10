# WINO-DLLM

<p>
  <a href="https://openreview.net/pdf?id=XtLQHlNLxy"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=flat" alt="ICLR 2026"></a>
</p>

Official implementation of **Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs**.

This repository provides scripts and instructions to evaluate [WINO](https://openreview.net/pdf?id=XtLQHlNLxy) on LLaDA and MMaDA.


## Related Project

We are continuing to improve efficient DLLM inference with **ReMix**: [Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](https://github.com/Feng-Hong/ReMix-DLLM), accepted to CVPR 2026. ReMix is a training-free decoding method that further explores fast semantic propagation for mask tokens, and it provides unified evaluation scripts for both LLaDA and MMaDA.

## Evaluation of WINO on LLaDA

1. Installation
We recommend using [uv](https://github.com/astral-sh/uv) for dependency and virtual environment management.
```bash
pipx install uv # or pip install uv
cd LLaDA
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```

2. Prepare Model and Datasets

Before running inference or evaluation, please download the following models and datasets from [Hugging Face](https://huggingface.co/) into the specified local directories (e.g., [`./LLaDA/models/`](./LLaDA/models/) and [`./LLaDA/data/`](./LLaDA/data/)). 

You may use either `huggingface-cli` or the Python `datasets` library to complete the download.

| Model Name         | Hugging Face Repo                                               | Local Path                     |
|--------------------|------------------------------------------------------------------|--------------------------------|
| LLaDA-8B-Instruct  | [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `./LLaDA/models/LLaDA-8B-Instruct/`  |

| Dataset Name  | Hugging Face Repo                                                                 | Local Path          |
|---------------|------------------------------------------------------------------------------------|---------------------|
| GSM8K         | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)                    | `./LLaDA/data/gsm8k/`     |
| MATH-500      | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `./LLaDA/data/math500/`   |
| HumanEval     | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) | `./LLaDA/data/humaneval/` |
| ai2_arc     | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)              | `./LLaDA/data/ai2_arc/`       |

Datasets not listed above are already included in the [`./LLaDA/data/`](./LLaDA/data/) directory

3. Quick Demo

Please make sure to set the correct model path in generate.py.

```bash
python generate.py
```
4. Evaluation

To evaluate WINO on a benchmark such as GSM8K. Please configure the model and data paths in the corresponding config file.

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config ./configs/gsm8k.yaml
```
All available config files can be found in the [`./LLaDA/configs/`](./LLaDA/configs/) directory.



## Evaluation of WINO on MMaDA

We evaluate **WINO** using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

To run the evaluation, follow these steps:

1. **Install MMaDA dependencies**
```bash
cd MMaDA
# pipx install uv
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```

A quick inference demo can be performed after this step.
```bash
python generate_demo.py
```

2. **Install lmms-eval dependencies**
```bash
cd lmms_eval
uv pip install -e .
```

3. **Set some necessary environmental variables**
   Some environmental variables are necessary for certain tasks to run.
```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
```

Once all dependencies are installed and your API key is set, you can run the evaluation script directly:

```bash
cd ..
# Evaluating MMaDA on the reported six multimodel benchmarks
bash scripts/eval_baseline.sh
# Evaluating WINO on the reported six multimodel benchmarks
bash scripts/eval_wino.sh
```

## WINO+ Trajectory Data Preparation

WINO+ post-training uses offline WINO trajectories. The lightweight data preparation code is under
[`prepare_trainingdata/`](./prepare_trainingdata/). It currently supports GSM8K, Countdown, and IconQA.

All trajectory collectors write JSONL records with training-facing fields such as:

```json
{
  "prompt_ids": [1, 2, 3],
  "generated_ids": [4, 5, 6],
  "trajectory_accepted": [0, 0, 1],
  "trajectory_proposed": [0, 0, 1],
  "correct": true
}
```

Example: collect LLaDA GSM8K trajectories.

```bash
python -m prepare_trainingdata.llada.prepare_gsm8k \
  --model-path /path/to/LLaDA-8B-Instruct \
  --output-file ./data/gsm8k_processed.jsonl

python -m prepare_trainingdata.llada.collect_gsm8k_trajectories \
  --model-path /path/to/LLaDA-8B-Instruct \
  --input-file ./data/gsm8k_processed.jsonl \
  --output-file ./data/gsm8k_wino_trajectory.jsonl
```

Example: collect MMaDA IconQA trajectories.

```bash
python -m prepare_trainingdata.mmada.prepare_iconqa \
  --model-path /path/to/MMaDA-8B-MixCoT \
  --input-file /path/to/iconqa_train_dataset.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_processed.jsonl

python -m prepare_trainingdata.mmada.collect_iconqa_trajectories \
  --mmada-model-path /path/to/MMaDA-8B-MixCoT \
  --vq-model-path showlab/magvitv2 \
  --input-file ./data/iconqa_processed.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_wino_trajectory.jsonl
```

Filter correct trajectories before training:

```bash
python -m prepare_trainingdata.common.filter_trajectories \
  --input-file ./data/iconqa_wino_trajectory.jsonl \
  --output-file ./data/iconqa_wino_trajectory_filtered.jsonl
```

See [`prepare_trainingdata/README.md`](./prepare_trainingdata/README.md) for task-specific details.

## WINO+ LoRA Training

WINO+ training uses separate `uv` environments from the LLaDA and MMaDA evaluation environments above. Do not reuse
or modify existing external project environments when setting up these training runs.

Create a dedicated LLaDA training environment:

```bash
cd /path/to/WINO-DLLM
uv venv --python 3.10 training/llada/.venv
source training/llada/.venv/bin/activate
uv pip install -r training/llada/requirements.txt
deactivate
```

Create a dedicated MMaDA training environment:

```bash
cd /path/to/WINO-DLLM
uv venv --python 3.11 training/mmada/.venv
source training/mmada/.venv/bin/activate
uv pip install -r training/mmada/requirements.txt
deactivate
```

Activate only the matching training environment before launching each trainer.

### LLaDA WINO+ Training

The LLaDA trainer supports the two-stage setup used in the paper: first train on GSM8K trajectories, then continue
from the first adapter on Countdown trajectories.

Edit [`training/llada/config/llada_wino_plus_two_stage.yaml`](./training/llada/config/llada_wino_plus_two_stage.yaml)
to set the base model path, trajectory files, and output directories, then run:

```bash
source training/llada/.venv/bin/activate
python -m training.llada.train_wino_plus_lora \
  --config training/llada/config/llada_wino_plus_two_stage.yaml
```

### MMaDA WINO+ Training

The MMaDA trainer consumes tokenized trajectory JSONL files whose `prompt_ids` already contain image tokens and the
text prompt. It does not reload the VQ model during training.

For 8 GPU DeepSpeed ZeRO-3 training:

```bash
source training/mmada/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --config_file training/mmada/accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  -m training.mmada.train_wino_plus_lora \
  --config training/mmada/config/mmada_wino_plus_lora.yaml
```

You can override config values from the command line:

```bash
source training/mmada/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --config_file training/mmada/accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  -m training.mmada.train_wino_plus_lora \
  --config training/mmada/config/mmada_wino_plus_lora.yaml \
  model.mmada.tokenizer_path=/path/to/LLaDA-8B-Instruct \
  model.mmada.pretrained_model_path=/path/to/MMaDA-8B-MixCoT \
  dataset.params.train_trajectory_path=/path/to/iconqa_wino_trajectory_filtered.jsonl
```

For a short smoke test on real trajectory data, add:

```bash
training.max_train_steps=1 \
experiment.output_dir=/tmp/mmada_wino_plus_smoke
```

## Merge LoRA Adapters

After WINO+ LoRA training, merge a single adapter into the base model for evaluation.

Merge LLaDA LoRA:

```bash
source training/llada/.venv/bin/activate
python -m training.llada.merge_lora \
  --base-model /path/to/LLaDA-8B-Instruct \
  --adapter /path/to/llada/final_adapter_or_checkpoint \
  --output-dir /path/to/merged-llada-winoplus
```

Merge MMaDA LoRA:

```bash
source training/mmada/.venv/bin/activate
python -m training.mmada.merge_lora \
  --base-model /path/to/MMaDA-8B-MixCoT \
  --adapter /path/to/mmada/adapter \
  --output-dir /path/to/merged-mmada-winoplus
```

## Evaluation of WINO+ Models

WINO remains available through the original evaluation commands above. For WINO+ LoRA-merged models, use the
confidence-threshold decoding path.

LLaDA configs support:

```yaml
method: confidence_threshold
```

MMaDA confidence-threshold evaluation can be launched with:

```bash
cd MMaDA
MODEL_PATH=/path/to/merged-mmada-winoplus \
NGPU=8 \
bash scripts/eval_winoplus.sh
```
