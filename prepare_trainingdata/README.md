# WINO+ Trajectory Data Preparation

This directory contains lightweight trajectory collection code for the paper
datasets used in WINO+ post-training:

- GSM8K
- Countdown
- IconQA

It intentionally does not include raw datasets, model weights, cached
trajectories, worker outputs, or copied model implementations.

## Output Schema

All collection scripts write JSONL records with the same training-facing fields:

```json
{
  "unique_id": "sample-id",
  "source": "gsm8k|countdown|iconqa",
  "prompt_ids": [1, 2, 3],
  "generated_ids": [4, 5, 6],
  "trajectory_accepted": [0, 0, 1],
  "trajectory_proposed": [0, 0, 1],
  "correct": true,
  "prompt_length": 128,
  "generated_text": "...",
  "decoding_steps": 12,
  "used_temperature": 0.0,
  "metadata": {}
}
```

`trajectory_accepted` is the field used by WINO+ training. Older GSM8K records
that used `wino_trajectory` are normalized to `trajectory_accepted`.

## GSM8K

Prepare prompts from Hugging Face GSM8K:

```bash
python -m prepare_trainingdata.llada.prepare_gsm8k \
  --model-path /path/to/LLaDA-8B-Instruct \
  --output-file ./data/gsm8k_processed.jsonl
```

Collect trajectories:

```bash
python -m prepare_trainingdata.llada.collect_gsm8k_trajectories \
  --model-path /path/to/LLaDA-8B-Instruct \
  --input-file ./data/gsm8k_processed.jsonl \
  --output-file ./data/gsm8k_wino_trajectory.jsonl
```

## Countdown

Prepare prompts from a local directory of Countdown parquet files:

```bash
python -m prepare_trainingdata.llada.prepare_countdown \
  --model-path /path/to/LLaDA-8B-Instruct \
  --input-parquet-dir /path/to/countdown/parquet_dir \
  --output-file ./data/countdown_processed.jsonl
```

Collect trajectories:

```bash
python -m prepare_trainingdata.llada.collect_countdown_trajectories \
  --model-path /path/to/LLaDA-8B-Instruct \
  --input-file ./data/countdown_processed.jsonl \
  --output-file ./data/countdown_wino_trajectory.jsonl
```

## IconQA

Prepare prompts from a local IconQA JSONL and image directory:

```bash
python -m prepare_trainingdata.mmada.prepare_iconqa \
  --model-path /path/to/MMaDA-8B-MixCoT \
  --input-file /path/to/iconqa_train_dataset.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_processed.jsonl
```

Collect trajectories. The default judge is local rule-based matching and does
not require an API key.

```bash
python -m prepare_trainingdata.mmada.collect_iconqa_trajectories \
  --mmada-model-path /path/to/MMaDA-8B-MixCoT \
  --vq-model-path showlab/magvitv2 \
  --input-file ./data/iconqa_processed.jsonl \
  --image-root /path/to/iconqa/images \
  --output-file ./data/iconqa_wino_trajectory.jsonl
```

To use GPT-style judging, pass `--judge gpt` and set `OPENAI_API_KEY`. Optional
environment variables are `OPENAI_API_URL` and `MODEL_VERSION`.

## Filtering

Filter correct trajectories and optionally cap by prompt length or cumulative
trajectory steps:

```bash
python -m prepare_trainingdata.common.filter_trajectories \
  --input-file ./data/countdown_wino_trajectory.jsonl \
  --output-file ./data/countdown_wino_trajectory_filtered.jsonl
```

