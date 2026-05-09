#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
cd "$(dirname "$SCRIPT_PATH")/.."

if [ -f dev/bin/activate ]; then
    source dev/bin/activate
fi

GEN_METHOD=confidence_threshold
GEN_LENGTH=${GEN_LENGTH:-256}
DIFF_STEP=${DIFF_STEP:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-128}
NGPU=${NGPU:-1}
PORT=${PORT:-12345}
MODEL_PATH=${MODEL_PATH:-/PATH/TO/MMaDA-WINOPlus-MERGED-MODEL}
OUTPUT_PATH=${OUTPUT_PATH:-"./results"}

# WINO+ confidence thresholds
TH_FLICKR30K=0.7
TH_AI2D=0.7
TH_MATHVISION=0.7
TH_MATHVISTA=0.5
TH_MMMU=0.55
TH_SCIENCEQA=0.6

COMMON_ARGS="pretrained=${MODEL_PATH},gen_method=${GEN_METHOD},gen_length=${GEN_LENGTH},diff_step=${DIFF_STEP},block_length=${BLOCK_LENGTH}"

# Running on Flickr30K
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_FLICKR30K},reasoning=False \
    --tasks flickr30k_test_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"

# Running on AI2D
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_AI2D},reasoning=True \
    --tasks ai2d_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"

# Running on MATH-Vision
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_MATHVISION},reasoning=True \
    --tasks mathvision_test_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"

# Running on MathVista-mini
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_MATHVISTA},reasoning=True \
    --tasks mathvista_testmini_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"

# Running on MMMU-val
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_MMMU},reasoning=True \
    --tasks mmmu_val_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"

# Running on ScienceQA-Img
accelerate launch --num_processes=${NGPU} --main_process_port=${PORT} -m lmms_eval \
    --model mmada \
    --model_args=${COMMON_ARGS},threshold=${TH_SCIENCEQA},reasoning=True \
    --tasks scienceqa_img_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_PATH}"
