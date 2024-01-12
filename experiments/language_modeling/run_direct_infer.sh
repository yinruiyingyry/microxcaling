#!/usr/bin/env bash

function usage() {
  echo "Usage: $0 --model facebook/opt-350m --format int8 --round_weight nearest --round_output nearest --round_mx_output even"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    --format)
      FORMAT="$2"
      shift # past argument
      shift # past value
      ;;
    --round_weight)
      ROUND_WEIGHT="$2"
      shift # past argument
      shift # past value
      ;;
    --round_output)
      ROUND_OUTPUT="$2"
      shift # past argument
      shift # past value
      ;;
    --round_mx_output)
      ROUND_MX_OUTPUT="$2"
      shift # past argument
      shift # past value
      ;;
    --no_mx)
      NO_MX="--no_mx True"
      shift # past argument
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-facebook/opt-350m}
#MODEL=/group/modelzoo/sequence_learning/weights/nlp-pretrained-model/llama-2-7b-hf
DATA_NAME_OR_DIR=wikitext
OUTPUT_DIR=./output/test_clm

W_ELEM_FORMAT="int8"
#W_ELEM_FORMAT="fp8_e4m3"
#W_ELEM_FORMAT="fp8_e5m2"
#W_ELEM_FORMAT="fp6_e2m3"
#W_ELEM_FORMAT="fp6_e3m2"
#W_ELEM_FORMAT="fp4_e2m1"
W_ELEM_FORMAT=${FORMAT:-$W_ELEM_FORMAT}
A_ELEM_FORMAT=${W_ELEM_FORMAT}

ROUND_WEIGHT=${ROUND:-nearest}
ROUND_OUTPUT=${ROUND:-nearest}
ROUND_MX_OUTPUT=${ROUND:-even}
MX_SPECS="--w_elem_format ${W_ELEM_FORMAT} --a_elem_format ${A_ELEM_FORMAT} \
    --round_weight ${ROUND_WEIGHT} --round_output ${ROUND_OUTPUT} --round_mx_output ${ROUND_MX_OUTPUT} \
    --scale_bits 8 --block_size 32 --bfloat 16 --custom_cuda ${NO_MX}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
MODEL_NAME="$(echo ${MODEL} | awk -F'/' '{print $(NF)}')"
if [[ -v NO_MX ]]; then
  FORMAT="float"
fi
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${FORMAT}.$(date +%Y_%m%d_%H%M)"

set -x
#export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="${CUR_DIR}/../../" python ${CUR_DIR}/run_clm.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATA_NAME_OR_DIR} \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_eval \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --torch_dtype float32 \
    --input_block_size 2048 \
    ${MX_SPECS} 2>&1 | tee ${LOG_FILE}
