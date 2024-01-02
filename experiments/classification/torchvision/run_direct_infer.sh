#!/usr/bin/env bash

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
    --round)
      ROUND="$2"
      shift # past argument
      shift # past value
      ;;
    --no-mx)
      NO_MX="--no-mx"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CUR_DIR=$(pwd)
DATA_PATH=/group/modelzoo/test_dataset/Imagenet

declare -A WEIGHTS
WEIGHTS["resnet18"]="ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS["resnet50"]="ResNet50_Weights.IMAGENET1K_V1"
WEIGHTS["mobilenet_v2"]="MobileNet_V2_Weights.IMAGENET1K_V2"

MODEL=${MODEL:-resnet18}
if [[ ! -n "${WEIGHTS[${MODEL}]}" ]]; then
 echo "Not a valid model: '${MODEL}'"
 exit 1
fi
#RESNET18_CONFIG="--model resnet18 --weights ResNet18_Weights.IMAGENET1K_V2 --use-v2"
#RESNET50_CONFIG="--model resnet50 --weights ResNet50_Weights.IMAGENET1K_V2 --use-v2"
#MOBILENETV2_CONFIG="--model mobilenet_v2 --weights MobileNet_V2_Weights.IMAGENET1K_V2 --use-v2"
MODEL_CONFIG="--model ${MODEL} --weights ${WEIGHTS[${MODEL}]} --use-v2"

W_ELEM_FORMAT="int8"
#W_ELEM_FORMAT="fp8_e4m3"
#W_ELEM_FORMAT="fp8_e5m2"
#W_ELEM_FORMAT="fp6_e2m3"
#W_ELEM_FORMAT="fp6_e3m2"
#W_ELEM_FORMAT="fp4_e2m1"
W_ELEM_FORMAT=${FORMAT:-$W_ELEM_FORMAT}
A_ELEM_FORMAT=${W_ELEM_FORMAT}

ROUND=${ROUND:-even}
MX_SPECS="--w_elem_format ${W_ELEM_FORMAT} --a_elem_format ${A_ELEM_FORMAT} --scale_bits 8 --block_size 32 --bfloat 16 --round_mx_output ${ROUND} --custom_cuda"

CMD="PYTHONPATH=${CUR_DIR}/../../../ python -u train.py --test-only --data-path ${DATA_PATH} ${MODEL_CONFIG} ${MX_SPECS} ${NO_MX}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${MODEL}-${FORMAT}-${ROUND}-${NO_MX}.$(date +%Y_%m%d_%H%M)"

echo "${CMD}" | tee ${LOG_FILE}
eval "${CMD}" 2>&1 | tee -a ${LOG_FILE}
