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
      NO_MX="--no-mx"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH=/group/modelzoo/test_dataset/Imagenet

MODEL=${MODEL:-deit_tiny_patch16_224}

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

CMD="PYTHONPATH=${CUR_DIR}/../../../ MX_ASSERT=True python -u ${CUR_DIR}/main.py \
    --eval --model ${MODEL} --data-path ${DATA_PATH} ${MX_SPECS}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
if [[ -v NO_MX ]]; then
  FORMAT="float"
fi
LOG_FILE="${LOG_DIR}/${MODEL}-${FORMAT}.$(date +%Y_%m%d_%H%M)"

echo "${CMD}" | tee ${LOG_FILE}
eval "${CMD}" 2>&1 | tee -a ${LOG_FILE}
