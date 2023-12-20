#!/usr/bin/env bash
CUR_DIR=$(pwd)
DATA_DIR=/group/dphi_algo_scratch_05/yuwang/dataset/imagenet/pytorch

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$0.$(date +%Y%m%d%H%M)"

RESNET18_CONFIG="--model resnet18 --pretrained /group/modelzoo/compression/pretrained/resnet18.pth"
RESNET50_CONFIG="--model resnet50 --pretrained /group/modelzoo/compression/pretrained/resnet50.pth"
MOBILENET_V2_CONFIG="--model mobilenet_v2 --pretrained /group/modelzoo/compression/pretrained/mobilenet_v2.pth"
#MODEL_CONFIG=${RESNET18_CONFIG}
#MODEL_CONFIG=${RESNET50_CONFIG}
MODEL_CONFIG=${MOBILENET_V2_CONFIG}

W_ELEM_FORAMT="int8"
W_ELEM_FORAMT="fp8_e4m3"
W_ELEM_FORAMT="fp8_e5m2"
#W_ELEM_FORAMT="fp6_e2m3"
#W_ELEM_FORAMT="fp6_e3m2"
#W_ELEM_FORAMT="fp4_e2m1"
A_ELEM_FORAMT=${W_ELEM_FORAMT}

MX_SPECS="--w_elem_format ${W_ELEM_FORAMT} --a_elem_format ${A_ELEM_FORAMT} --scale_bits 8 --block_size 32 --bfloat 16 --round even --custom_cuda"
CMD="PYTHONPATH=${CUR_DIR}/../ python -u cnn_mx.py --data_dir ${DATA_DIR} ${MODEL_CONFIG} ${MX_SPECS}"

echo "${CMD}" > ${LOG_FILE}
eval "${CMD}" 2>&1 | tee -a ${LOG_FILE}
