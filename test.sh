#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_Test_$1_$now"
CUDA_VISIBLE_DEVICES=$3 python3 -u main.py --archs $1 --config $2 --test --ckpts $4 2>&1|tee logs/$log_name.log

# bash test.sh RGBI3D /experiments/aqa/RGBI3D/V1/cls/e1/config.py 0 experiments/aqa/RGBI3D/V1/cls/e1/last.pth
