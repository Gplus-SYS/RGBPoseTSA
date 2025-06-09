#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_Train_$1_$now"
CUDA_VISIBLE_DEVICES=$3 python3 -u main.py --archs $1 --config $2 $4 --timestamp $now 2>&1|tee logs/$log_name.log

# bash train.sh RGBPoseI3D_early_fusion configs/FineDiving_RGBPoseI3D_early_fusion.py 3 [--resume]
