#!/usr/bin/env bash
set -e

nohup python train.py > log/wide_deep.log 2>&1 &
nohup python train.py --model_type wide > log/wide.log 2>&1 &
nohup python train.py --model_type deep > log/deep.log 2>&1 &
