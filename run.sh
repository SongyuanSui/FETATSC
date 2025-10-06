#!/bin/bash

TRAIN="TRAIN.json"
TEST="TEST.json"
OUTPUT="OUT.json"

python scripts/multi_agent.py \
  --train_json $TRAIN \
  --test_json $TEST \
  --output_file $OUTPUT \
  --workers 10 \
  --max_len 512 \
  --dtw_len 512 \
  --dtw_radius_frac 0.05 \
  --z_norm 1 \
  --neighbors_k 3 \
  --preselect_k 0 \
  --decimals 1 \
  --seed 42