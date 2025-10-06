#!/bin/bash

python preprocess_select.py \
  --train_json "NAME.json" \
  --output_json "NAME_selected.json" \
  --max_len 512 \
  --z_norm 1 \
  --budget_m 12 \
  --alpha 0.7 \
  --nn_eval_samples -1 \
  --seed 42