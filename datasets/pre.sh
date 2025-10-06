#!/bin/bash
# ./pre.sh <basename>

BASENAME=$1

python preprocess_ts.py \
  --ts "./${BASENAME}.ts" \
  --json "./${BASENAME}.json" \
  --save-label-map
