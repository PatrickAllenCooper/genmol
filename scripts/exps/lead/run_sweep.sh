#!/bin/bash
# scripts/exps/lead/run_sweep.sh
ORACLES=(parp1)
START_IDX=0
SIM_THR=0.4
SEEDS=(0 1 2 3 4)

for oracle in "${ORACLES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for strategy in random bandit; do
      echo "=== oracle=$oracle strategy=$strategy seed=$seed ==="
      python scripts/exps/lead/run.py \
        -o $oracle -i $START_IDX -d $SIM_THR -s $seed \
        --strategy $strategy
    done
  done
done