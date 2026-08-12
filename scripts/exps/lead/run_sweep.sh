#!/bin/bash
# scripts/exps/lead/run_lam_sweep.sh
if pgrep -f "scripts/exps/lead/run.py" > /dev/null; then
  echo "ERROR: run.py already running. Aborting."; exit 1
fi

ORACLE=parp1
START_IDX=0
SIM_THR=0.4
UCB_C=2.0
POP_CAP=150
LAMS=(0 1 3 6 10)
SEEDS=(0 1 2)

for lam in "${LAMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== lam_rq=$lam seed=$seed ==="
    python -u scripts/exps/lead/run.py \
      -o $ORACLE -i $START_IDX -d $SIM_THR -s $seed \
      --strategy bandit --ucb_c $UCB_C --pop_cap $POP_CAP \
      --lam_rq $lam --lam_rs 1.0 --lam_rsim 1.0
  done
done