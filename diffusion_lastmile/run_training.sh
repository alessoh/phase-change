#!/usr/bin/env bash
# Trains every model used in results/: 2 levels (stop, zone) x 2 objectives (diffusion and the
# one-shot supervised ablation) x 3 seeds = 12 runs, each with the per-level defaults in train.py
# (stop: 7,000 steps; zone: 2,500 steps). Diffusion and supervised runs of the same level use the
# identical network, data, batch size, schedule and number of steps.
#
# Each run is executed in resumable chunks of at most ~9 minutes (the sandbox kills commands
# after 10) until its log says "training finished". Runs are single-threaded and PARALLEL runs
# execute at a time (default 6 on the 4-core machine used here; this measured about 1.8x the
# throughput of one 4-thread run). Zone runs go first because they are short.
#
#   ./run_training.sh                          all 12 runs
#   JOBS="stop:diffusion:0 zone:supervised:1" ./run_training.sh     a subset
set -u
cd "$(dirname "$0")"
mkdir -p checkpoints
PARALLEL=${PARALLEL:-6}
DEFAULT_JOBS=""
for lvl in zone stop; do
  for obj in diffusion supervised; do
    for seed in 0 1 2; do DEFAULT_JOBS="$DEFAULT_JOBS $lvl:$obj:$seed"; done
  done
done
JOBS=${JOBS:-$DEFAULT_JOBS}

run_one() {
  IFS=: read -r lvl obj seed <<< "$1"
  log="checkpoints/${lvl}_${obj}_seed${seed}_stdout.log"
  for i in $(seq 1 80); do
    if [ -f "$log" ] && grep -q "training finished" "$log"; then break; fi
    timeout 590 python train.py --level "$lvl" --objective "$obj" --seed "$seed" --threads 1 --max-minutes 8.5 >> "$log" 2>&1
  done
  echo "$(date '+%H:%M:%S') done $lvl $obj seed $seed: $(grep 'training finished' "$log" | tail -1)"
}
export -f run_one
printf '%s\n' $JOBS | xargs -P "$PARALLEL" -I{} bash -c 'run_one {}'
