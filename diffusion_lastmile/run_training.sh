#!/usr/bin/env bash
# Trains every model used in results/:
#   * 12 main runs: 2 levels (stop, zone) x 2 objectives (diffusion and the one-shot supervised
#     ablation) x 3 seeds, each with the per-level defaults in train.py (stop: 7,000 steps; zone:
#     2,500 steps). Diffusion and supervised runs of the same level use the identical network,
#     data, batch size, schedule and number of steps.
#   * 1 spec-budget run: stop-level diffusion, seed 0, 3,000 steps (train.py --tag budget), which
#     together with the zone diffusion seed-0 model fits the specification's 45-90 CPU-minute budget.
# The 12 main runs used about 1,000 CPU-minutes in total, far above that budget; README.md
# ("Training compute") gives the per-run numbers and the reason.
#
# Each run is executed in resumable chunks of at most ~9 minutes (the sandbox kills commands
# after 10) until its log says "training finished". Runs are single-threaded and PARALLEL runs
# execute at a time (default 6 on the 4-core machine used here; this measured about 1.8x the
# throughput of one 4-thread run). Zone runs go first because they are short.
#
#   ./run_training.sh                                              all 13 runs
#   JOBS="stop:diffusion:0 zone:supervised:1" ./run_training.sh    a subset (level:objective:seed)
#   JOBS="stop:diffusion:0:budget:3000" ./run_training.sh          level:objective:seed:tag:steps
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
DEFAULT_JOBS="$DEFAULT_JOBS stop:diffusion:0:budget:3000"
JOBS=${JOBS:-$DEFAULT_JOBS}

run_one() {
  IFS=: read -r lvl obj seed tag steps <<< "$1"
  extra=""
  name="${lvl}_${obj}_seed${seed}"
  if [ -n "${tag:-}" ]; then extra="--tag $tag"; name="${lvl}_${obj}_${tag}_seed${seed}"; fi
  if [ -n "${steps:-}" ]; then extra="$extra --total-steps $steps"; fi
  log="checkpoints/${name}_stdout.log"
  for i in $(seq 1 80); do
    if [ -f "$log" ] && grep -q "training finished" "$log"; then break; fi
    # --total-steps is only honoured when a run starts; on resume the checkpoint's value is kept
    # (it is identical here, so no schedule change happens)
    timeout 590 python train.py --level "$lvl" --objective "$obj" --seed "$seed" $extra --threads 1 --max-minutes 8.5 >> "$log" 2>&1
  done
  echo "$(date '+%H:%M:%S') done $name: $(grep 'training finished' "$log" | tail -1)"
}
export -f run_one
printf '%s\n' $JOBS | xargs -P "$PARALLEL" -I{} bash -c 'run_one {}'
