#!/usr/bin/env bash
# Runs every evaluation stage after ./run_training.sh has finished, on an otherwise idle machine
# (OR-Tools uses 4 single-threaded worker processes on the 4 cores; neural inference uses 4 torch
# threads in the main process while no solver is running). Logs go to results/logs/.
# Every stage is cached / idempotent, so the script can simply be re-run after an interruption.
#
# Prerequisites: data.py download / preprocess / preprocess-fresh and zone_level.py build (README).
# Models come from checkpoints/ or, if absent, from the exported weights in results/models/.
#
# The `round1` stage is OPTIONAL: it only re-scores the archived round-1 validation caches in
# results/cache_round1 (git-ignored, cannot be regenerated) and is skipped when they are absent.
set -eu
cd "$(dirname "$0")"
mkdir -p results/logs
if pgrep -f "python train.py" > /dev/null; then echo "training is still running; wait for it to finish"; exit 1; fi
if [ -d results/cache_round1 ]; then
  python evaluate.py round1                       2>&1 | tee results/logs/round1.log
else
  echo "results/cache_round1 not present: skipping the optional round1 stage"
fi
python evaluate.py tune --parts softdist,zone,diffusion,zonehist,hier,hier_sup 2>&1 | tee results/logs/tune.log
# settings are now frozen; the final (fresh) test split is evaluated after this point only
python evaluate.py run --split fresh              2>&1 | tee results/logs/run_fresh.log
python evaluate.py run --split test               2>&1 | tee results/logs/run_test.log
python evaluate.py run --split heldout            2>&1 | tee results/logs/run_heldout.log
python evaluate.py budget --split fresh           2>&1 | tee results/logs/budget_fresh.log
python evaluate.py budget --split test            2>&1 | tee results/logs/budget_test.log
python diagnostics.py                             2>&1 | tee results/logs/diagnostics.log
python evaluate.py export                         2>&1 | tee results/logs/export.log
python evaluate.py report                         > results/logs/report.log 2>&1
python make_figures.py                            2>&1 | tee results/logs/figures.log
python make_report_docx.py                        2>&1 | tee results/logs/docx.log
python tests/test_basic.py                        2>&1 | tee results/logs/tests.log
echo "evaluation finished"
