#!/usr/bin/env bash
# Runs every evaluation stage after ./run_training.sh has finished, on an otherwise idle machine
# (OR-Tools uses 4 single-threaded worker processes on the 4 cores; neural inference uses 4 torch
# threads in the main process while no solver is running). Logs go to results/logs/.
# Every stage is cached / idempotent, so the script can simply be re-run after an interruption.
set -eu
cd "$(dirname "$0")"
mkdir -p results/logs
if pgrep -f "python train.py" > /dev/null; then echo "training is still running; wait for it to finish"; exit 1; fi
python evaluate.py round1                         2>&1 | tee results/logs/round1.log
python evaluate.py tune                           2>&1 | tee results/logs/tune.log
python evaluate.py run --split test               2>&1 | tee results/logs/run_test.log
python evaluate.py run --split heldout            2>&1 | tee results/logs/run_heldout.log
python evaluate.py budget --split test            2>&1 | tee results/logs/budget_test.log
python diagnostics.py                             2>&1 | tee results/logs/diagnostics.log
python evaluate.py report                         > results/logs/report.log 2>&1
python evaluate.py export                         2>&1 | tee results/logs/export.log
python make_figures.py                            2>&1 | tee results/logs/figures.log
python make_report_docx.py                        2>&1 | tee results/logs/docx.log
python tests/test_basic.py                        2>&1 | tee results/logs/tests.log
echo "evaluation finished"
