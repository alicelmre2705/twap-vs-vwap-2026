#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python3}"
if [ -z "$("$PY" -c 'print(1)' 2>/dev/null)" ]; then
    echo "Warning: $PY is broken, falling back to /usr/bin/python3" >&2
    PY=/usr/bin/python3
fi

mkdir -p data/raw data/clean results/tables results/figures

"$PY" -m src.download_data
"$PY" -m src.clean_data
"$PY" -m src.run_experiment
"$PY" -m src.make_plots
"$PY" -m src.run_one_window AAPL

echo "Pipeline complete."
echo "Tables:  results/tables/"
echo "Figures: results/figures/"
