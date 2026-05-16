#!/bin/bash
# phase5_3_train
#   ./run.sh        — 运行主程序
#   ./run.sh test   — 运行 pytest
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    python src/step_5_3_train.py
fi
