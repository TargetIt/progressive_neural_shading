#!/bin/bash
# phase10_2_full_pipeline
#   ./run.sh        — 运行主程序
#   ./run.sh test   — 运行 pytest
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    python src/step_10_2_full_pipeline.py
fi
