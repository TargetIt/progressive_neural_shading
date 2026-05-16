#!/bin/bash
# phase9_1_multi_freq_encoding
#   ./run.sh        — 运行主程序
#   ./run.sh test   — 运行 pytest
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    python src/step_9_1_multi_freq_encoding.py
fi
