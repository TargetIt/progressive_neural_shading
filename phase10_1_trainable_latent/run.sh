#!/bin/bash
# phase10_1_trainable_latent
#   ./run.sh        — 运行主程序
#   ./run.sh test   — 运行 pytest
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    python src/step_10_1_trainable_latent.py
fi
