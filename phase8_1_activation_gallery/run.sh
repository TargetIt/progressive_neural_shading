#!/bin/bash
# phase8_1_activation_gallery
#   ./run.sh        — 运行主程序
#   ./run.sh test   — 运行 pytest
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    python src/step_8_1_activation_gallery.py
fi
