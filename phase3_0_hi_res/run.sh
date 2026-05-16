#!/bin/bash
# phase3_0_hi_res
#   ./run.sh        — 运行主程序
#   ./run.sh trace  — 一键 trace 诊断
cd "$(dirname "$0")"
if [ "$1" = "trace" ]; then
    python src/trace.py
else
    python src/step_3_0_hi_res.py
fi
