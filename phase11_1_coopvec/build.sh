#!/bin/bash
# Phase 11.1: CoopVec Hardware Acceleration
set -e
cd "$(dirname "$0")"
cmake -S src -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _build
echo "Build successful. Run: _build/mlp-training-coopvec"
