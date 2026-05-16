#!/bin/bash
# Phase 11.0: Vulkan MLP Training
set -e
cd "$(dirname "$0")"
cmake -S src -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _build
echo "Build successful. Run: _build/mlp-training"
