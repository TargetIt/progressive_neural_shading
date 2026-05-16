#!/bin/bash
# phase11_0_vulkan_mlp
#   ./build.sh        — CMake configure + build
#   ./build.sh trace  — pytest (project structure verification)
set -e
cd "$(dirname "$0")"
if [ "$1" = "trace" ] || [ "$1" = "test" ]; then
    python -m pytest tests/ -v
else
    cmake -S src -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build _build
    echo "Build successful."
fi
