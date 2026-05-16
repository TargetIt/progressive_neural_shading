# Phase 11.0: Trace / Verification Utilities
# C++/Vulkan project — Python trace is minimal (tests handle verification)
# This file provides project structure validation for consistency.

import os


def verify_project(base_dir):
    """Verify C++/Vulkan project structure."""
    required = [
        "src/CMakeLists.txt",
        "src/mlp-training/mlp-training.cpp",
        "src/mlp-training/network.slang",
        "src/mlp-training/kernels.slang",
        "src/mlp-training/adam.slang",
        "src/mlp-training/common.slang",
    ]
    missing = [f for f in required if not os.path.isfile(os.path.join(base_dir, f))]
    return len(missing) == 0, missing
