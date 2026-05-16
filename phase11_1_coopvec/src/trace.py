# Phase 11.1: Trace / Verification Utilities
# CoopVec C++/Vulkan project — minimal Python trace for consistency.

import os


def verify_project(base_dir):
    """Verify CoopVec project structure."""
    required = [
        "src/CMakeLists.txt",
        "src/mlp-training-coopvec/mlp-training-coopvec.cpp",
        "src/mlp-training-coopvec/mlp.slang",
        "src/mlp-training-coopvec/mlvec.slang",
        "src/mlp-training-coopvec/network.slang",
        "src/mlp-training-coopvec/kernels.slang",
        "src/mlp-training-coopvec/adam.slang",
        "src/mlp-training-coopvec/common.slang",
    ]
    missing = [f for f in required if not os.path.isfile(os.path.join(base_dir, f))]
    return len(missing) == 0, missing
