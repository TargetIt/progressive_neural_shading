# Phase 11.1 Requirements

## New Features (2026-05-16)

- **Cooperative Vector 加速**: NVIDIA GPU 硬件指令加速 MLP 训练
- **CoopVec matmul**: `coopVecMatMul` intrinsic 替代软件 matmul
- **性能对比**: SW vs HW 加速的 speedup 测量

## Functional Requirements

1. CMake configure + build 成功 (需要 NVIDIA GPU + CoopVec 支持)
2. CoopVec matmul 正确计算 (结果与 SW 版本一致)
3. 训练速度显著快于 Phase 11.0

## Acceptance Criteria

- [x] CoopVec project builds successfully
- [x] Training produces correct results
- [x] Measurable speedup vs Phase 11.0
