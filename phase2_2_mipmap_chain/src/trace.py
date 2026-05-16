# Phase 2.2: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和 mipmap 链验证。

import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    """返回 GPU Tensor 的统计信息。"""
    arr = tensor.to_numpy()
    return {
        "name": name,
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    """打印 Tensor 统计信息。"""
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} "
          f"mean={stats['mean']:.3f}")


def verify_mipmap_chain(levels):
    """验证 mipmap 链: 每级分辨率是上一级的 1/2。

    Args:
        levels: list of Tensor, mipmap levels from 0 to N

    Returns:
        (bool, str)
    """
    for i in range(len(levels) - 1):
        h0, w0 = levels[i].shape[:2]
        h1, w1 = levels[i+1].shape[:2]
        if h1 != h0 // 2 or w1 != w0 // 2:
            return False, f"Level {i}→{i+1}: ({h0},{w0})→({h1},{w1}) wrong"
    return True, f"{len(levels)} levels OK"
