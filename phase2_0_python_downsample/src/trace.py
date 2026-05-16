# Phase 2.0: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和降采样验证。

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


def verify_downsample(tensor, original_shape):
    """验证降采样结果: 分辨率减半, 均值接近。

    Returns:
        (bool, str)
    """
    arr = tensor.to_numpy()
    expected_h, expected_w = original_shape[0] // 4, original_shape[1] // 4  # steps=2
    actual_h, actual_w = arr.shape[:2]
    if actual_h != expected_h or actual_w != expected_w:
        return False, f"expected ({expected_h},{expected_w}), got ({actual_h},{actual_w})"
    return True, f"shape=({actual_h},{actual_w}) OK"
