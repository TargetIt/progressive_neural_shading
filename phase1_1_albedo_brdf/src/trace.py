# Phase 1.1: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计: min, max, mean
# 新增: BRDF 光照变化验证

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


def verify_lighting_variation(tensor: spy.Tensor, min_std: float = 0.001):
    """验证 BRDF 光照是否产生了像素间差异。

    Returns:
        (bool, str): 是否通过, 错误信息
    """
    arr = tensor.to_numpy()
    std = float(np.std(arr))
    if std > min_std:
        return True, f"std={std:.4f} (有光照变化)"
    else:
        return False, f"std={std:.4f} (无光照变化, 可能全屏单色)"

if __name__ == "__main__":
    """一键 trace: python src/trace.py"""
    import slangpy as spy
    from pathlib import Path


    src_dir = Path(__file__).parent
    device = spy.create_device(
        spy.DeviceType.automatic,
        enable_debug_layers=True,
        include_paths=[src_dir],
    )
    module = spy.Module.load_from_file(device, "step_1_1_albedo_brdf.slang")
    light_dir = spy.float3(0.3, 0.2, 1.0)
    view_dir = spy.float3(0.0, 0.0, 1.0)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.render(pixel=spy.call_id(), light_dir=light_dir, view_dir=view_dir, _result=output)
    ok, msg = verify_lighting_variation(output)
    print(f"verify_lighting_variation: {msg}")
    print("Trace complete.")
