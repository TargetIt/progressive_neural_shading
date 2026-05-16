# Phase 3.0: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和高分辨率对比验证。

import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    arr = tensor.to_numpy()
    return {
        "name": name, "shape": tensor.shape, "dtype": str(tensor.dtype),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} "
          f"mean={stats['mean']:.3f}")


def compare_1x_vs_2x(out_1x, out_2x_down):
    """对比 1x 和超采样后降采样的 2x 输出差异。

    Returns:
        (bool, str, float): pass/fail, message, mean_absolute_diff
    """
    arr1 = out_1x.to_numpy()
    arr2 = out_2x_down.to_numpy()
    diff = np.abs(arr1 - arr2).mean()
    if diff > 1.0:
        return False, f"large diff: {diff:.4f}", diff
    return True, f"diff={diff:.4f}", diff

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
    module = spy.Module.load_from_file(device, "step_3_0_hi_res.slang")
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    out_1x = spy.Tensor.empty_like(albedo)
    out_2x = spy.Tensor.empty(device, shape=(albedo.shape[0]*2, albedo.shape[1]*2), dtype=albedo.dtype)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=out_1x)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=out_2x)
    out_2x_down = spy.Tensor.empty_like(albedo)
    module.downsample3(spy.call_id(), out_2x, _result=out_2x_down)
    ok, msg, diff = compare_1x_vs_2x(out_1x, out_2x_down)
    print(f"compare_1x_vs_2x: {msg}")
    print("Trace complete.")
