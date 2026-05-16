# Phase 2.1: Trace / Debug Utilities
# ====================================
# 提供 shader 输出的基本统计和 GPU 降采样验证。

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


def verify_gpu_downsample(original, downsampled, steps=2):
    """验证 GPU 降采样: 分辨率正确减半, 无 NaN。

    Returns:
        (bool, str)
    """
    orig_h, orig_w = original.shape[:2]
    expected_h, expected_w = orig_h // (2 ** steps), orig_w // (2 ** steps)
    ds_h, ds_w = downsampled.shape[:2]
    if ds_h != expected_h or ds_w != expected_w:
        return False, f"expected ({expected_h},{expected_w}), got ({ds_h},{ds_w})"
    arr = downsampled.to_numpy()
    if np.any(np.isnan(arr)):
        return False, "NaN in downsampled output"
    return True, f"shape=({ds_h},{ds_w}) OK"

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
    module = spy.Module.load_from_file(device, "step_2_1_gpu_downsample.slang")
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    output = spy.Tensor.empty_like(albedo)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=output)
    ds = spy.Tensor.empty(device, shape=(output.shape[0]//2, output.shape[1]//2), dtype=output.dtype)
    module.downsample3(spy.call_id(), output, _result=ds)
    ok, msg = verify_gpu_downsample(output, ds, steps=1)
    print(f"verify_gpu_downsample: {msg}")
    print("Trace complete.")
