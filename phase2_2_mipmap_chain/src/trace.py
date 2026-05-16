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
    module = spy.Module.load_from_file(device, "step_2_2_mipmap.slang")
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    output = spy.Tensor.empty_like(albedo)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=output)
    l1 = spy.Tensor.empty(device, shape=(output.shape[0]//2, output.shape[1]//2), dtype=output.dtype)
    module.downsample3(spy.call_id(), output, _result=l1)
    l2 = spy.Tensor.empty(device, shape=(l1.shape[0]//2, l1.shape[1]//2), dtype=l1.dtype)
    module.downsample3(spy.call_id(), l1, _result=l2)
    ok, msg = verify_mipmap_chain([output, l1, l2])
    print(f"verify_mipmap_chain: {msg}")
    print("Trace complete.")
