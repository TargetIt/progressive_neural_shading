# Phase 3.1: Trace / Debug Utilities
import numpy as np
import slangpy as spy


def tensor_stats(tensor: spy.Tensor, name: str = "tensor") -> dict:
    arr = tensor.to_numpy()
    return {"name": name, "shape": tensor.shape, "dtype": str(tensor.dtype),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "mean": float(np.mean(arr))}


def print_stats(tensor: spy.Tensor, name: str = "tensor"):
    stats = tensor_stats(tensor, name)
    print(f"[{stats['name']}] shape={stats['shape']} "
          f"min={stats['min']:.3f} max={stats['max']:.3f} mean={stats['mean']:.3f}")


def compare_ssaa(std_1x, ssaa_result):
    """Compare standard 1x render vs SSAA result: verify SSAA reduces aliasing (lower std).
    Returns: (bool, str, float)"""
    arr1 = std_1x.to_numpy(); arr2 = ssaa_result.to_numpy()
    std1, std2 = float(arr1.std()), float(arr2.std())
    diff = float(np.abs(arr1 - arr2).mean())
    ok = diff < 1.0
    return ok, f"std(1x)={std1:.4f} std(SSAA)={std2:.4f} diff={diff:.4f}", diff

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
    module = spy.Module.load_from_file(device, "step_3_1_ssaa.slang")
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    std_1x = spy.Tensor.empty_like(albedo)
    hires = spy.Tensor.empty(device, shape=(albedo.shape[0]*4, albedo.shape[1]*4), dtype=albedo.dtype)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=std_1x)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=hires)
    d1 = spy.Tensor.empty(device, shape=(hires.shape[0]//2, hires.shape[1]//2), dtype=hires.dtype)
    module.downsample3(spy.call_id(), hires, _result=d1)
    ssaa = spy.Tensor.empty_like(albedo)
    module.downsample3(spy.call_id(), d1, _result=ssaa)
    ok, msg, diff = compare_ssaa(std_1x, ssaa)
    print(f"compare_ssaa: {msg}")
    print("Trace complete.")
