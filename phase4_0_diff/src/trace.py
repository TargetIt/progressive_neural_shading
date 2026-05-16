# Phase 4.0: Trace / Debug Utilities
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


def error_stats(diff_tensor):
    """Return per-pixel error statistics.
    Returns: dict with mean_error, max_error, pct_above(>0.01)"""
    arr = diff_tensor.to_numpy()
    gray = arr[..., 0]
    return {"mean_error": float(gray.mean()), "max_error": float(gray.max()),
            "pct_above_001": float((gray > 0.01).mean()) * 100}

if __name__ == "__main__":
    """一键 trace: python src/trace.py"""
    import slangpy as spy
    from pathlib import Path
    import numpy as np

    src_dir = Path(__file__).parent
    device = spy.create_device(
        spy.DeviceType.automatic,
        enable_debug_layers=True,
        include_paths=[src_dir],
    )
    module = spy.Module.load_from_file(device, "step_4_0_diff.slang")
    import numpy as np
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    ref = spy.Tensor.empty_like(albedo)
    mod_roughness = spy.Tensor.from_numpy(device, 0.5 * np.ones(roughness.shape, dtype=np.float32))
    diff = spy.Tensor.empty_like(albedo)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=ref)
    module.abs_diff(pixel=spy.call_id(), reference=ref, material={"albedo": albedo, "normal": normal, "roughness": mod_roughness}, light_dir=light_dir, view_dir=view_dir, _result=diff)
    stats = error_stats(diff)
    print(f"error_stats: mean_error={stats['mean_error']:.4f}, pct_above={stats['pct_above_001']:.1f}%")
    print("Trace complete.")
