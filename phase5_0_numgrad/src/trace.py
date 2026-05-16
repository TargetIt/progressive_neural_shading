# Phase 5.0: Trace / Debug Utilities
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


def gradient_stats(grad_tensor, h=0.01):
    """Return numerical gradient statistics.
    Returns: dict with mean_abs_grad, max_abs_grad, zero_fraction"""
    arr = grad_tensor.to_numpy()
    abs_grad = np.abs(arr)
    return {"mean_abs_grad": float(abs_grad.mean()),
            "max_abs_grad": float(abs_grad.max()),
            "h": h}

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
    module = spy.Module.load_from_file(device, "step_5_0_numgrad.slang")
    import numpy as np
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    h = 0.01
    ref = spy.Tensor.empty_like(albedo)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=ref)
    rp = spy.Tensor.from_numpy(device, np.clip(roughness.to_numpy()+h, 0, 1).astype(np.float32))
    rm = spy.Tensor.from_numpy(device, np.clip(roughness.to_numpy()-h, 0, 1).astype(np.float32))
    lp = spy.Tensor.empty_like(albedo); lm = spy.Tensor.empty_like(albedo)
    module.loss(pixel=spy.call_id(), reference=ref, material={"albedo": albedo, "normal": normal, "roughness": rp}, light_dir=light_dir, view_dir=view_dir, _result=lp)
    module.loss(pixel=spy.call_id(), reference=ref, material={"albedo": albedo, "normal": normal, "roughness": rm}, light_dir=light_dir, view_dir=view_dir, _result=lm)
    grad_np = (lp.to_numpy() - lm.to_numpy()) / (2.0 * h)
    grad_gray = np.mean(grad_np, axis=2)
    grad = spy.Tensor.from_numpy(device, np.stack([grad_gray]*3, axis=2).astype(np.float32))
    stats = gradient_stats(grad, h=h)
    print(f"gradient_stats: mean_abs={stats['mean_abs_grad']:.6f}, max_abs={stats['max_abs_grad']:.6f}")
    print("Trace complete.")
