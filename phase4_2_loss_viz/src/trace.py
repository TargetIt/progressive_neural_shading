# Phase 4.2: Trace / Debug Utilities
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


def loss_stats(loss_tensor):
    """Return loss statistics for the error heatmap.
    Returns: dict with mean_loss, max_loss, coverage (pct > 0.001)"""
    arr = loss_tensor.to_numpy()
    gray = (arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3.0
    return {"mean_loss": float(gray.mean()), "max_loss": float(gray.max()),
            "coverage_pct": float((gray > 0.001).mean()) * 100}

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
    module = spy.Module.load_from_file(device, "step_4_2_loss_viz.slang")
    assets = src_dir.parent / "assets"
    albedo = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    output = spy.Tensor.empty_like(albedo)
    module.render(pixel=spy.call_id(), material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=output)
    d1 = spy.Tensor.empty(device, shape=(output.shape[0]//2, output.shape[1]//2), dtype=output.dtype)
    module.downsample3(spy.call_id(), output, _result=d1)
    ref = spy.Tensor.empty(device, shape=(d1.shape[0]//2, d1.shape[1]//2), dtype=d1.dtype)
    module.downsample3(spy.call_id(), d1, _result=ref)
    loss = spy.Tensor.empty_like(ref)
    la = spy.Tensor.empty_like(ref); ln = spy.Tensor.empty_like(ref); lr = spy.Tensor.empty(device, shape=ref.shape[:2], dtype=spy.float1)
    for _ in range(2):
        na = spy.Tensor.empty(device, shape=(albedo.shape[0]//2, albedo.shape[1]//2), dtype=albedo.dtype)
        module.downsample3(spy.call_id(), albedo, _result=na); albedo = na
        nn = spy.Tensor.empty(device, shape=(normal.shape[0]//2, normal.shape[1]//2), dtype=normal.dtype)
        module.downsample3(spy.call_id(), normal, _result=nn); normal = nn
        nr = spy.Tensor.empty(device, shape=(roughness.shape[0]//2, roughness.shape[1]//2), dtype=roughness.dtype)
        module.downsample1(spy.call_id(), roughness, _result=nr); roughness = nr
    module.loss(pixel=spy.call_id(), reference=ref, material={"albedo": albedo, "normal": normal, "roughness": roughness}, light_dir=light_dir, view_dir=view_dir, _result=loss)
    stats = loss_stats(loss)
    print(f"loss_stats: mean_loss={stats['mean_loss']:.4f}, coverage={stats['coverage_pct']:.1f}%")
    print("Trace complete.")
