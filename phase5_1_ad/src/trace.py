# Phase 5.1: Trace / Debug Utilities
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


def ad_gradient_stats(albedo_grad, normal_grad, roughness_grad):
    """Summarize AD-computed gradients.
    Returns: dict with |grad| per parameter"""
    ag = albedo_grad.to_numpy(); ng = normal_grad.to_numpy(); rg = roughness_grad.to_numpy()
    return {"|grad_albedo|": float(np.abs(ag).mean()),
            "|grad_normal|": float(np.abs(ng).mean()),
            "|grad_roughness|": float(np.abs(rg).mean())}

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
    module = spy.Module.load_from_file(device, "step_5_1_ad.slang")
    import numpy as np
    assets = src_dir.parent / "assets"
    albedo_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    train_albedo = spy.Tensor.from_numpy(device, albedo_map.to_numpy().copy())
    train_normal = spy.Tensor.from_numpy(device, np.tile(np.array([0,0,1], dtype=np.float32), (albedo_map.shape[0], albedo_map.shape[1], 1)))
    train_roughness = spy.Tensor.from_numpy(device, np.ones(roughness_map.shape, dtype=np.float32)*0.5)
    albedo_grad = spy.Tensor.zeros_like(albedo_map)
    normal_grad = spy.Tensor.zeros_like(normal_map)
    roughness_grad = spy.Tensor.zeros_like(roughness_map)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    ref = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(), material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map}, light_dir=light_dir, view_dir=view_dir, _result=ref)
    albedo_grad.zero_(); normal_grad.zero_(); roughness_grad.zero_()
    module.calculate_grad(pixel=spy.call_id(), reference=ref, material={"albedo": train_albedo, "normal": train_normal, "roughness": train_roughness, "albedo_grad": albedo_grad, "normal_grad": normal_grad, "roughness_grad": roughness_grad}, light_dir=light_dir, view_dir=view_dir)
    stats = ad_gradient_stats(albedo_grad, normal_grad, roughness_grad)
    print(f"ad_gradient_stats: |grad_albedo|={stats['|grad_albedo|']:.6f}, |grad_roughness|={stats['|grad_roughness|']:.6f}")
    print("Trace complete.")
