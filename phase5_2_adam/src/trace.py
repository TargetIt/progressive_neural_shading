# Phase 5.2: Trace / Debug Utilities
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


def adam_stats(step, loss, lr):
    """Return Adam optimization progress."""
    return {"step": step, "loss": float(loss), "lr": lr}

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
    module = spy.Module.load_from_file(device, "step_5_2_adam.slang")
    import numpy as np
    assets = src_dir.parent / "assets"
    albedo_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
    normal_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
    roughness_map = spy.Tensor.load_from_image(device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)
    def ds(t, n):
        for _ in range(n):
            d = spy.Tensor.empty(device, shape=(t.shape[0]//2, t.shape[1]//2), dtype=t.dtype)
            if d.dtype.name == "vector": module.downsample3(spy.call_id(), t, _result=d)
            else: module.downsample1(spy.call_id(), t, _result=d)
            t = d
        return t
    la = ds(albedo_map, 2); ln = ds(normal_map, 2); lr = ds(roughness_map, 2)
    train_a = spy.Tensor.from_numpy(device, np.ones(la.shape, dtype=np.float32)*0.5)
    train_n = spy.Tensor.from_numpy(device, np.tile(np.array([0,0,1], dtype=np.float32), (la.shape[0], la.shape[1], 1)))
    train_r = spy.Tensor.from_numpy(device, np.ones(lr.shape, dtype=np.float32)*0.5)
    ag = spy.Tensor.zeros_like(la); ng = spy.Tensor.zeros_like(ln); rg = spy.Tensor.zeros_like(lr)
    ma = spy.Tensor.zeros_like(train_a); va = spy.Tensor.zeros_like(train_a)
    mn = spy.Tensor.zeros_like(train_n); vn = spy.Tensor.zeros_like(train_n)
    mr = spy.Tensor.zeros_like(train_r); vr = spy.Tensor.zeros_like(train_r)
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)
    full = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(), material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map}, light_dir=light_dir, view_dir=view_dir, _result=full)
    ref = ds(full, 2)
    for _ in range(3):
        ag.zero_(); ng.zero_(); rg.zero_()
        module.calculate_grad(pixel=spy.call_id(), reference=ref, material={"albedo": train_a, "normal": train_n, "roughness": train_r, "albedo_grad": ag, "normal_grad": ng, "roughness_grad": rg}, light_dir=light_dir, view_dir=view_dir)
        module.optimizer_step3(train_a, ag, ma, va, 0.002, 1, False)
        module.optimizer_step3(train_n, ng, mn, vn, 0.002, 1, True)
        module.optimizer_step1(train_r, rg, mr, vr, 0.002, 1)
    pred = spy.Tensor.empty_like(ref)
    module.render(pixel=spy.call_id(), material={"albedo": train_a, "normal": train_n, "roughness": train_r}, light_dir=light_dir, view_dir=view_dir, _result=pred)
    loss = spy.Tensor.empty_like(ref)
    module.loss(pixel=spy.call_id(), reference=ref, material={"albedo": train_a, "normal": train_n, "roughness": train_r}, light_dir=light_dir, view_dir=view_dir, _result=loss)
    loss_val = float(np.mean(loss.to_numpy()))
    stats = adam_stats(3, loss_val, 0.002)
    print(f"adam_stats: step={stats['step']}, loss={stats['loss']:.6f}")
    print("Trace complete.")
