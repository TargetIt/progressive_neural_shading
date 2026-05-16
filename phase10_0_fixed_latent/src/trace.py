# Phase 10.0: Trace / Debug Utilities
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


def latent_stats(latent_tensor):
    """Analyze latent texture: resolution and channel stats."""
    arr = latent_tensor.to_numpy()
    return {"shape": latent_tensor.shape,
            "channels": latent_tensor.shape[-1] if len(latent_tensor.shape) > 2 else 1,
            "mean_abs": float(np.abs(arr).mean())}

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
    module = spy.Module.load_from_file(device, "step_10_0_fixed_latent.slang")
    import numpy as np
    latent = spy.Tensor.from_numpy(device, np.random.normal(0, 1, (16, 16, 8)).astype(np.float32))
    resolution = spy.int2(64, 64)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.decode(pixel=spy.call_id(), resolution=resolution, latent=latent, _result=output)
    stats = latent_stats(latent)
    print(f"latent_stats: shape={stats['shape']}, channels={stats['channels']}")
    print("Trace complete.")
