# Phase 9.0: Trace / Debug Utilities
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


def encoding_stats(tensor):
    """Analyze sin-encoded output: frequency spectrum, range."""
    arr = tensor.to_numpy()
    return {"range": (float(arr.min()), float(arr.max())),
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
    module = spy.Module.load_from_file(device, "step_9_0_sin_encoding.slang")
    import numpy as np
    weights_data = np.random.uniform(-1, 1, (3, 4)).astype(np.float32)
    weights = spy.Tensor.from_numpy(device, weights_data)
    resolution = spy.int2(64, 64)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.render(pixel=spy.call_id(), resolution=resolution, weights=weights, _result=output)
    stats = encoding_stats(output)
    print(f"encoding_stats: range={stats['range']}, mean_abs={stats['mean_abs']:.3f}")
    print("Trace complete.")
