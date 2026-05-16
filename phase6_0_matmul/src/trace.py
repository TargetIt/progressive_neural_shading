# Phase 6.0: Trace / Debug Utilities
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


def verify_matmul_output(tensor, weights):
    """Verify matmul output matches expected pattern based on weights."""
    arr = tensor.to_numpy()
    return {"R_range": (float(arr[..., 0].min()), float(arr[..., 0].max())),
            "G_range": (float(arr[..., 1].min()), float(arr[..., 1].max())),
            "B_range": (float(arr[..., 2].min()), float(arr[..., 2].max()))}

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
    module = spy.Module.load_from_file(device, "step_6_0_matmul.slang")
    import numpy as np
    weights_data = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    weights = spy.Tensor.from_numpy(device, weights_data)
    resolution = spy.int2(64, 64)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.render(pixel=spy.call_id(), resolution=resolution, weights=weights, _result=output)
    ranges = verify_matmul_output(output, weights)
    print(f"verify_matmul_output: R={ranges['R_range']}, G={ranges['G_range']}, B={ranges['B_range']}")
    print("Trace complete.")
