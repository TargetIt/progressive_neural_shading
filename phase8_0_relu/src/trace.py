# Phase 8.0: Trace / Debug Utilities
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


def relu_analysis(tensor):
    """Analyze ReLU output: dead neuron fraction, activation stats."""
    arr = tensor.to_numpy()
    return {"dead_pct": float((arr == 0).mean()) * 100,
            "active_mean": float(arr[arr > 0].mean()) if (arr > 0).any() else 0}

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
    module = spy.Module.load_from_file(device, "step_8_0_relu.slang")
    import numpy as np
    weights_data = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
    weights = spy.Tensor.from_numpy(device, weights_data)
    biases_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    biases = spy.Tensor.from_numpy(device, biases_data)
    resolution = spy.int2(64, 64)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.render(pixel=spy.call_id(), resolution=resolution, weights=weights, biases=biases, _result=output)
    stats = relu_analysis(output)
    print(f"relu_analysis: dead={stats['dead_pct']:.1f}%, active_mean={stats['active_mean']:.3f}")
    print("Trace complete.")
