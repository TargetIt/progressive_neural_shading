# Phase 6.1: Trace / Debug Utilities
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


def relu_stats(tensor):
    """Verify ReLU output: no negative values, sparsity check."""
    arr = tensor.to_numpy()
    neg = int((arr < 0).sum())
    zeros = int((arr == 0).sum())
    return {"negatives": neg, "zeros": zeros, "total": arr.size}

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
    module = spy.Module.load_from_file(device, "step_6_1_neuron.slang")
    import numpy as np
    weights_data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    weights = spy.Tensor.from_numpy(device, weights_data)
    biases_data = np.array([-0.3, -0.3, 0.0], dtype=np.float32)
    biases = spy.Tensor.from_numpy(device, biases_data)
    resolution = spy.int2(64, 64)
    output = spy.Tensor.empty(device, shape=(64, 64), dtype=spy.float3)
    module.render(pixel=spy.call_id(), resolution=resolution, weights=weights, biases=biases, _result=output)
    stats = relu_stats(output)
    print(f"relu_stats: zeros={stats['zeros']}/{stats['total']}")
    print("Trace complete.")
