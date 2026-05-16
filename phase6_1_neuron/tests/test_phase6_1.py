# Phase 6.1 Test Suite — Neuron Complete (Bias + ReLU)
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")

def test_shader_compile():
    print("--- Shader Compile ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_6_1_neuron.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_relu_clips_negatives():
    print("--- ReLU Clips Negatives ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_6_1_neuron.slang")

        # Weights: all zeros, Biases: [-1, -0.5, 0]
        # Output should be: [0, 0, 0] because all are <= 0
        weights = spy.Tensor.from_numpy(app.device, np.zeros((3, 2), dtype=np.float32))
        biases = spy.Tensor.from_numpy(app.device, np.array([-1.0, -0.5, 0.0], dtype=np.float32))

        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                   weights=weights, biases=biases, _result=out)
        result = out.to_numpy()

        # R channel: -1 -> 0, G channel: -0.5 -> 0, B channel: 0 -> 0
        check(np.all(result[..., 0] == 0.0), "R clipped to 0")
        check(np.all(result[..., 1] == 0.0), "G clipped to 0")
        check(np.all(result[..., 2] == 0.0), "B (bias=0) is 0")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_relu_passes_positives():
    print("--- ReLU Passes Positives ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_6_1_neuron.slang")

        # W = [[1,0],[0,1],[1,1]], b = [0.2, 0.3, 0]
        weights = spy.Tensor.from_numpy(app.device, np.array([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        ], dtype=np.float32))
        biases = spy.Tensor.from_numpy(app.device, np.array([0.2, 0.3, 0.0], dtype=np.float32))

        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.int2(6, 6), resolution=spy.int2(8, 8),
                   weights=weights, biases=biases, _result=out)
        result = out.to_numpy()
        # uv = (6.5/8, 6.5/8) = (0.8125, 0.8125)
        # R = max(0, 0.8125*1 + 0.2) = 1.0125
        # G = max(0, 0.8125*1 + 0.3) = 1.1125
        # B = max(0, 0.8125*1 + 0.8125*1 + 0) = 1.625
        check(result[6, 6, 0] > 0.8, f"R positive: {result[6,6,0]}")
        check(result[6, 6, 1] > 0.8, f"G positive: {result[6,6,1]}")
        check(result[6, 6, 2] > 1.5, f"B positive: {result[6,6,2]}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_bias_shift():
    print("--- Bias Shift ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_6_1_neuron.slang")

        # Same weights, different biases -> different outputs
        weights = spy.Tensor.from_numpy(app.device, np.zeros((3, 2), dtype=np.float32))

        # Bias = 0.5 -> all pixels should be 0.5
        biases = spy.Tensor.from_numpy(app.device, np.array([0.5, 0.5, 0.5], dtype=np.float32))
        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                   weights=weights, biases=biases, _result=out)
        result = out.to_numpy()
        check(np.allclose(result, 0.5, atol=1e-4), "uniform bias=0.5 produces uniform output")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 6.1 Neuron Complete - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_relu_clips_negatives(); test_relu_passes_positives(); test_bias_shift()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
