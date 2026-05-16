# Phase 8.0 Test Suite — ReLU Activation
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
        mod = spy.Module.load_from_file(app.device, "step_8_0_relu.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_relu_clips_negatives():
    print("--- ReLU Clips Negatives (no_act vs relu) ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_8_0_relu.slang")

        weights = spy.Tensor.from_numpy(app.device, np.array([
            [-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0],
        ], dtype=np.float32))
        biases = spy.Tensor.from_numpy(app.device, np.array([-0.5, -0.5, 0.0], dtype=np.float32))

        out_no_act = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        out_relu = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)

        mod.render_no_act(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                          weights=weights, biases=biases, _result=out_no_act)
        mod.render_with_relu(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                             weights=weights, biases=biases, _result=out_relu)

        no_act = out_no_act.to_numpy()
        relu = out_relu.to_numpy()

        # Without activation, outputs can be negative
        check(np.any(no_act < 0), "no_act has negative values")
        # With ReLU, all outputs should be >= 0
        check(np.all(relu >= 0), "ReLU output has no negative values")
        # ReLU output should be max(0, no_act)
        check(np.allclose(relu, np.maximum(no_act, 0), atol=1e-5),
              "ReLU = max(0, no_act)")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_relu_identity_for_positive():
    print("--- ReLU Identity for Positive ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_8_0_relu.slang")

        # Positive weights + bias -> outputs are positive
        weights = spy.Tensor.from_numpy(app.device, np.array([
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
        ], dtype=np.float32))
        biases = spy.Tensor.from_numpy(app.device, np.array([0.0, 0.0, 0.0], dtype=np.float32))

        out_no_act = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        out_relu = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)

        mod.render_no_act(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                          weights=weights, biases=biases, _result=out_no_act)
        mod.render_with_relu(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                             weights=weights, biases=biases, _result=out_relu)

        no_act = out_no_act.to_numpy()
        relu = out_relu.to_numpy()
        # When all inputs are positive, ReLU should be identity
        check(np.allclose(relu, no_act, atol=1e-5),
              "ReLU = identity for positive inputs")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_7_1_deep_network.slang")
        check(mod is not None, "Phase 7.1 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 8.0 ReLU Activation - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_relu_clips_negatives(); test_relu_identity_for_positive(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
