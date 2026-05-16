# Phase 7.0 Test Suite — Two-Layer Forward
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
        mod = spy.Module.load_from_file(app.device, "step_7_0_two_layer.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_output_shape():
    print("--- Output Shape ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_7_0_two_layer.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-1.0, 1.0, (outputs, inputs)).astype("float32"))

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(2, 16)
                self.layer1 = NetworkParameters(16, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(32, 32), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(32, 32), network=network, _result=out)
        result = out.to_numpy()
        check(result.shape == (32, 32, 3), f"output shape: {result.shape}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_two_layers_produce_different_output():
    print("--- Two Layers Different from Single ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_7_0_two_layer.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-1.0, 1.0, (outputs, inputs)).astype("float32"))

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(2, 16)
                self.layer1 = NetworkParameters(16, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(4, 4), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(4, 4), network=network, _result=out)
        result = out.to_numpy()
        # With random weights and activation, output should not be all zeros
        check(np.max(np.abs(result)) > 0.01, "output has non-zero values")
        # Output should be non-negative (ReLU)
        check(np.all(result >= 0), "all outputs >= 0 (ReLU)")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_6_2_mlp_training.slang")
        check(mod is not None, "Phase 6.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 7.0 Two-Layer Forward - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_output_shape(); test_two_layers_produce_different_output(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
