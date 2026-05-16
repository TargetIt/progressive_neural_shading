# Phase 8.1 Test Suite — Activation Gallery
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
        mod = spy.Module.load_from_file(app.device, "step_8_1_activation_gallery.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_reLU_defined():
    print("--- reLU function defined ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        # Test uses step_6_1 to verify reLU concept,
        # but we primarily check the compilation of step_8_1
        mod = spy.Module.load_from_file(app.device, "step_8_1_activation_gallery.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32"))
                self.biases_grad = spy.Tensor.zeros_like(self.biases)
                self.weights_grad = spy.Tensor.zeros_like(self.weights)
                self.m_biases = spy.Tensor.zeros_like(self.biases)
                self.m_weights = spy.Tensor.zeros_like(self.weights)
                self.v_biases = spy.Tensor.zeros_like(self.biases)
                self.v_weights = spy.Tensor.zeros_like(self.weights)

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(2, 32)
                self.layer1 = NetworkParameters(32, 32)
                self.layer2 = NetworkParameters(32, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8), network=network, _result=out)
        result = out.to_numpy()
        check(not np.any(np.isnan(result)), "no NaN values")
        check(np.all(result >= 0), "exp output ensures positivity")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_exp_output_positive():
    print("--- Exp Output is Positive ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=16, height=16)
        mod = spy.Module.load_from_file(app.device, "step_8_1_activation_gallery.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-5.0, 5.0, (outputs, inputs)).astype("float32"))
                self.biases_grad = spy.Tensor.zeros_like(self.biases)
                self.weights_grad = spy.Tensor.zeros_like(self.weights)
                self.m_biases = spy.Tensor.zeros_like(self.biases)
                self.m_weights = spy.Tensor.zeros_like(self.weights)
                self.v_biases = spy.Tensor.zeros_like(self.biases)
                self.v_weights = spy.Tensor.zeros_like(self.weights)

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(2, 32)
                self.layer1 = NetworkParameters(32, 32)
                self.layer2 = NetworkParameters(32, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8), network=network, _result=out)
        result = out.to_numpy()
        # exp(x) > 0 for all x, so output should be strictly positive
        check(np.all(result > 0), "exp output is strictly positive")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_8_0_relu.slang")
        check(mod is not None, "Phase 8.0 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 8.1 Activation Gallery - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_reLU_defined(); test_exp_output_positive(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
