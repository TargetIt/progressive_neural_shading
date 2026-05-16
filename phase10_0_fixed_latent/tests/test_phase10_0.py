# Phase 10.0 Test Suite — Fixed Latent
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
        mod = spy.Module.load_from_file(app.device, "step_10_0_fixed_latent.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_decode_output():
    print("--- Decode Output ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_10_0_fixed_latent.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-1.0, 1.0, (outputs, inputs)).astype("float32"))

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(4, 16)
                self.layer1 = NetworkParameters(16, 3)
                latent_data = np.random.uniform(0.0, 1.0, (4, 4, 4)).astype(np.float32)
                self.latent_texture = spy.Tensor.from_numpy(app.device, latent_data)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(16, 16), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(16, 16), network=network, _result=out)
        result = out.to_numpy()
        check(result.shape == (16, 16, 3), f"output shape: {result.shape}")
        check(not np.any(np.isnan(result)), "no NaN values")
        check(not np.any(np.isinf(result)), "no Inf values")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_latent_interpolation():
    print("--- Latent Interpolation ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_10_0_fixed_latent.slang")

        class NetworkParameters(spy.InstanceList):
            def __init__(self, inputs, outputs):
                super().__init__(mod[f"NetworkParameters<{inputs},{outputs}>"])
                self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
                self.weights = spy.Tensor.from_numpy(app.device,
                    np.random.uniform(-1.0, 1.0, (outputs, inputs)).astype("float32"))

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(4, 16)
                self.layer1 = NetworkParameters(16, 3)
                # All same latent -> should get similar outputs
                latent_data = np.ones((4, 4, 4), dtype=np.float32) * 0.5
                self.latent_texture = spy.Tensor.from_numpy(app.device, latent_data)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8), network=network, _result=out)
        result = out.to_numpy()
        # With uniform latent, output should be uniform (or very close)
        std = np.std(result)
        check(std < 0.1, f"uniform latent -> low variance output: std={std:.4f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_9_1_multi_freq_encoding.slang")
        check(mod is not None, "Phase 9.1 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 10.0 Fixed Latent - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_decode_output(); test_latent_interpolation(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
