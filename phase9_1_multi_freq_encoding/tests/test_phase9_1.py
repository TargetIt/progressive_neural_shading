# Phase 9.1 Test Suite — Multi-Freq Encoding
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
        mod = spy.Module.load_from_file(app.device, "step_9_1_multi_freq_encoding.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_network_forward_pass():
    print("--- Network Forward Pass ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_9_1_multi_freq_encoding.slang")

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
                self.layer0 = NetworkParameters(16, 32)
                self.layer1 = NetworkParameters(32, 32)
                self.layer2 = NetworkParameters(32, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(16, 16), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(16, 16), network=network, _result=out)
        result = out.to_numpy()
        check(result.shape == (16, 16, 3), f"output shape: {result.shape}")
        check(np.all(result > 0), "exp output ensures positivity")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_encoding_structure():
    print("--- Encoding Structure ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=16, height=16)
        # Verify the encoded input count is correct by checking compilation
        # of a shader that references NetworkParameters<16, *>
        mod = spy.Module.load_from_file(app.device, "step_9_1_multi_freq_encoding.slang")

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
                self.layer0 = NetworkParameters(16, 32)
                self.layer1 = NetworkParameters(32, 32)
                self.layer2 = NetworkParameters(32, 3)

        network = Network()
        out = spy.Tensor.empty(app.device, shape=(4, 4), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(4, 4), network=network, _result=out)
        result = out.to_numpy()
        check(not np.any(np.isnan(result)), "no NaN values")
        check(not np.any(np.isinf(result)), "no Inf values")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_training_loop():
    print("--- Training Loop (short) ---")
    try:
        from app import App; import slangpy as spy
        from pathlib import Path
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_9_1_multi_freq_encoding.slang")

        data_path = Path(__file__).parent.parent / "src"
        image_path = data_path / "slangstars.png"
        if not image_path.exists():
            check(False, "slangstars.png not found")
            app.window.close()
            return

        image = spy.Tensor.load_from_image(app.device, image_path, linearize=True)

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

            def optimize(self, lr, step):
                mod.optimizer_step(self.biases, self.biases_grad, self.m_biases, self.v_biases, lr, step)
                mod.optimizer_step(self.weights, self.weights_grad, self.m_weights, self.v_weights, lr, step)

        class Network(spy.InstanceList):
            def __init__(self):
                super().__init__(mod["Network"])
                self.layer0 = NetworkParameters(16, 32)
                self.layer1 = NetworkParameters(32, 32)
                self.layer2 = NetworkParameters(32, 3)

            def optimize(self, lr, step):
                self.layer0.optimize(lr, step)
                self.layer1.optimize(lr, step)
                self.layer2.optimize(lr, step)

        network = Network()
        batch_size = (16, 16)

        loss_before = spy.Tensor.empty_like(image)
        mod.loss(pixel=spy.call_id(), resolution=spy.int2(16, 16), network=network,
                 reference=image, _result=loss_before)
        loss_before_val = float(np.mean(loss_before.to_numpy()))

        for i in range(5):
            mod.calculate_grads(
                seed=spy.wang_hash(seed=i, warmup=2),
                batch_index=spy.grid(batch_size),
                batch_size=spy.int2(batch_size),
                reference=image,
                network=network,
            )
            network.optimize(0.01, i + 1)

        loss_after = spy.Tensor.empty_like(image)
        mod.loss(pixel=spy.call_id(), resolution=spy.int2(16, 16), network=network,
                 reference=image, _result=loss_after)
        loss_after_val = float(np.mean(loss_after.to_numpy()))

        check(loss_after_val < loss_before_val,
              f"loss decreased: {loss_before_val:.6f} -> {loss_after_val:.6f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 9.1 Multi-Freq Encoding - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_network_forward_pass(); test_encoding_structure(); test_training_loop()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
