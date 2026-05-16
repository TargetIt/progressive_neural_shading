# Phase 5.1 Test Suite — Automatic Differentiation
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
        mod = spy.Module.load_from_file(app.device, "step_5_1_ad.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_autodiff_gradient():
    print("--- Autodiff Gradient ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_1_ad.slang")

        # Uniform material
        albedo = spy.Tensor.from_numpy(app.device, np.ones((8, 8, 3), dtype=np.float32) * 0.5)
        normal = spy.Tensor.from_numpy(app.device, np.zeros((8, 8, 3), dtype=np.float32))
        roughness = spy.Tensor.from_numpy(app.device, np.ones((8, 8), dtype=np.float32) * 0.5)

        # Trainable copies (slightly different)
        train_albedo = spy.Tensor.from_numpy(app.device, np.ones((8, 8, 3), dtype=np.float32) * 0.6)
        train_normal = spy.Tensor.from_numpy(app.device, np.zeros((8, 8, 3), dtype=np.float32))
        train_roughness = spy.Tensor.from_numpy(app.device, np.ones((8, 8), dtype=np.float32) * 0.7)

        # Gradients
        ag = spy.Tensor.zeros_like(albedo)
        ng = spy.Tensor.zeros_like(normal)
        rg = spy.Tensor.zeros_like(roughness)

        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        view_dir = spy.float3(0, 0, 1)

        # Render reference
        ref = spy.Tensor.empty_like(albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": albedo, "normal": normal, "roughness": roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=ref)

        # Compute autodiff gradients
        mod.calculate_grad(pixel=spy.call_id(), reference=ref,
                           material={"albedo": train_albedo, "normal": train_normal,
                                     "roughness": train_roughness,
                                     "albedo_grad": ag, "normal_grad": ng,
                                     "roughness_grad": rg},
                           light_dir=light_dir, view_dir=view_dir)

        ag_np = ag.to_numpy(); ng_np = ng.to_numpy(); rg_np = rg.to_numpy()

        # Verify gradient tensors were written (non-zero, since train differs from ref)
        check(float(np.abs(ag_np).mean()) > 0, f"albedo grad: {float(np.abs(ag_np).mean()):.6f}")
        check(float(np.abs(rg_np).mean()) > 0, f"roughness grad: {float(np.abs(rg_np).mean()):.6f}")

        # Albedo grad should be non-zero since train_albedo != albedo
        # The gradient should point in the direction to reduce the loss
        check(float(np.abs(ag_np).max()) > 0, f"albedo grad max: {float(np.abs(ag_np).max()):.6f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_0_numgrad.slang")
        check(mod is not None, "Phase 5.0 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 5.1 Automatic Differentiation - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_autodiff_gradient(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
