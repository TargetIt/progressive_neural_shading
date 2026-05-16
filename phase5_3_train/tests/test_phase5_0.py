# Phase 5.0 Test Suite — Numerical Gradient
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
        mod = spy.Module.load_from_file(app.device, "step_5_0_numgrad.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_numerical_gradient():
    print("--- Numerical Gradient ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_0_numgrad.slang")

        # Uniform inputs: constant roughness=0.5
        albedo = spy.Tensor.from_numpy(app.device, np.ones((16, 16, 3), dtype=np.float32) * 0.5)
        normal = spy.Tensor.from_numpy(app.device, np.zeros((16, 16, 3), dtype=np.float32))
        roughness = spy.Tensor.from_numpy(app.device, np.ones((16, 16), dtype=np.float32) * 0.5)
        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        view_dir = spy.float3(0, 0, 1)
        h = 0.01

        # Reference
        ref = spy.Tensor.empty_like(albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": albedo, "normal": normal, "roughness": roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=ref)

        # Loss at x+h
        rp = spy.Tensor.from_numpy(app.device, np.ones((16, 16), dtype=np.float32) * 0.51)
        loss_p = spy.Tensor.empty_like(albedo)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": albedo, "normal": normal, "roughness": rp},
                 light_dir=light_dir, view_dir=view_dir, _result=loss_p)

        # Loss at x-h
        rm = spy.Tensor.from_numpy(app.device, np.ones((16, 16), dtype=np.float32) * 0.49)
        loss_m = spy.Tensor.empty_like(albedo)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": albedo, "normal": normal, "roughness": rm},
                 light_dir=light_dir, view_dir=view_dir, _result=loss_m)

        # Numerical gradient
        grad = (loss_p.to_numpy() - loss_m.to_numpy()) / (2.0 * h)
        mean_grad = float(np.mean(grad))

        # Both loss+ and loss- should be > 0 since we perturbed away from ref
        mean_lp = float(np.mean(loss_p.to_numpy()))
        mean_lm = float(np.mean(loss_m.to_numpy()))
        check(mean_lp > 0, f"loss_plus > 0: {mean_lp:.6f}")
        check(mean_lm > 0, f"loss_minus > 0: {mean_lm:.6f}")
        check(abs(mean_grad) > 0, f"non-zero gradient: {mean_grad:.6f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_4_2_loss_viz.slang")
        check(mod is not None, "Phase 4.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 5.0 Numerical Gradient - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_numerical_gradient(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
