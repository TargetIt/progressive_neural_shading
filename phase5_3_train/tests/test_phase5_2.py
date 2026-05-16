# Phase 5.2 Test Suite — Adam Optimizer
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
        mod = spy.Module.load_from_file(app.device, "step_5_2_adam.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_adam_optimization():
    print("--- Adam Optimization (loss decreases) ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_2_adam.slang")

        # Target material with uniform values
        target_albedo = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8, 3), dtype=np.float32) * 0.7)
        target_normal = spy.Tensor.from_numpy(app.device,
            np.zeros((8, 8, 3), dtype=np.float32))
        target_roughness = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8), dtype=np.float32) * 0.3)

        # Trainable material (different initialization)
        train_albedo = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8, 3), dtype=np.float32) * 0.5)
        train_normal = spy.Tensor.from_numpy(app.device,
            np.zeros((8, 8, 3), dtype=np.float32))
        train_roughness = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8), dtype=np.float32) * 0.7)

        # Gradient and Adam state tensors
        ag = spy.Tensor.zeros_like(target_albedo)
        ng = spy.Tensor.zeros_like(target_normal)
        rg = spy.Tensor.zeros_like(target_roughness)
        ma = spy.Tensor.zeros_like(target_albedo)
        va = spy.Tensor.zeros_like(target_albedo)
        mn = spy.Tensor.zeros_like(target_normal)
        vn = spy.Tensor.zeros_like(target_normal)
        mr = spy.Tensor.zeros_like(target_roughness)
        vr = spy.Tensor.zeros_like(target_roughness)

        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        view_dir = spy.float3(0, 0, 1)
        lr = 0.01

        # Render target reference
        ref = spy.Tensor.empty_like(target_albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": target_albedo, "normal": target_normal,
                             "roughness": target_roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=ref)

        # Compute loss before optimization
        loss_before = spy.Tensor.empty_like(ref)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": train_albedo, "normal": train_normal,
                           "roughness": train_roughness},
                 light_dir=light_dir, view_dir=view_dir, _result=loss_before)
        loss_before_val = float(np.mean(loss_before.to_numpy()))

        # Optimization steps
        for step in range(20):
            ag.zero_(); ng.zero_(); rg.zero_()
            mod.calculate_grad(pixel=spy.call_id(), reference=ref,
                               material={"albedo": train_albedo, "normal": train_normal,
                                         "roughness": train_roughness,
                                         "albedo_grad": ag, "normal_grad": ng,
                                         "roughness_grad": rg},
                               light_dir=light_dir, view_dir=view_dir)
            mod.optimizer_step3(train_albedo, ag, ma, va, lr, step + 1, False)
            mod.optimizer_step3(train_normal, ng, mn, vn, lr, step + 1, True)
            mod.optimizer_step1(train_roughness, rg, mr, vr, lr, step + 1)

        # Compute loss after optimization
        loss_after = spy.Tensor.empty_like(ref)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": train_albedo, "normal": train_normal,
                           "roughness": train_roughness},
                 light_dir=light_dir, view_dir=view_dir, _result=loss_after)
        loss_after_val = float(np.mean(loss_after.to_numpy()))

        check(loss_after_val < loss_before_val,
              f"loss decreased: {loss_before_val:.6f} -> {loss_after_val:.6f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_1_ad.slang")
        check(mod is not None, "Phase 5.1 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 5.2 Adam Optimizer - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_adam_optimization(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
