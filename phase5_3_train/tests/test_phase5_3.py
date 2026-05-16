# Phase 5.3 Test Suite — Full Training Loop
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
        mod = spy.Module.load_from_file(app.device, "step_5_3_train.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_training_loop():
    print("--- Training Loop (loss decreases) ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_3_train.slang")

        # Create small materials (8x8 pixels)
        ref_albedo = spy.Tensor.from_numpy(app.device,
            np.ones((32, 32, 3), dtype=np.float32) * 0.7)
        ref_normal = spy.Tensor.from_numpy(app.device,
            np.zeros((32, 32, 3), dtype=np.float32))
        ref_roughness = spy.Tensor.from_numpy(app.device,
            np.ones((32, 32), dtype=np.float32) * 0.3)

        # Trainable: initialized differently (trained at 8x8 after 2x downsample twice)
        train_albedo = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8, 3), dtype=np.float32) * 0.5)
        train_normal = spy.Tensor.from_numpy(app.device,
            np.tile(np.array([0, 0, 1], dtype=np.float32), (8, 8, 1)))
        train_roughness = spy.Tensor.from_numpy(app.device,
            np.ones((8, 8), dtype=np.float32) * 0.7)

        # Gradient tensors
        ag = spy.Tensor.zeros_like(train_albedo)
        ng = spy.Tensor.zeros_like(train_normal)
        rg = spy.Tensor.zeros_like(train_roughness)

        # Adam state
        ma = spy.Tensor.zeros_like(train_albedo)
        va = spy.Tensor.zeros_like(train_albedo)
        mn = spy.Tensor.zeros_like(train_normal)
        vn = spy.Tensor.zeros_like(train_normal)
        mr = spy.Tensor.zeros_like(train_roughness)
        vr = spy.Tensor.zeros_like(train_roughness)

        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))

        # Compute reference (render at full 32x32, then downsample to 8x8 via two 2x passes)
        full = spy.Tensor.empty_like(ref_albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": ref_albedo, "normal": ref_normal, "roughness": ref_roughness},
                   light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=full)
        tmp = spy.Tensor.empty(device=app.device, shape=(16, 16), dtype=full.dtype)
        mod.downsample3(spy.call_id(), full, _result=tmp)
        ref = spy.Tensor.empty(device=app.device, shape=(8, 8), dtype=tmp.dtype)
        mod.downsample3(spy.call_id(), tmp, _result=ref)

        # Loss before training
        loss_before = spy.Tensor.empty_like(ref)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": train_albedo, "normal": train_normal,
                           "roughness": train_roughness},
                 light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=loss_before)
        loss_before_val = float(np.mean(loss_before.to_numpy()))

        # Training steps with calculate_grads
        for step in range(30):
            ag.zero_(); ng.zero_(); rg.zero_()
            mod.calculate_grads(
                seed=spy.wang_hash(seed=step, warmup=2),
                pixel=spy.grid(shape=train_albedo.shape),
                material={"albedo": train_albedo, "normal": train_normal,
                          "roughness": train_roughness,
                          "albedo_grad": ag, "normal_grad": ng, "roughness_grad": rg},
                ref_material={"albedo": ref_albedo, "normal": ref_normal, "roughness": ref_roughness})
            mod.optimizer_step3(train_albedo, ag, ma, va, 0.01, step + 1, False)
            mod.optimizer_step3(train_normal, ng, mn, vn, 0.01, step + 1, True)
            mod.optimizer_step1(train_roughness, rg, mr, vr, 0.01, step + 1)

        # Loss after training
        loss_after = spy.Tensor.empty_like(ref)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": train_albedo, "normal": train_normal,
                           "roughness": train_roughness},
                 light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=loss_after)
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
        mod = spy.Module.load_from_file(app.device, "step_5_2_adam.slang")
        check(mod is not None, "Phase 5.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 5.3 Full Training Loop - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_training_loop(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
