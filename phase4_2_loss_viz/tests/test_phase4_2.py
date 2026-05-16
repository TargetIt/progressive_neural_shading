# Phase 4.2 Test Suite — Loss Visualization
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
        mod = spy.Module.load_from_file(app.device, "step_4_2_loss_viz.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_loss_pipeline():
    print("--- Loss Pipeline (downsample + loss) ---")
    try:
        from app import App; import slangpy as spy
        from pathlib import Path
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_4_2_loss_viz.slang")

        # Create uniform inputs at 32x32
        albedo = spy.Tensor.from_numpy(app.device, np.ones((32, 32, 3), dtype=np.float32) * 0.5)
        normal = spy.Tensor.from_numpy(app.device, np.zeros((32, 32, 3), dtype=np.float32))
        roughness = spy.Tensor.from_numpy(app.device, np.ones((32, 32), dtype=np.float32) * 0.3)
        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        view_dir = spy.float3(0, 0, 1)

        # Full render then downsample for reference
        full = spy.Tensor.empty_like(albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": albedo, "normal": normal, "roughness": roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=full)

        ref = spy.Tensor.empty(device=app.device, shape=(16, 16), dtype=full.dtype)
        mod.downsample3(spy.call_id(), full, _result=ref)

        # Low-res render
        lr_albedo = spy.Tensor.empty(device=app.device, shape=(16, 16, 3), dtype=np.float32)
        mod.downsample3(spy.call_id(), albedo, _result=lr_albedo)
        lr_normal = spy.Tensor.empty(device=app.device, shape=(16, 16, 3), dtype=np.float32)
        mod.downsample3(spy.call_id(), normal, _result=lr_normal)
        lr_roughness = spy.Tensor.empty(device=app.device, shape=(16, 16), dtype=np.float32)
        mod.downsample1(spy.call_id(), roughness, _result=lr_roughness)

        pred = spy.Tensor.empty_like(ref)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": lr_albedo, "normal": lr_normal, "roughness": lr_roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=pred)

        # Loss
        loss_out = spy.Tensor.empty_like(ref)
        mod.loss(pixel=spy.call_id(), reference=ref,
                 material={"albedo": lr_albedo, "normal": lr_normal, "roughness": lr_roughness},
                 light_dir=light_dir, view_dir=view_dir, _result=loss_out)

        arr = loss_out.to_numpy()
        # With uniform inputs, pred should be very close to ref => near-zero loss
        check(float(arr.mean()) < 0.01, f"near-zero loss: {float(arr.mean()):.6f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_2_2_mipmap.slang")
        check(mod is not None, "Phase 2.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 4.2 Loss Visualization - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_loss_pipeline(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
