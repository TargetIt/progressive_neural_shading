# Phase 4.0 Test Suite — Per-Pixel Absolute Difference
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
        mod = spy.Module.load_from_file(app.device, "step_4_0_diff.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_abs_diff_identical():
    print("--- Abs Diff Identical Inputs = Zero ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_4_0_diff.slang")

        # Create constant inputs
        albedo = spy.Tensor.from_numpy(app.device, np.ones((16, 16, 3), dtype=np.float32) * 0.5)
        normal = spy.Tensor.from_numpy(app.device, np.zeros((16, 16, 3), dtype=np.float32))
        roughness = spy.Tensor.from_numpy(app.device, np.ones((16, 16), dtype=np.float32) * 0.5)

        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        view_dir = spy.float3(0, 0, 1)

        # Render reference
        ref = spy.Tensor.empty_like(albedo)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": albedo, "normal": normal, "roughness": roughness},
                   light_dir=light_dir, view_dir=view_dir, _result=ref)

        # Diff against itself should give ~zero (identical material)
        diff = spy.Tensor.empty_like(albedo)
        mod.abs_diff(pixel=spy.call_id(), reference=ref,
                     material={"albedo": albedo, "normal": normal, "roughness": roughness},
                     light_dir=light_dir, view_dir=view_dir, _result=diff)

        arr = diff.to_numpy()
        check(float(arr.mean()) < 0.001, f"near-zero diff: {float(arr.mean()):.6f}")
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
    print("Phase 4.0 Per-Pixel Diff - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_abs_diff_identical(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
