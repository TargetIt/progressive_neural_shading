# Phase 3.1 Test Suite — SSAA Pipeline
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
        mod = spy.Module.load_from_file(app.device, "step_3_1_ssaa.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_ssaa_downsample():
    print("--- SSAA Downsample (4x2 passes) ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_3_1_ssaa.slang")

        # Create 16x16 hi-res data (simulates 4x SSAA of a 4x4 base image)
        src = spy.Tensor.from_numpy(app.device, np.ones((16, 16, 3), dtype=np.float32))
        # Downsample 4x via two 2x passes
        tmp = spy.Tensor.empty(device=app.device, shape=(8, 8), dtype=src.dtype)
        mod.downsample3(spy.call_id(), src, _result=tmp)
        dest = spy.Tensor.empty(device=app.device, shape=(4, 4), dtype=tmp.dtype)
        mod.downsample3(spy.call_id(), tmp, _result=dest)

        arr = dest.to_numpy()
        check(dest.shape == (4, 4), f"final shape: {dest.shape}")
        check(abs(float(arr.mean()) - 1.0) < 0.01, f"mean preserved: {float(arr.mean()):.3f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_3_0_hi_res.slang")
        check(mod is not None, "Phase 3.0 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 3.1 SSAA Pipeline - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_ssaa_downsample(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
