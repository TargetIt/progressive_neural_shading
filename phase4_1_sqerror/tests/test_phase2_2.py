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
        mod = spy.Module.load_from_file(app.device, "step_2_2_mipmap.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_gpu_downsample():
    print("--- GPU Downsample ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_2_2_mipmap.slang")
        src = spy.Tensor.from_numpy(app.device, np.ones((64, 64, 3), dtype=np.float32))
        dest = spy.Tensor.empty(device=app.device, shape=(32, 32), dtype=src.dtype)
        mod.downsample3(spy.call_id(), src, _result=dest)
        arr = dest.to_numpy()
        check(dest.shape == (32, 32), f"shape halved: {dest.shape}")
        check(abs(float(arr.mean()) - 1.0) < 0.01, f"mean: {float(arr.mean()):.3f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")
        check(mod is not None, "Phase 1.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 2.2 Mipmap Chain - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_gpu_downsample(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
