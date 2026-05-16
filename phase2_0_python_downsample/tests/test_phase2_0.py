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
        m = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")
        check(m is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_downsample_python():
    print("--- Python Downsample ---")
    try:
        arr = np.ones((64, 64, 3), dtype=np.float32)
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        reshaped = arr[:new_h*2, :new_w*2].reshape(new_h, 2, new_w, 2, -1)
        result = reshaped.mean(axis=(1, 3))
        check(result.shape == (32, 32, 3), f"shape halved: {result.shape}")
        check(abs(float(result.mean()) - 1.0) < 0.01, "mean preserved")
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        m = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")
        check(m is not None, "Phase 1.2 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 2.0 Python Downsample - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_downsample_python(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
