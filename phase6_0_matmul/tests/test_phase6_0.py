# Phase 6.0 Test Suite — Matrix Multiply in Shader
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
        mod = spy.Module.load_from_file(app.device, "step_6_0_matmul.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_matmul_known_values():
    print("--- Matmul Known Values ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=4, height=4)
        mod = spy.Module.load_from_file(app.device, "step_6_0_matmul.slang")

        # Identity-like 3x2 weight matrix
        weights = spy.Tensor.from_numpy(app.device, np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=np.float32))

        # Render at a specific pixel
        out = spy.Tensor.empty(app.device, shape=(4, 4), dtype=spy.float3)
        mod.render(pixel=spy.int2(2, 2), resolution=spy.int2(4, 4),
                   weights=weights, _result=out)
        result = out.to_numpy()
        # Pixel (2,2) has uv = (2.5/4, 2.5/4) = (0.625, 0.625)
        # Expected: W * (0.625, 0.625) = (0.625, 0.625, 0.625)
        expected = np.array([0.625, 0.625, 0.625], dtype=np.float32)
        check(np.allclose(result[2, 2], expected, atol=1e-4),
              f"matmul: got {result[2,2]}, expected {expected}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_matmul_uv_dependence():
    print("--- Matmul UV Dependence ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=8, height=8)
        mod = spy.Module.load_from_file(app.device, "step_6_0_matmul.slang")

        # W = [[1,0],[0,1],[0,0]] -> R=u, G=v, B=0
        weights = spy.Tensor.from_numpy(app.device, np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ], dtype=np.float32))

        out = spy.Tensor.empty(app.device, shape=(8, 8), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(8, 8),
                   weights=weights, _result=out)
        result = out.to_numpy()

        # Check that R increases with x (u), G increases with y (v)
        r_first = result[0, 0, 0]  # top-left R
        r_last = result[0, -1, 0]  # top-right R
        g_first = result[0, 0, 1]  # top-left G
        g_last = result[-1, 0, 1]  # bottom-left G
        check(r_last > r_first, f"R increases with u: {r_first} -> {r_last}")
        check(g_last > g_first, f"G increases with v: {g_first} -> {g_last}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_backward_compat():
    print("--- Backward Compat ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_5_3_train.slang")
        check(mod is not None, "Phase 5.3 shader OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 6.0 Matrix Multiply - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_matmul_known_values(); test_matmul_uv_dependence(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
