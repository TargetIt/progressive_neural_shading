# Phase 3.0 Test Suite — Higher Resolution Render
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
        mod = spy.Module.load_from_file(app.device, "step_3_0_hi_res.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_higher_res_render():
    print("--- Higher Resolution Render ---")
    try:
        from app import App; import slangpy as spy
        from pathlib import Path
        app = App(title="T", width=64, height=64)
        mod = spy.Module.load_from_file(app.device, "step_3_0_hi_res.slang")

        # Render at 2x resolution
        output = spy.Tensor.empty(device=app.device, shape=(128, 128), dtype=spy.float3)
        mod.render(pixel=spy.call_id(),
                   material={"albedo": spy.Tensor.from_numpy(app.device, np.ones((64, 64, 3), dtype=np.float32)),
                             "normal": spy.Tensor.from_numpy(app.device, np.zeros((64, 64, 3), dtype=np.float32)),
                             "roughness": spy.Tensor.from_numpy(app.device, np.ones((64, 64), dtype=np.float32))},
                   light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                   view_dir=spy.float3(0, 0, 1), _result=output)

        arr = output.to_numpy()
        check(output.shape == (128, 128), f"2x shape: {output.shape}")
        check(float(arr.mean()) > 0, f"non-zero output: {float(arr.mean()):.3f}")
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
    print("Phase 3.0 Higher Resolution Render - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_higher_res_render(); test_backward_compat()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
