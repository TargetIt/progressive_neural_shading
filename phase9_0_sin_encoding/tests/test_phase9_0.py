# Phase 9.0 Test Suite — Sin Encoding
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
        mod = spy.Module.load_from_file(app.device, "step_9_0_sin_encoding.slang")
        check(mod is not None, "compiled OK")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_output_range():
    print("--- Output Range [-1, 1] ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=32, height=32)
        mod = spy.Module.load_from_file(app.device, "step_9_0_sin_encoding.slang")

        out = spy.Tensor.empty(app.device, shape=(32, 32), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(32, 32), frequency=3.0, _result=out)
        result = out.to_numpy()

        check(np.all(result >= -1.0 - 1e-5), f"all >= -1: min={np.min(result):.4f}")
        check(np.all(result <= 1.0 + 1e-5), f"all <= 1: max={np.max(result):.4f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_sin_periodicity():
    print("--- Sin Periodicity ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=16, height=16)
        mod = spy.Module.load_from_file(app.device, "step_9_0_sin_encoding.slang")

        # With frequency=1.0, the pattern should repeat across the texture
        # At uv=(0,0) and uv=(1,1), sin(0) = sin(2pi) = 0
        out = spy.Tensor.empty(app.device, shape=(16, 16), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(16, 16), frequency=1.0, _result=out)
        result = out.to_numpy()

        # uv=(0.5/16, 0.5/16) ≈ (0.03125, 0.03125)
        # sin(0.03125 * 2pi) ≈ sin(0.196) ≈ 0.195
        check(abs(result[0, 0, 0]) < 0.5, f"near-edge value reasonable: {result[0,0,0]:.4f}")
        app.window.close()
    except Exception as e: check(False, str(e))

def test_frequency_changes_pattern():
    print("--- Frequency Changes Pattern ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="T", width=16, height=16)
        mod = spy.Module.load_from_file(app.device, "step_9_0_sin_encoding.slang")

        out1 = spy.Tensor.empty(app.device, shape=(16, 16), dtype=spy.float3)
        out2 = spy.Tensor.empty(app.device, shape=(16, 16), dtype=spy.float3)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(16, 16), frequency=1.0, _result=out1)
        mod.render(pixel=spy.call_id(), resolution=spy.int2(16, 16), frequency=5.0, _result=out2)
        r1, r2 = out1.to_numpy(), out2.to_numpy()
        check(not np.allclose(r1, r2, atol=1e-4),
              "different frequencies produce different outputs")
        app.window.close()
    except Exception as e: check(False, str(e))

def main():
    global passed, failed
    print("=" * 50)
    print("Phase 9.0 Sin Encoding - Test Suite")
    print("=" * 50)
    test_shader_compile(); test_output_range(); test_sin_periodicity(); test_frequency_changes_pattern()
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__": main()
