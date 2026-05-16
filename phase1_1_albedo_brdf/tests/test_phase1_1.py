# Phase 1.1 Test Suite — Albedo BRDF
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_imports():
    print("\n--- Import Tests ---")
    try:
        import slangpy as spy
        from app import App
        from trace import tensor_stats, verify_lighting_variation
        check(True, "all imports OK")
    except Exception as e:
        check(False, str(e))


def test_shader_compile():
    print("\n--- Shader Compile ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_1_albedo_brdf.slang")
        check(module is not None, "shader compiled")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_output_not_solid():
    """BRDF 输出不应该是单一颜色 (因为有光照变化)。"""
    print("\n--- Output Variation Test ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_1_albedo_brdf.slang")
        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        module.render(pixel=spy.call_id(),
                      light_dir=spy.float3(0.3, 0.2, 1.0),
                      view_dir=spy.float3(0, 0, 1),
                      _result=output)
        arr = output.to_numpy()
        # 标准差 > 0 说明有光照变化
        std_r = float(arr[..., 0].std())
        check(std_r > 0.001, f"lighting variation (std={std_r:.4f})")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_trace_stats():
    print("\n--- Trace Stats ---")
    try:
        from app import App; import slangpy as spy
        from trace import tensor_stats, verify_lighting_variation
        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_1_albedo_brdf.slang")
        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        module.render(pixel=spy.call_id(),
                      light_dir=spy.float3(0.3, 0.2, 1.0),
                      view_dir=spy.float3(0, 0, 1),
                      _result=output)
        stats = tensor_stats(output)
        check(stats['shape'] == (64, 64), f"shape={stats['shape']}")
        check(stats['max'] > stats['min'], "max > min (lighting variation)")
        ok, msg = verify_lighting_variation(output)
        check(ok, f"lighting variation: {msg}")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_render_loop():
    """验证完整渲染循环不崩溃。"""
    print("\n--- Render Loop Test ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_1_albedo_brdf.slang")
        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        light_dir = spy.float3(0.3, 0.2, 1.0)
        view_dir = spy.float3(0, 0, 1)
        # Run a few frames
        for _ in range(3):
            app.process_events()
            module.render(pixel=spy.call_id(), light_dir=light_dir,
                          view_dir=view_dir, _result=output)
            app.blit(output)
        check(True, "render loop completed")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 1.1: Albedo BRDF — Test Suite")
    print("=" * 60)
    test_imports()
    test_shader_compile()
    test_output_not_solid()
    test_trace_stats()
    test_render_loop()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
