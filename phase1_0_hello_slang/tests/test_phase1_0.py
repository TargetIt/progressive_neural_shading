# Phase 1.0 Test Suite — Hello Slang
# ====================================
# 由于需要 GPU + slangpy, 测试策略:
#   - API 测试: import 验证 (无 GPU)
#   - 编译测试: 需要 GPU (标记为 integration)
#   - 输出验证: 需要 GPU + 窗口 (标记为 visual)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_imports():
    """验证 Python 模块可以正常导入。"""
    print("\n--- Import Tests ---")
    try:
        import slangpy as spy
        check(True, "slangpy imported")
    except ImportError as e:
        check(False, f"slangpy import: {e}")

    try:
        from app import App
        check(True, "app.App imported")
    except Exception as e:
        check(False, f"app import: {e}")

    try:
        from trace import tensor_stats, print_stats, verify_solid_color
        check(True, "trace module imported")
    except Exception as e:
        check(False, f"trace import: {e}")


def test_app_creation():
    """验证 App 可以创建 (需要 GPU)。"""
    print("\n--- App Creation Test ---")
    try:
        from app import App
        app = App(title="Test", width=64, height=64)
        check(app.device is not None, "device created")
        check(app.window is not None, "window created")
        check(app.process_events(), "process_events returns True")
        app.window.close()
        check(True, "app closed cleanly")
    except Exception as e:
        check(False, f"App creation failed: {e}")


def test_shader_compile():
    """验证 shader 可以编译。"""
    print("\n--- Shader Compile Test ---")
    try:
        from app import App
        import slangpy as spy

        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_0_hello.slang")
        check(module is not None, "shader compiled")
        app.window.close()
    except Exception as e:
        check(False, f"Shader compile failed: {e}")


def test_output_color():
    """验证 shader 输出纯红色。"""
    print("\n--- Output Color Test ---")
    try:
        from app import App
        import slangpy as spy
        from trace import verify_solid_color

        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_0_hello.slang")

        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        module.render(pixel=spy.call_id(), _result=output)

        ok, msg = verify_solid_color(output, (1.0, 0.0, 0.0))
        check(ok, f"all pixels red: {msg}")
        app.window.close()
    except Exception as e:
        check(False, f"Output color test failed: {e}")


def test_tensor_stats():
    """验证 trace 统计功能。"""
    print("\n--- Tensor Stats Test ---")
    try:
        from app import App
        import slangpy as spy
        from trace import tensor_stats

        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_0_hello.slang")
        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        module.render(pixel=spy.call_id(), _result=output)

        stats = tensor_stats(output)
        check(stats['shape'] == (64, 64), f"shape is {stats['shape']}")
        check(abs(stats['mean'] - 1.0/3.0) < 0.1,  # all red → R mean ≈ 1.0
              f"R channel mean ≈ 1.0 (got {stats['mean']:.2f})")
        app.window.close()
    except Exception as e:
        check(False, f"Tensor stats test failed: {e}")


def test_app_blit():
    """验证 app.blit 不崩溃。"""
    print("\n--- Blit Test ---")
    try:
        from app import App
        import slangpy as spy

        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_0_hello.slang")
        output = spy.Tensor.empty(app.device, shape=(64, 64), dtype=spy.float3)
        module.render(pixel=spy.call_id(), _result=output)

        # Blit should not crash
        app.blit(output)
        check(True, "blit completed without error")
        app.window.close()
    except Exception as e:
        check(False, f"Blit test failed: {e}")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 1.0: Hello Slang — Test Suite")
    print("=" * 60)
    test_imports()
    test_app_creation()
    test_shader_compile()
    test_output_color()
    test_tensor_stats()
    test_app_blit()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
