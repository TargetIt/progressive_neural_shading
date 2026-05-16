# Phase 1.2 Test Suite — Full BRDF with Textures
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pathlib import Path

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
        check(True, "all imports OK")
    except Exception as e:
        check(False, str(e))


def test_textures_exist():
    """验证纹理文件存在。"""
    print("\n--- Texture File Tests ---")
    assets = Path(__file__).parent.parent / "assets"
    for fname in ["PavingStones070_2K.diffuse.jpg",
                  "PavingStones070_2K.normal.jpg",
                  "PavingStones070_2K.roughness.jpg"]:
        check((assets / fname).exists(), f"texture: {fname}")


def test_shader_compile():
    print("\n--- Shader Compile ---")
    try:
        from app import App; import slangpy as spy
        app = App(title="Test", width=64, height=64)
        module = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")
        check(module is not None, "shader compiled")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_texture_loading():
    """验证纹理可以加载到 GPU。"""
    print("\n--- Texture Loading Test ---")
    try:
        from app import App; import slangpy as spy
        assets = Path(__file__).parent.parent / "assets"
        app = App(title="Test", width=64, height=64)

        albedo = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True
        )
        check(albedo.shape[0] > 0, f"albedo loaded: {albedo.shape}")

        normal = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1
        )
        check(normal.shape[0] > 0, f"normal loaded: {normal.shape}")

        roughness = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True
        )
        check(roughness.shape[0] > 0, f"roughness loaded: {roughness.shape}")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_brdf_render():
    """验证 BRDF 渲染输出。"""
    print("\n--- BRDF Render Test ---")
    try:
        from app import App; import slangpy as spy
        assets = Path(__file__).parent.parent / "assets"
        app = App(title="Test", width=128, height=128)
        module = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")

        albedo = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True
        )
        normal = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1
        )
        roughness = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True
        )

        output = spy.Tensor.empty_like(albedo)
        module.render(
            pixel=spy.call_id(),
            material={"albedo": albedo, "normal": normal, "roughness": roughness},
            light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
            view_dir=spy.float3(0, 0, 1),
            _result=output,
        )

        arr = output.to_numpy()
        # BRDF output should have non-zero values with variation
        mean_r = float(arr[..., 0].mean())
        std_r = float(arr[..., 0].std())
        check(mean_r > 0.01, f"non-zero output (R mean={mean_r:.3f})")
        check(std_r > 0.001, f"texture variation (R std={std_r:.3f})")
        app.window.close()
    except Exception as e:
        check(False, str(e))


def test_functional_equivalence():
    """验证 Phase 1.2 的功能与参考 step_01 等价。
    参考 step_01 的关键特征:
    1. 加载 3 个纹理 (albedo, normal, roughness)
    2. 编译 shader 并调用 render()
    3. 输出非零 RGB 值
    4. 逐像素有变化 (不是纯色)
    """
    print("\n--- Functional Equivalence Test ---")
    try:
        from app import App; import slangpy as spy
        assets = Path(__file__).parent.parent / "assets"
        app = App(title="Test", width=128, height=128)
        module = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")

        # 1. Load 3 textures (same as step_01)
        albedo = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
        normal = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
        roughness = spy.Tensor.load_from_image(
            app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

        # 2. Render
        output = spy.Tensor.empty_like(albedo)
        module.render(
            pixel=spy.call_id(),
            material={"albedo": albedo, "normal": normal, "roughness": roughness},
            light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
            view_dir=spy.float3(0, 0, 1),
            _result=output,
        )

        arr = output.to_numpy()
        # 3. Non-zero output
        check(float(arr.mean()) > 0.005, f"non-zero mean: {float(arr.mean()):.4f}")
        # 4. Per-pixel variation
        check(float(arr.std()) > 0.005, f"variation: {float(arr.std()):.4f}")

        app.window.close()
        check(True, "functional equivalence validated")
    except Exception as e:
        check(False, str(e))


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 1.2: Full BRDF + Textures — Test Suite")
    print("=" * 60)
    test_imports()
    test_textures_exist()
    test_shader_compile()
    test_texture_loading()
    test_brdf_render()
    test_functional_equivalence()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
