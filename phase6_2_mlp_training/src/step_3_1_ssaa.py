# Phase 3.1: SSAA Pipeline — Equivalent to step_03_supersample
# ============================================================
# Step 3 最终形态。通过 4x4 子像素平均实现 SSAA (Supersampling AA)。
# 功能等价于参考代码 step_03_supersample.py。
#
# 对比:
#   左: 标准渲染 (1x 分辨率)
#   右: SSAA 渲染 (4x4 子像素平均后降采样)
#
# Run: python src/step_3_1_ssaa.py

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 3.1: SSAA Pipeline", width=1024, height=1024)
module = spy.Module.load_from_file(app.device, "step_3_1_ssaa.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

print("Running. Press ESC to exit.")
while app.process_events():
    # Standard render at base resolution
    output_std = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output_std)

    # SSAA render: 4x4 sub-pixel averaging per output pixel
    # => equivalently: render at 4x res, then downsample 4x
    output_hires = spy.Tensor.empty(device=app.device,
        shape=(albedo_map.shape[0] * 4, albedo_map.shape[1] * 4), dtype=albedo_map.dtype)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output_hires)

    # Downsample 4x via two 2x passes
    output_ssaa = output_hires
    for _ in range(2):
        dest = spy.Tensor.empty(device=app.device,
            shape=(output_ssaa.shape[0] // 2, output_ssaa.shape[1] // 2), dtype=output_ssaa.dtype)
        module.downsample3(spy.call_id(), output_ssaa, _result=dest)
        output_ssaa = dest

    # Blit side by side: standard | SSAA
    app.blit(output_std, size=spy.int2(512, 1024), offset=spy.int2(0, 0))
    app.blit(output_ssaa, size=spy.int2(512, 1024), offset=spy.int2(512, 0))
    app.present()
