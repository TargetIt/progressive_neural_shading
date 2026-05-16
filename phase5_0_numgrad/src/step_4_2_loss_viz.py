# Phase 4.2: Loss Visualization — Equivalent to step_04_loss
# ===========================================================
# Step 4 最终形态。三栏显示:
#   左: reference (全分辨率渲染后降采样)
#   中: loss (平方误差热图, no tonemap)
#   右: prediction (低分辨率材质直接渲染)
# 功能等价于参考代码 step_04_loss.py。
#
# Run: python src/step_4_2_loss_viz.py

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 4.2: Loss Visualization", width=3092, height=1024)
module = spy.Module.load_from_file(app.device, "step_4_2_loss_viz.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)


def downsample(source, steps):
    for _ in range(steps):
        dest = spy.Tensor.empty(device=app.device,
            shape=(source.shape[0] // 2, source.shape[1] // 2), dtype=source.dtype)
        if dest.dtype.name == "vector":
            module.downsample3(spy.call_id(), source, _result=dest)
        else:
            module.downsample1(spy.call_id(), source, _result=dest)
        source = dest
    return source


print("Running. Press ESC to exit.")
while app.process_events():
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    view_dir = spy.float3(0, 0, 1)

    # Full resolution render
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=output)

    # Downsample output 2x for reference
    reference = downsample(output, 2)

    # Quarter-res prediction: downsample materials, then render
    lr_albedo = downsample(albedo_map, 2)
    lr_normal = downsample(normal_map, 2)
    lr_roughness = downsample(roughness_map, 2)

    lr_output = spy.Tensor.empty_like(reference)
    module.render(pixel=spy.call_id(),
                  material={"albedo": lr_albedo, "normal": lr_normal, "roughness": lr_roughness},
                  light_dir=light_dir, view_dir=view_dir, _result=lr_output)

    # Loss between reference and low-res prediction
    loss_output = spy.Tensor.empty_like(reference)
    module.loss(pixel=spy.call_id(), reference=reference,
                material={"albedo": lr_albedo, "normal": lr_normal, "roughness": lr_roughness},
                light_dir=light_dir, view_dir=view_dir, _result=loss_output)

    # Three-panel display
    panel_w = 1024
    app.blit(reference, size=spy.int2(panel_w, 1024), offset=spy.int2(0, 0))
    app.blit(loss_output, size=spy.int2(panel_w, 1024), offset=spy.int2(panel_w + 10, 0), tonemap=False)
    app.blit(lr_output, size=spy.int2(panel_w, 1024), offset=spy.int2(2 * panel_w + 20, 0))
    app.present()
