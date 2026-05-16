# Phase 2.2: Mipmap Chain — Equivalent to step_02_mipmap
# ========================================================
# Step 2 最终形态。功能等价于参考代码 step_02_mipmap.py。
#
# 运行: python src/step_2_2_mipmap.py

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 2.2: Mipmap Chain", width=1024, height=1024)
module = spy.Module.load_from_file(app.device, "step_2_2_mipmap.slang")

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
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output)

    output = downsample(output, 2)
    app.blit(output)
    app.present()
