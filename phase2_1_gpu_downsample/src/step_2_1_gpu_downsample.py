# Phase 2.1: GPU Downsampling
# =============================
# 在 GPU 上并行降采样 (对比 Phase 2.0 的 Python 端实现)。
#
# 运行: python src/step_2_1_gpu_downsample.py

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 2.1: GPU Downsample", width=1024, height=512)
module = spy.Module.load_from_file(app.device, "step_2_1_gpu_downsample.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)


def downsample_gpu(source, steps):
    """GPU 端降采样: 每次分辨率减半。"""
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

    downsampled = downsample_gpu(output, steps=2)
    app.blit(downsampled, tonemap=True)
    app.present()
