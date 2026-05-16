# Phase 2.0: Python Manual Downsampling
# =====================================
# 在 Python/NumPy 端手动实现 2x2 box filter 降采样。
# 这是理解 mipmap 的第一步: 为什么需要降采样? 怎么做?
#
# 概念:
#   - Mipmap: 多级分辨率纹理链 (每级是上一级的 1/4 像素数)
#   - Box filter: 4 个相邻像素取平均 = 最简单的降采样
#   - 为什么用 GPU? Phase 2.0 用 CPU 做, 感受性能差异
#
# 运行: python src/step_2_0_downsample.py

from app import App
import slangpy as spy
from pathlib import Path
import numpy as np

app = App(title="Phase 2.0: Python Downsample", width=1024, height=512)
module = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)


def downsample_python(tensor, steps=1):
    """Python 端手动降采样: 2x2 box filter 平均。
    每次调用分辨率减半。
    """
    arr = tensor.to_numpy()
    for _ in range(steps):
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        # 2x2 box filter: reshape → mean over 2x2 blocks
        reshaped = arr[:new_h*2, :new_w*2].reshape(new_h, 2, new_w, 2, -1)
        arr = reshaped.mean(axis=(1, 3))
    # 转回 GPU Tensor
    return spy.Tensor.from_numpy(app.device, arr.astype(np.float32))


print("Running. Press ESC to exit.")
while app.process_events():
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output)

    # Python 端降采样 2 级 (分辨率 / 4)
    downsampled = downsample_python(output, steps=2)
    app.blit(downsampled, tonemap=True)
    app.present()
