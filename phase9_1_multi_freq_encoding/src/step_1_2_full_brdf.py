# Phase 1.2: Full BRDF with Textures — Equivalent to step_01_basicprogram
# =========================================================================
# 从 Phase 1.1 演进: 加载真实纹理 (反照率/法线/粗糙度贴图)。
# 功能等价于参考代码 step_01_basicprogram.py。
#
# 新增:
#   - spy.Tensor.load_from_image(): 从 JPEG/PNG 加载纹理到 GPU
#   - MaterialParameters: 封装纹理的 struct
#   - linearize=True: 去除 sRGB gamma 校正
#   - scale/offset: 法线贴图从 [0,1] 映射到 [-1,1]
#
# 运行:
#   python src/step_1_2_full_brdf.py
# 应该看到用 PBR 材质渲染的石材地面。

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 1.2: Full BRDF with Textures", width=1024, height=1024)

# 编译 shader
print("Compiling shader...")
module = spy.Module.load_from_file(app.device, "step_1_2_full_brdf.slang")

# 加载纹理 (assets/ 目录)
data_path = Path(__file__).parent.parent / "assets"

# 反照率贴图 (sRGB → 线性空间)
albedo_map = spy.Tensor.load_from_image(
    app.device, data_path / "PavingStones070_2K.diffuse.jpg", linearize=True
)

# 法线贴图 ([0,1] → [-1,1])
normal_map = spy.Tensor.load_from_image(
    app.device, data_path / "PavingStones070_2K.normal.jpg", scale=2, offset=-1
)

# 粗糙度贴图 (灰度)
roughness_map = spy.Tensor.load_from_image(
    app.device, data_path / "PavingStones070_2K.roughness.jpg", grayscale=True
)

print("Running. Press ESC to exit.")
while app.process_events():
    # 分配输出 Tensor
    output = spy.Tensor.empty_like(albedo_map)

    # 渲染
    module.render(
        pixel=spy.call_id(),
        material={
            "albedo": albedo_map,
            "normal": normal_map,
            "roughness": roughness_map,
        },
        light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
        view_dir=spy.float3(0, 0, 1),
        _result=output,
    )

    # 显示 (tonemap: 需要 ACES 色调映射, 见 app.slang)
    app.blit(output, tonemap=True)

    app.present()
