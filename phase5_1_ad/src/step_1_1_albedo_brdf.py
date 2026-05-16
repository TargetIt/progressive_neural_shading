# Phase 1.1: Albedo BRDF — Lighting with Hardcoded Color
# ========================================================
# 从 Phase 1.0 演进: 加入 BRDF 光照计算。
# 关键新增: light_dir, view_dir, eval_brdf()
#
# 运行:
#   python src/step_1_1_albedo_brdf.py
# 应该看到蓝色的光照球效果 (中央高光 + 边缘变暗)

from app import App
import slangpy as spy

app = App(title="Phase 1.1: Albedo BRDF — Blue Lighting", width=512, height=512)

print("Compiling shader...")
module = spy.Module.load_from_file(app.device, "step_1_1_albedo_brdf.slang")

output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

# 光源方向和观察方向 (固定)
light_dir = spy.float3(0.3, 0.2, 1.0)   # 从右上方照过来
view_dir = spy.float3(0.0, 0.0, 1.0)     # 正视

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(
        pixel=spy.call_id(),
        light_dir=light_dir,
        view_dir=view_dir,
        _result=output,
    )
    app.blit(output)
    app.present()
