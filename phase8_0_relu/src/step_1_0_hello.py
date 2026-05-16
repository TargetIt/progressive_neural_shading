# Phase 1.0: Hello Slang — Minimal Shader Entry Point
# =====================================================
# 加载 .slang shader → GPU 编译 → 逐像素执行 → 显示
#
# 关键 API:
#   spy.Module.load_from_file(device, "file.slang")  — 编译 shader
#   module.render(pixel=spy.call_id(), _result=output) — 逐像素调用
#   app.blit(output)                                    — 显示到屏幕

from app import App
import slangpy as spy

# Step 1: 创建 App (窗口 + GPU 设备)
app = App(title="Phase 1.0: Hello Slang — Solid Red", width=512, height=512)

# Step 2: 加载并编译 Slang shader
# 首次调用会触发 GPU 编译 (可能需要几秒)
print("Compiling shader...")
module = spy.Module.load_from_file(app.device, "step_1_0_hello.slang")

# Step 3: 分配输出 Tensor (512×512 的 RGB 浮点图像)
output = spy.Tensor.empty(
    device=app.device,
    shape=(512, 512),
    dtype=spy.float3,
)

# Step 4: 渲染循环
print("Running. Press ESC to exit.")
while app.process_events():
    # 调用 GPU shader: 对每个像素并行执行 render()
    module.render(
        pixel=spy.call_id(),   # slangpy 自动分配像素坐标
        _result=output,        # 输出写入此 Tensor
    )

    # 将 Tensor 显示到窗口
    app.blit(output)

    # 提交帧到屏幕
    app.present()
