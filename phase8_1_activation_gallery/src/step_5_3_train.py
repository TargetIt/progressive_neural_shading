# Phase 5.3: Full Training Loop — Equivalent to step_05_train
# =============================================================
# Step 5 最终形态。完整的材质纹理优化训练管线:
#   - 可微渲染 (Phase 5.1)
#   - SSAA 参考采样 + LCG 随机方向
#   - Adam 优化器 (Phase 5.2)
#   - 学习率衰减
# 功能等价于参考代码 step_05_train.py。
#
# Run: python src/step_5_3_train.py

import math
import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(width=3092, height=1024, title="Phase 5.3: Full Training Loop")
module = spy.Module.load_from_file(app.device, "step_5_3_train.slang")

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


# Downsampled reference maps
lr_albedo_map = downsample(albedo_map, 2)
lr_normal_map = downsample(normal_map, 2)
lr_roughness_map = downsample(roughness_map, 2)

# Trainable parameters
lr_trained_albedo_map = spy.Tensor.zeros_like(lr_albedo_map)
lr_trained_normal_map = spy.Tensor.zeros_like(lr_normal_map)
lr_trained_roughness_map = spy.Tensor.zeros_like(lr_roughness_map)

module.init3(lr_trained_albedo_map, spy.float3(0.5, 0.5, 0.5))
module.init_normal(lr_trained_normal_map)
module.init1(lr_trained_roughness_map, 0.5)

# Gradient tensors
lr_albedo_grad = spy.Tensor.zeros_like(lr_albedo_map)
lr_normal_grad = spy.Tensor.zeros_like(lr_normal_map)
lr_roughness_grad = spy.Tensor.zeros_like(lr_roughness_map)

# Adam state tensors
m_albedo = spy.Tensor.zeros_like(lr_albedo_grad)
v_albedo = spy.Tensor.zeros_like(lr_albedo_grad)
m_normal = spy.Tensor.zeros_like(lr_normal_grad)
v_normal = spy.Tensor.zeros_like(lr_normal_grad)
m_roughness = spy.Tensor.zeros_like(lr_roughness_grad)
v_roughness = spy.Tensor.zeros_like(lr_roughness_grad)

optimize_counter = 0

print("Running. Press ESC to exit.")
while app.process_events():
    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    xpos = 0

    # Full res rendered output BRDF from full res inputs
    output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=output)

    # Downsample the output tensor for reference
    reference = downsample(output, 2)

    # Panel 1: reference
    app.blit(reference, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0))
    xpos += 1024 + 10

    # Panel 2: render with trained materials
    lr_output = spy.Tensor.empty_like(reference)
    module.render(pixel=spy.call_id(),
                  material={"albedo": lr_trained_albedo_map, "normal": lr_trained_normal_map,
                            "roughness": lr_trained_roughness_map},
                  light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=lr_output)

    app.blit(lr_output, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0))
    xpos += 1024 + 10

    # Loss for original materials (reference baseline)
    orig_loss_output = spy.Tensor.empty_like(reference)
    module.loss(pixel=spy.call_id(), reference=reference,
                material={"albedo": lr_albedo_map, "normal": lr_normal_map,
                          "roughness": lr_roughness_map},
                light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=orig_loss_output)

    # Loss for trained materials
    loss_output = spy.Tensor.empty_like(reference)
    module.loss(pixel=spy.call_id(), reference=reference,
                material={"albedo": lr_trained_albedo_map, "normal": lr_trained_normal_map,
                          "roughness": lr_trained_roughness_map},
                light_dir=light_dir, view_dir=spy.float3(0, 0, 1), _result=loss_output)

    # Panel 3: loss
    app.blit(loss_output, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0), tonemap=False)
    xpos += 1024 + 10

    # Learning rate schedule: linear decay
    training_progress = min(optimize_counter / 3000, 1.0)
    learning_rate = 0.002 * (1.0 - training_progress) + 0.0002 * training_progress

    # Optimization iterations
    for _ in range(50):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            pixel=spy.grid(shape=lr_albedo_map.shape),
            material={"albedo": lr_trained_albedo_map, "normal": lr_trained_normal_map,
                      "roughness": lr_trained_roughness_map,
                      "albedo_grad": lr_albedo_grad, "normal_grad": lr_normal_grad,
                      "roughness_grad": lr_roughness_grad},
            ref_material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map})
        optimize_counter += 1

        # Adam updates
        module.optimizer_step3(lr_trained_albedo_map, lr_albedo_grad,
                               m_albedo, v_albedo, learning_rate, optimize_counter, False)
        module.optimizer_step3(lr_trained_normal_map, lr_normal_grad,
                               m_normal, v_normal, learning_rate, optimize_counter, True)
        module.optimizer_step1(lr_trained_roughness_map, lr_roughness_grad,
                               m_roughness, v_roughness, learning_rate, optimize_counter)

    # Report loss values
    orig_loss_np = orig_loss_output.to_numpy()
    orig_loss_value = float(np.mean(orig_loss_np))
    loss_np = loss_output.to_numpy()
    loss_value = float(np.mean(loss_np))
    print(f"  Step {optimize_counter:4d} | Loss: {loss_value:.6f} | Baseline: {orig_loss_value:.6f}")

    app.present()
