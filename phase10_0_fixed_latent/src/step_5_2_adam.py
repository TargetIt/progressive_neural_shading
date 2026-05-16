# Phase 5.2: Adam Optimizer
# ==========================
# Demonstrate Adam optimization of material parameters using
# autodiff gradients. Shows loss decreasing over optimization steps.
#
# Run: python src/step_5_2_adam.py

import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 5.2: Adam Optimizer", width=1536, height=512)
module = spy.Module.load_from_file(app.device, "step_5_2_adam.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

# Downsample for faster optimization
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

lr_albedo_map = downsample(albedo_map, 2)
lr_normal_map = downsample(normal_map, 2)
lr_roughness_map = downsample(roughness_map, 2)

# Trainable parameters (initialized to constants)
train_albedo = spy.Tensor.from_numpy(app.device,
    np.ones(lr_albedo_map.shape, dtype=np.float32) * 0.5)
train_normal = spy.Tensor.from_numpy(app.device,
    np.tile(np.array([0, 0, 1], dtype=np.float32),
            (lr_albedo_map.shape[0], lr_albedo_map.shape[1], 1)))
train_roughness = spy.Tensor.from_numpy(app.device,
    np.ones(lr_roughness_map.shape, dtype=np.float32) * 0.5)

# Gradient tensors
albedo_grad = spy.Tensor.zeros_like(lr_albedo_map)
normal_grad = spy.Tensor.zeros_like(lr_normal_map)
roughness_grad = spy.Tensor.zeros_like(lr_roughness_map)

# Adam moment tensors
m_albedo = spy.Tensor.zeros_like(train_albedo)
v_albedo = spy.Tensor.zeros_like(train_albedo)
m_normal = spy.Tensor.zeros_like(train_normal)
v_normal = spy.Tensor.zeros_like(train_normal)
m_roughness = spy.Tensor.zeros_like(train_roughness)
v_roughness = spy.Tensor.zeros_like(train_roughness)

light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
view_dir = spy.float3(0, 0, 1)
learning_rate = 0.002
optimize_counter = 0

print("Running. Press ESC to exit.")
while app.process_events():
    # Reference render (from full-res then downsample)
    full_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=full_output)
    reference = downsample(full_output, 2)

    # Optimize: multiple steps per frame
    for _ in range(10):
        optimize_counter += 1
        albedo_grad.zero_(); normal_grad.zero_(); roughness_grad.zero_()

        module.calculate_grad(pixel=spy.call_id(), reference=reference,
                              material={"albedo": train_albedo, "normal": train_normal,
                                        "roughness": train_roughness,
                                        "albedo_grad": albedo_grad, "normal_grad": normal_grad,
                                        "roughness_grad": roughness_grad},
                              light_dir=light_dir, view_dir=view_dir)

        # Adam updates
        module.optimizer_step3(train_albedo, albedo_grad, m_albedo, v_albedo,
                               learning_rate, optimize_counter, False)
        module.optimizer_step3(train_normal, normal_grad, m_normal, v_normal,
                               learning_rate, optimize_counter, True)
        module.optimizer_step1(train_roughness, roughness_grad, m_roughness, v_roughness,
                               learning_rate, optimize_counter)

    # Current prediction
    pred = spy.Tensor.empty_like(reference)
    module.render(pixel=spy.call_id(),
                  material={"albedo": train_albedo, "normal": train_normal,
                            "roughness": train_roughness},
                  light_dir=light_dir, view_dir=view_dir, _result=pred)

    # Current loss
    loss_out = spy.Tensor.empty_like(reference)
    module.loss(pixel=spy.call_id(), reference=reference,
                material={"albedo": train_albedo, "normal": train_normal,
                          "roughness": train_roughness},
                light_dir=light_dir, view_dir=view_dir, _result=loss_out)

    loss_val = float(np.mean(loss_out.to_numpy()))
    print(f"  Step {optimize_counter:4d} | Loss: {loss_val:.6f}", end="\r")

    panel_w = 512
    app.blit(reference, size=spy.int2(panel_w, 512), offset=spy.int2(0, 0))
    app.blit(pred, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w, 0))
    app.blit(loss_out, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w * 2, 0), tonemap=False)
    app.present()
