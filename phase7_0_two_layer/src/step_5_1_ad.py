# Phase 5.1: Automatic Differentiation with Slang bwd_diff
# =========================================================
# Use Slang's built-in automatic differentiation to compute gradients
# of the loss function w.r.t. material parameters.
#
# Run: python src/step_5_1_ad.py

import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 5.1: Automatic Differentiation", width=1536, height=512)
module = spy.Module.load_from_file(app.device, "step_5_1_ad.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

# Trainable parameters
train_albedo = spy.Tensor.from_numpy(app.device, albedo_map.to_numpy().copy())
train_normal = spy.Tensor.from_numpy(app.device, np.tile(np.array([0, 0, 1], dtype=np.float32),
    (albedo_map.shape[0], albedo_map.shape[1], 1)))
train_roughness = spy.Tensor.from_numpy(app.device, np.ones(roughness_map.shape, dtype=np.float32) * 0.5)

# Gradient tensors (must be initialized)
albedo_grad = spy.Tensor.zeros_like(albedo_map)
normal_grad = spy.Tensor.zeros_like(normal_map)
roughness_grad = spy.Tensor.zeros_like(roughness_map)

light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
view_dir = spy.float3(0, 0, 1)

print("Running. Press ESC to exit.")
while app.process_events():
    # Reference: render with original material
    ref_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=ref_output)

    # Zero out gradients before backward pass
    albedo_grad.zero_(); normal_grad.zero_(); roughness_grad.zero_()

    # Compute gradients via Slang autodiff
    module.calculate_grad(pixel=spy.call_id(), reference=ref_output,
                          material={"albedo": train_albedo, "normal": train_normal,
                                    "roughness": train_roughness,
                                    "albedo_grad": albedo_grad, "normal_grad": normal_grad,
                                    "roughness_grad": roughness_grad},
                          light_dir=light_dir, view_dir=view_dir)

    # Report gradient stats
    ag = albedo_grad.to_numpy()
    ng = normal_grad.to_numpy()
    rg = roughness_grad.to_numpy()
    print(f"  |grad_albedo|={float(np.abs(ag).mean()):.6f} "
          f"|grad_normal|={float(np.abs(ng).mean()):.6f} "
          f"|grad_rough|={float(np.abs(rg).mean()):.6f}", end="\r")

    app.blit(ref_output, size=spy.int2(512, 1024), offset=spy.int2(0, 0))
    app.blit(train_albedo, size=spy.int2(512, 1024), offset=spy.int2(512, 0))
    app.blit(train_roughness, size=spy.int2(512, 1024), offset=spy.int2(1024, 0))
    app.present()
