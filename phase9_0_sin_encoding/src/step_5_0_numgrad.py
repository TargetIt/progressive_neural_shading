# Phase 5.0: Numerical Gradient via Finite Differences
# =====================================================
# Compute per-pixel numerical gradient of loss w.r.t. material parameters.
# Uses central finite differences: dL/dx ~ (L(x+h) - L(x-h)) / (2h).
#
# Demonstrates that the loss increases when we perturb away from the reference,
# and the gradient correctly points in the direction of steepest ascent.
#
# Run: python src/step_5_0_numgrad.py

import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 5.0: Numerical Gradient", width=1024, height=1024)
module = spy.Module.load_from_file(app.device, "step_5_0_numgrad.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
view_dir = spy.float3(0, 0, 1)
h = 0.01  # finite difference step

print("Running. Press ESC to exit.")
while app.process_events():
    # Reference: render with original material
    ref_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=ref_output)

    # Perturb roughness up: x + h
    rough_plus = spy.Tensor.from_numpy(app.device,
        np.clip(roughness_map.to_numpy() + h, 0.0, 1.0).astype(np.float32))
    loss_plus = spy.Tensor.empty_like(albedo_map)
    module.loss(pixel=spy.call_id(), reference=ref_output,
                material={"albedo": albedo_map, "normal": normal_map, "roughness": rough_plus},
                light_dir=light_dir, view_dir=view_dir, _result=loss_plus)

    # Perturb roughness down: x - h
    rough_minus = spy.Tensor.from_numpy(app.device,
        np.clip(roughness_map.to_numpy() - h, 0.0, 1.0).astype(np.float32))
    loss_minus = spy.Tensor.empty_like(albedo_map)
    module.loss(pixel=spy.call_id(), reference=ref_output,
                material={"albedo": albedo_map, "normal": normal_map, "roughness": rough_minus},
                light_dir=light_dir, view_dir=view_dir, _result=loss_minus)

    # Numerical gradient: (L(x+h) - L(x-h)) / (2h)
    loss_p_np = loss_plus.to_numpy()
    loss_m_np = loss_minus.to_numpy()
    grad_np = (loss_p_np - loss_m_np) / (2.0 * h)

    # Take mean across channels for display as grayscale
    grad_gray = np.mean(grad_np, axis=2)
    grad_output = spy.Tensor.from_numpy(app.device,
        np.stack([grad_gray, grad_gray, grad_gray], axis=2).astype(np.float32))

    # Report stats
    mse = float(np.mean((loss_p_np + loss_m_np) / 2.0))
    print(f"  MSE: {mse:.6f}  |grad|_mean: {float(np.abs(grad_np).mean()):.6f}", end="\r")

    app.blit(ref_output, size=spy.int2(512, 1024), offset=spy.int2(0, 0))
    app.blit(grad_output, size=spy.int2(512, 1024), offset=spy.int2(512, 0), tonemap=False)
    app.present()
