# Phase 4.1: Squared Error (MSE)
# ===============================
# Compute per-pixel squared error and overall MSE between two renders.
# Displays: reference (left), prediction (center), squared error (right)
#
# Run: python src/step_4_1_sqerror.py

import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 4.1: Squared Error", width=1536, height=512)
module = spy.Module.load_from_file(app.device, "step_4_1_sqerror.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
view_dir = spy.float3(0, 0, 1)

print("Running. Press ESC to exit.")
while app.process_events():
    # Reference render
    ref_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=ref_output)

    # Prediction: different light angle
    pred_light = spy.math.normalize(spy.float3(0.5, 0.1, 0.8))
    pred_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=pred_light, view_dir=view_dir, _result=pred_output)

    # Squared error (uses reference + prediction material with reference light)
    sq_output = spy.Tensor.empty_like(albedo_map)
    module.sq_error(pixel=spy.call_id(), reference=ref_output,
                    material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                    light_dir=light_dir, view_dir=view_dir, _result=sq_output)

    # Compute MSE from the squared error output
    mse = float(np.mean(sq_output.to_numpy()))
    print(f"  MSE: {mse:.6f}", end="\r")

    panel_w = 512
    app.blit(ref_output, size=spy.int2(panel_w, 512), offset=spy.int2(0, 0))
    app.blit(pred_output, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w, 0))
    app.blit(sq_output, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w * 2, 0), tonemap=False)
    app.present()
