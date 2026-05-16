# Phase 4.0: Per-Pixel Absolute Difference
# =========================================
# Compute per-pixel absolute difference between two BRDF renders.
# Displays: reference (left), prediction (center), abs error grayscale (right)
#
# Run: python src/step_4_0_diff.py

import numpy as np
from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 4.0: Per-Pixel Abs Diff", width=1536, height=512)
module = spy.Module.load_from_file(app.device, "step_4_0_diff.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
view_dir = spy.float3(0, 0, 1)

print("Running. Press ESC to exit.")
while app.process_events():
    # Reference: render with original roughness
    ref_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=light_dir, view_dir=view_dir, _result=ref_output)

    # Prediction: render with modified roughness (smoother)
    pred_output = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map,
                            "roughness": spy.Tensor.from_numpy(app.device,
                                0.5 * np.ones(roughness_map.shape, dtype=np.float32))},
                  light_dir=light_dir, view_dir=view_dir, _result=pred_output)

    # Diff
    diff_output = spy.Tensor.empty_like(albedo_map)
    module.abs_diff(pixel=spy.call_id(), reference=ref_output,
                    material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                    light_dir=light_dir, view_dir=view_dir, _result=diff_output)

    # Three-panel display
    panel_w = 512
    app.blit(ref_output, size=spy.int2(panel_w, 512), offset=spy.int2(0, 0))
    app.blit(pred_output, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w, 0))
    app.blit(diff_output, size=spy.int2(panel_w, 512), offset=spy.int2(panel_w * 2, 0), tonemap=False)
    app.present()
