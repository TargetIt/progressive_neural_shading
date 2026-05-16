# Phase 3.0: Higher Resolution Render
# ====================================
# Demonstrates rendering BRDF at higher resolutions (2x, 4x).
# Renders at base resolution and at 2x, showing the size difference.
#
# Run: python src/step_3_0_hi_res.py

from app import App
import slangpy as spy
from pathlib import Path

app = App(title="Phase 3.0: Higher Resolution Render", width=1024, height=1024)
module = spy.Module.load_from_file(app.device, "step_3_0_hi_res.slang")

assets = Path(__file__).parent.parent / "assets"
albedo_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.diffuse.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(app.device, assets / "PavingStones070_2K.roughness.jpg", grayscale=True)

tex_h, tex_w = albedo_map.shape

print(f"Texture size: {tex_h}x{tex_w}")
print("Running. Press ESC to exit.")

while app.process_events():
    # Render at base resolution
    output_1x = spy.Tensor.empty_like(albedo_map)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output_1x)

    # Render at 2x resolution (supersampled)
    output_2x = spy.Tensor.empty(device=app.device,
        shape=(tex_h * 2, tex_w * 2), dtype=albedo_map.dtype)
    module.render(pixel=spy.call_id(),
                  material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
                  light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
                  view_dir=spy.float3(0, 0, 1), _result=output_2x)

    # Downsample 2x result back to 1x for comparison
    output_2x_down = spy.Tensor.empty_like(albedo_map)
    module.downsample3(spy.call_id(), output_2x, _result=output_2x_down)

    # Blit side by side: original | supersampled then downsampled
    app.blit(output_1x, size=spy.int2(512, 1024), offset=spy.int2(0, 0))
    app.blit(output_2x_down, size=spy.int2(512, 1024), offset=spy.int2(512, 0))
    app.present()
