# Phase 9.0: Sin Encoding
# ========================
# Demonstrates single-frequency sin positional encoding.
# Maps UV coordinates through sin() to produce colorful patterns.
#
# Run: python src/step_9_0_sin_encoding.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512, height=512, title="Phase 9.0: Sin Encoding")
module = spy.Module.load_from_file(app.device, "step_9_0_sin_encoding.slang")

frequency = 4.0  # 4 periods across the texture
resolution = spy.int2(512, 512)
output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(pixel=spy.call_id(), resolution=resolution, frequency=frequency, _result=output)
    app.blit(output, tonemap=False)
    app.present()
