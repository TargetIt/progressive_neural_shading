# Phase 6.0: Matrix Multiply in Shader
# =====================================
# Demonstrates a simple linear layer on the GPU:
#   output = W * input  (no bias, no activation)
#
# Each pixel computes W * uv, where uv is the normalized pixel coordinate.
# W is a 3x2 weight matrix passed as a Tensor.
#
# Run: python src/step_6_0_matmul.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512, height=512, title="Phase 6.0: Matrix Multiply")
module = spy.Module.load_from_file(app.device, "step_6_0_matmul.slang")

# Create a weight matrix
# W is 3x2: maps 2D input (uv) to 3D output (color)
weights_data = np.array([
    [1.0, 0.0],  # R channel depends only on u
    [0.0, 1.0],  # G channel depends only on v
    [0.5, 0.5],  # B channel depends on both
], dtype=np.float32)
weights = spy.Tensor.from_numpy(app.device, weights_data)

resolution = spy.int2(512, 512)
output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(pixel=spy.call_id(), resolution=resolution, weights=weights, _result=output)
    app.blit(output, tonemap=False)
    app.present()
