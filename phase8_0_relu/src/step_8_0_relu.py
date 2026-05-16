# Phase 8.0: ReLU Activation
# ===========================
# Compares ReLU vs no activation side by side.
# Left: linear output (can be negative)
# Right: ReLU output (negative values clipped to 0)
#
# Run: python src/step_8_0_relu.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=1024, height=512, title="Phase 8.0: ReLU Activation")
module = spy.Module.load_from_file(app.device, "step_8_0_relu.slang")

# Create weights and biases that produce negative values
weights_data = np.array([
    [1.0, 0.0],
    [0.0, -1.0],
    [1.0, -1.0],
], dtype=np.float32)
weights = spy.Tensor.from_numpy(app.device, weights_data)

# Biases that push some outputs negative
biases_data = np.array([-0.2, 0.3, -0.5], dtype=np.float32)
biases = spy.Tensor.from_numpy(app.device, biases_data)

resolution = spy.int2(512, 512)

# Output tensors for no-activation and ReLU panels
output_no_act = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)
output_with_relu = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    # Left panel: no activation
    module.render_no_act(pixel=spy.call_id(), resolution=resolution,
                         weights=weights, biases=biases, _result=output_no_act)
    app.blit(output_no_act, size=spy.int2(512, 512), offset=spy.int2(0, 0), tonemap=False)

    # Right panel: with ReLU
    module.render_with_relu(pixel=spy.call_id(), resolution=resolution,
                            weights=weights, biases=biases, _result=output_with_relu)
    app.blit(output_with_relu, size=spy.int2(512, 512), offset=spy.int2(512, 0), tonemap=False)

    app.present()
