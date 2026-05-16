# Phase 6.1: Neuron Complete — Bias + ReLU Activation
# ====================================================
# Builds on Phase 6.0 by adding bias and ReLU activation.
# Each pixel: output = max(0, W * uv + b)
#
# Run: python src/step_6_1_neuron.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512, height=512, title="Phase 6.1: Neuron Complete")
module = spy.Module.load_from_file(app.device, "step_6_1_neuron.slang")

# Weight matrix: 3x2
weights_data = np.array([
    [1.0, 0.0],  # R depends on u
    [0.0, 1.0],  # G depends on v
    [1.0, 1.0],  # B depends on both
], dtype=np.float32)
weights = spy.Tensor.from_numpy(app.device, weights_data)

# Bias vector: 3 elements
biases_data = np.array([-0.3, -0.3, 0.0], dtype=np.float32)
biases = spy.Tensor.from_numpy(app.device, biases_data)

resolution = spy.int2(512, 512)
output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(pixel=spy.call_id(), resolution=resolution,
                  weights=weights, biases=biases, _result=output)
    app.blit(output, tonemap=False)
    app.present()
