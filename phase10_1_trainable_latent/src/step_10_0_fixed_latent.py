# Phase 10.0: Fixed Latent
# =========================
# Demonstrates a fixed latent texture with neural decoding.
# A coarse 4x4x4 latent grid is bilinearly sampled and decoded to RGB.
#
# Run: python src/step_10_0_fixed_latent.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512, height=512, title="Phase 10.0: Fixed Latent")
module = spy.Module.load_from_file(app.device, "step_10_0_fixed_latent.slang")


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs
        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-1.0, 1.0, (outputs, inputs)).astype("float32")
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(4, 16)
        self.layer1 = NetworkParameters(16, 3)

        # Create fixed latent texture (4x4 grid, 4 channels)
        # Each cell stores a random latent code
        latent_data = np.random.uniform(0.0, 1.0, (4, 4, 4)).astype(np.float32)
        self.latent_texture = spy.Tensor.from_numpy(app.device, latent_data)


network = Network()
resolution = spy.int2(512, 512)
output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(pixel=spy.call_id(), resolution=resolution, network=network, _result=output)
    app.blit(output, tonemap=False)
    app.present()
