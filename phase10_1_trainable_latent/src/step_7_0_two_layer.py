# Phase 7.0: Two-Layer Forward
# =============================
# Stacks two linear layers with ReLU activation:
#   layer0: 2 inputs -> 16 hidden
#   layer1: 16 inputs -> 3 outputs (RGB)
#
# Run: python src/step_7_0_two_layer.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512, height=512, title="Phase 7.0: Two-Layer Forward")
module = spy.Module.load_from_file(app.device, "step_7_0_two_layer.slang")


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
        self.layer0 = NetworkParameters(2, 16)
        self.layer1 = NetworkParameters(16, 3)


network = Network()
resolution = spy.int2(512, 512)
output = spy.Tensor.empty(app.device, shape=(512, 512), dtype=spy.float3)

print("Running. Press ESC to exit.")
while app.process_events():
    module.render(pixel=spy.call_id(), resolution=resolution, network=network, _result=output)
    app.blit(output, tonemap=False)
    app.present()
