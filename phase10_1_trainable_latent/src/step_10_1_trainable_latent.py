# Phase 10.1: Trainable Latent
# =============================
# Optimizes latent codes via gradient descent.
# Network weights are fixed; only the latent texture is trained.
#
# Run: python src/step_10_1_trainable_latent.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512 * 3 + 10 * 2, height=512, title="Phase 10.1: Trainable Latent")
module = spy.Module.load_from_file(app.device, "step_10_1_trainable_latent.slang")

data_path = Path(__file__).parent
image = spy.Tensor.load_from_image(app.device, data_path.joinpath("slangstars.png"), linearize=True)


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs
        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32")
        )


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int):
        super().__init__(module["LatentTexture"])
        self.width = width
        self.height = height
        initial = np.random.uniform(0.0, 1.0, (height, width, 4)).astype("float32")
        self.texture = spy.Tensor.from_numpy(app.device, initial)
        self.texture_grads = spy.Tensor.zeros_like(self.texture)
        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.texture, self.texture_grads, self.m_texture, self.v_texture,
            learning_rate, optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(32, 32)
        self.layer0 = NetworkParameters(4, 16)
        self.layer1 = NetworkParameters(16, 3)


network = Network()
optimize_counter = 0

print("Compiling shaders... this may take a while")

while app.process_events():
    offset = 0
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10

    res = spy.int2(256, 256)
    batch_size = (64, 64)

    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=lr_output)
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10

    loss_output = spy.Tensor.empty_like(image)
    module.loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
    )
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)

    learning_rate = 0.01

    for i in range(20):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=network,
        )
        optimize_counter += 1
        network.latent_texture.optimize(learning_rate, optimize_counter)

    print(f"Loss: {np.mean(loss_output.to_numpy()):.5f}")
    app.present()
