# Phase 8.1: Activation Gallery (= network step_03)
# ===================================================
# Compares ReLU, LeakyReLU, tanh, and exp activations.
# Uses leakyReLU for hidden layers and exp for output.
#
# Run: python src/step_8_1_activation_gallery.py

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(width=512 * 3 + 10 * 2, height=512, title="Phase 8.1: Activation Gallery")
module = spy.Module.load_from_file(app.device, "step_8_1_activation_gallery.slang")

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

        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.biases, self.biases_grad, self.m_biases, self.v_biases,
            learning_rate, optimize_counter,
        )
        module.optimizer_step(
            self.weights, self.weights_grad, self.m_weights, self.v_weights,
            learning_rate, optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(2, 32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


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

    learning_rate = 0.001

    for i in range(20):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=network,
        )
        optimize_counter += 1
        network.optimize(learning_rate, optimize_counter)

    print(f"Loss: {np.mean(loss_output.to_numpy()):.5f}")
    app.present()
