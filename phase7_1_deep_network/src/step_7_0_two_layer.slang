// Phase 7.0: Two-Layer Forward
// =============================
// Stacks two linear layers with ReLU activation:
//   layer0: 2 inputs -> 16 hidden
//   layer1: 16 inputs -> 3 outputs (RGB)
//
// No training, just forward pass to show multi-layer architecture.

import slangpy;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;

    [Differentiable]
    float get_bias(int neuron)
    {
        return biases.get({neuron});
    }
    [Differentiable]
    float get_weight(int neuron, int input)
    {
        return weights.get({neuron, input});
    }

    // Evaluate all neurons in the layer
    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row)
        {
            var sum = get_bias(row);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum += get_weight(row, col) * x[col];
            y[row] = sum;
        }
        return y;
    }
}

[Differentiable]
float activation(float x)
{
    return max(x, 0.0f);
}

struct Network {
    NetworkParameters<2, 16> layer0;
    NetworkParameters<16, 3> layer1;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float inputs[2] = {uv.x, uv.y};
        float hidden[16] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            hidden[i] = activation(hidden[i]);
        float output[3] = layer1.forward(hidden);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output[i] = activation(output[i]);
        return float3(output[0], output[1], output[2]);
    }
}

float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}
