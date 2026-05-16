// Phase 6.1: Neuron Complete — Bias + ReLU Activation
// ====================================================
// Builds on Phase 6.0 by adding:
//   - Bias term: sum = W*x + b
//   - ReLU activation: output = max(0, sum)
//
// Each pixel computes: output = max(0, W * uv + b)
// where W is a 3x2 weight matrix and b is a 3-element bias vector.

import slangpy;

[Differentiable]
float relu(float x)
{
    return max(x, 0.0f);
}

// Full neuron: W*x + b with ReLU activation
[Differentiable]
float3 neuron(Tensor<float, 2> weights, Tensor<float, 1> biases, float2 input)
{
    float3 output;
    for (int row = 0; row < 3; ++row)
    {
        float sum = biases.get({row});
        for (int col = 0; col < 2; ++col)
            sum += weights.get({row, col}) * input[col];
        output[row] = relu(sum);
    }
    return output;
}

// Main render entry point
float3 render(int2 pixel, int2 resolution, Tensor<float, 2> weights, Tensor<float, 1> biases)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return neuron(weights, biases, uv);
}
