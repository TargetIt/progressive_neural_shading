// Phase 8.0: ReLU Activation
// ===========================
// Demonstrates ReLU activation vs no activation.
// Renders two versions of the same single-layer network:
//   - Without activation: output = W * uv + b (can be negative)
//   - With ReLU: output = max(0, W * uv + b) (clips negatives to 0)

import slangpy;

[Differentiable]
float relu(float x)
{
    return max(x, 0.0f);
}

// Single-layer network without activation (linear only)
[Differentiable]
float3 eval_no_act(Tensor<float, 2> weights, Tensor<float, 1> biases, float2 uv)
{
    float3 output;
    for (int row = 0; row < 3; ++row)
    {
        float sum = biases.get({row});
        for (int col = 0; col < 2; ++col)
            sum += weights.get({row, col}) * uv[col];
        output[row] = sum;
    }
    return output;
}

// Single-layer network WITH ReLU activation
[Differentiable]
float3 eval_with_relu(Tensor<float, 2> weights, Tensor<float, 1> biases, float2 uv)
{
    float3 output;
    for (int row = 0; row < 3; ++row)
    {
        float sum = biases.get({row});
        for (int col = 0; col < 2; ++col)
            sum += weights.get({row, col}) * uv[col];
        output[row] = relu(sum);
    }
    return output;
}

// Render no-activation panel (left half of output)
float3 render_no_act(int2 pixel, int2 resolution, Tensor<float, 2> weights, Tensor<float, 1> biases)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return eval_no_act(weights, biases, uv);
}

// Render with ReLU panel (right half of output)
float3 render_with_relu(int2 pixel, int2 resolution, Tensor<float, 2> weights, Tensor<float, 1> biases)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return eval_with_relu(weights, biases, uv);
}
