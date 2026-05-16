// Phase 6.0: Matrix Multiply in Shader
// =====================================
// Demonstrates a simple linear layer: output = W * input
// (no bias, no activation)
//
// Each pixel computes W * uv where W is a hardcoded 3x2 matrix
// and uv is the normalized pixel coordinate.

import slangpy;

// Perform matrix-vector multiply: W (3x2) * input (2) -> output (3)
[Differentiable]
float3 linear_layer(Tensor<float, 2> weights, float2 input)
{
    float3 output;
    for (int row = 0; row < 3; ++row)
    {
        float sum = 0.0f;
        for (int col = 0; col < 2; ++col)
            sum += weights.get({row, col}) * input[col];
        output[row] = sum;
    }
    return output;
}

// Main render entry point
float3 render(int2 pixel, int2 resolution, Tensor<float, 2> weights)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return linear_layer(weights, uv);
}

// Helper to compute matmul on the CPU-equivalent for verification
// Computes output = W * input for a single (row,col) position
float single_matmul_element(Tensor<float, 2> weights, float2 input, int row)
{
    float sum = 0.0f;
    for (int col = 0; col < 2; ++col)
        sum += weights.get({row, col}) * input[col];
    return sum;
}
