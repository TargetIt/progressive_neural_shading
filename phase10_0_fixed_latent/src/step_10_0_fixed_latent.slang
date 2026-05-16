// Phase 10.0: Fixed Latent
// =========================
// Demonstrates a fixed (non-trainable) latent texture.
// A coarse grid of latent codes is bilinearly sampled at each pixel,
// then decoded to RGB through a small neural network.
//
// Latent grid: 4x4 with 4 channels each
// Network: 4 -> 16 -> 3

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

// Bilinear sample from latent texture
[Differentiable]
float4 sample_latent(Tensor<float, 3> latent, float2 uv)
{
    uint2 size = {latent.shape[1], latent.shape[0]};  // width, height
    float2 texel = uv * float2(size - 1);
    int2 c0 = clamp(int2(texel), int2(0, 0), int2(size) - 1);
    int2 c1 = min(c0 + 1, int2(size) - 1);
    float2 frac = saturate(texel - float2(c0));

    float4 result;
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float v00 = latent.get({c0.y, c0.x, i});
        float v10 = latent.get({c1.y, c0.x, i});
        float v01 = latent.get({c0.y, c1.x, i});
        float v11 = latent.get({c1.y, c1.x, i});
        result[i] = lerp(lerp(v00, v10, frac.x), lerp(v01, v11, frac.x), frac.y);
    }
    return result;
}

[Differentiable]
float leakyReLU(float x, float negativeSlope = 0.01f)
{
    if (x > 0.0f)
        return x;
    else
        return x * negativeSlope;
}

struct Network {
    NetworkParameters<4, 16> layer0;
    NetworkParameters<16, 3> layer1;

    Tensor<float, 3> latent_texture;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float4 latent = sample_latent(latent_texture, uv);
        float input_arr[4] = {latent.x, latent.y, latent.z, latent.w};

        float hidden[16] = layer0.forward(input_arr);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            hidden[i] = leakyReLU(hidden[i]);

        float output[3] = layer1.forward(hidden);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output[i] = exp(output[i]);

        return float3(output[0], output[1], output[2]);
    }
};

float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}
