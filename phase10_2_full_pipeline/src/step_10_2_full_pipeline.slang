// Phase 10.2: Full Pipeline (= network step_05)
// ==============================================
// Neural texture compression with latent optimization.
// Full training loop optimizes both latent texture and network weights.
// Architecture: Latent(32x32x4) -> 4 -> 32 -> 32 -> 3 RGB

import slangpy;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;

    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

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

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad)
    {
        biases_grad.set({neuron}, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad)
    {
        weights_grad.set({neuron, input}, grad);
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

struct LatentTexture<int NumLatents> {
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> texture_grads;

    [BackwardDerivative(get_latent_bwd)]
    float get_latent(int y, int x, int latent_index)
    {
        return texture.get({y, x, latent_index});
    }

    void get_latent_bwd(int y, int x, int latent_index, float grad)
    {
        texture_grads.set({y, x, latent_index}, grad);
    }

    [Differentiable]
    float[NumLatents] sample(no_diff inout float2 uv)
    {
        int2 size = {texture.shape[1], texture.shape[0]};
        uv *= float2(size - 1);
        int2 c0 = clamp(int2(uv), int2(0, 0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);
        uv = saturate(uv - float2(c0));

        float result[NumLatents];
        [ForceUnroll]
        for (int i = 0; i < NumLatents; ++i) {
            float v00 = get_latent(c0.y, c0.x, i);
            float v10 = get_latent(c1.y, c0.x, i);
            float v01 = get_latent(c0.y, c1.x, i);
            float v11 = get_latent(c1.y, c1.x, i);
            result[i] = lerp(lerp(v00, v10, uv.x), lerp(v01, v11, uv.x), uv.y);
        }
        return result;
    }
};

[Differentiable]
float leakyReLU(float x, float negativeSlope = 0.01f)
{
    if (x > 0.0f)
        return x;
    else
        return x * negativeSlope;
}

struct Network {
    static const int NumLatents = 4;

    LatentTexture<NumLatents> latent_texture;
    NetworkParameters<NumLatents, 32> layer0;
    NetworkParameters<32, 32> layer1;
    NetworkParameters<32, 3> layer2;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float2 uv_copy = uv;
        float encoded_inputs[NumLatents] = latent_texture.sample(uv_copy);

        float output0[32] = layer0.forward(encoded_inputs);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output0[i] = leakyReLU(output0[i]);
        float output1[32] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output1[i] = leakyReLU(output1[i]);
        float output2[3] = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output2[i] = exp(output2[i]);
        return float3(output2[0], output2[1], output2[2]);
    }
};

[Differentiable]
float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float3 gamma_correct(float3 color)
{
    return pow(color, 1.0f/2.2f);
}

[Differentiable]
float3 loss(int2 pixel, int2 resolution, no_diff float3 reference, Network network)
{
    float3 color = render(pixel, resolution, network);
    float3 error = gamma_correct(color) - gamma_correct(reference);
    return error * error;
}

struct LCG
{
    uint state;
    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimizer_step(inout float primal, inout float grad, inout float mean, inout float variance,
                    float learning_rate, int iteration)
{
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    primal -= step;
    grad = 0.0f;
}

void calculate_grads(uint seed, int2 batch_index, int2 batch_size,
                     Tensor<float3, 2> reference, Network network)
{
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    int2 resolution = {reference.shape[1], reference.shape[0]};
    int2 pixel = int2(uv * float2(resolution));
    float3 reference_color = reference.getv(pixel);
    bwd_diff(loss)(pixel, resolution, reference_color, network, 1.0f);
}
