// Phase 6.2: MLP Training (= network step_01)
// ============================================
// Full neural network training loop:
//   - NetworkParameters with atomic gradient accumulation
//   - Adam optimizer
//   - LCG stochastic sampling
//   - 2 inputs (uv) -> 3 outputs (RGB)
//
// Trains a single-layer MLP to match slangstars.png.

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

struct Network {
    NetworkParameters<2, 3> layer;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float inputs[2] = {uv.x, uv.y};
        float output0[3] = layer.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output0[i] = activation(output0[i]);
        return float3(output0[0], output0[1], output0[2]);
    }
}

[Differentiable]
float activation(float x)
{
    return max(x, 0.0f);
}

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
