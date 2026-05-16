// Phase 5.3: Full Training Loop — Equivalent to step_05_train
// =============================================================
// Step 5 最终形态。完整的材质纹理优化训练管线:
//   - 可微材质 (Phase 5.1)
//   - SSAA 参考采样
//   - Adam 优化器 (Phase 5.2)
//   - 随机光照/视角 (LCG)
// 功能等价于参考代码 step_05_train.slang。
//
// 包含: MaterialParameters (with autodiff) + render() + loss()
//       + LCG RNG + calculate_grads() + adam_step/optimizer_step
//       + init functions + downsample

import slangpy;
import brdf;

struct MaterialParameters
{
    RWTensor<float3, 2> albedo;
    RWTensor<float3, 2> normal;
    RWTensor<float, 2> roughness;

    RWTensor<float3, 2> albedo_grad;
    RWTensor<float3, 2> normal_grad;
    RWTensor<float, 2> roughness_grad;

    [Differentiable]
    float3 get_albedo(int2 p)
    {
        return albedo.getv(p);
    }
    [BackwardDerivativeOf(get_albedo)]
    void get_albedo_bwd(int2 p, float3 grad)
    {
        albedo_grad.setv(p, grad);
    }

    [Differentiable]
    float3 get_normal(int2 p)
    {
        return normal.getv(p);
    }
    [BackwardDerivativeOf(get_normal)]
    void get_normal_bwd(int2 p, float3 grad)
    {
        normal_grad.setv(p, grad);
    }

    [Differentiable]
    float get_roughness(int2 p)
    {
        return roughness.getv(p);
    }
    [BackwardDerivativeOf(get_roughness)]
    void get_roughness_bwd(int2 p, float grad)
    {
        roughness_grad.setv(p, grad);
    }
};

[Differentiable]
float3 render(int2 pixel, no_diff MaterialParameters mat, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float light_intensity = 1.5;
    float3 brdf_eval = eval_brdf(
        mat.get_albedo(pixel), normalize(light_dir),
        normalize(view_dir), normalize(mat.get_normal(pixel)),
        mat.get_roughness(pixel), 0.0, 1.0);
    return brdf_eval * light_intensity;
}

[Differentiable]
float3 loss(int2 pixel, no_diff float3 reference, no_diff MaterialParameters mat,
            no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render(pixel, mat, light_dir, view_dir);
    float3 error = color - reference;
    return error * error;
}

// LCG random number generator
#define PI 3.14159265358979323846f
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

    [mutating]
    float3 next_dir_on_hemisphere()
    {
        float phi = 2.0f * PI * next_float();
        float cos_theta = next_float();
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    }
};

// Compute gradients with SSAA: 4x4 sub-pixel samples from reference
void calculate_grads(uint seed, int2 pixel, MaterialParameters material, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);
    float3 light_dir = lcg.next_dir_on_hemisphere();
    float3 view_dir = lcg.next_dir_on_hemisphere();

    int2 hi_res_pixel = pixel * 4;
    float3 sum = 0;
    for (int x0 = 0; x0 < 4; x0++)
    {
        for (int y0 = 0; y0 < 4; y0++)
        {
            int2 hi_res_pixel_offset = hi_res_pixel + int2(x0, y0);
            sum += render(hi_res_pixel_offset, ref_material, light_dir, view_dir);
        }
    }
    sum /= 16.0f;

    bwd_diff(loss)(pixel, sum, material, light_dir, view_dir, 1);
}

// Parameter initialization
void init_normal(inout float3 val)
{
    val = float3(0, 0, 1);
}
void init3(inout float3 val, float3 v)
{
    val = v;
}
void init1(inout float val, float v)
{
    val = v;
}

// Adam optimizer
void adam_step(inout float primal, float grad, inout float mean, inout float variance,
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
}

void optimizer_step1(inout float primal, inout float grad, inout float mean, inout float variance,
                     float learning_rate, int iteration)
{
    adam_step(primal, grad, mean, variance, learning_rate, iteration);
    primal = saturate(primal);
}

void optimizer_step3(inout float3 primal, float3 grad, inout float3 mean, inout float3 variance,
                     float learning_rate, int iteration, bool renormalize)
{
    for (int i = 0; i < 3; ++i)
        adam_step(primal[i], grad[i], mean[i], variance[i], learning_rate, iteration);

    if (renormalize)
        primal = normalize(primal);
    else
        primal = saturate(primal);
}

float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}

float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
