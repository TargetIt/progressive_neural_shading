// Phase 5.1: Automatic Differentiation
// ======================================
// Introduce Slang's [Differentiable] + bwd_diff for automatic gradient computation.
// Material access functions have backward derivatives that write to gradient tensors.
//
// Contains: MaterialParameters (with autodiff) + render() + loss()
//           + calculate_grad() + downsample

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

// Compute gradients via bwd_diff
void calculate_grad(int2 pixel, float3 reference, MaterialParameters mat,
                    float3 light_dir, float3 view_dir)
{
    bwd_diff(loss)(pixel, reference, mat, light_dir, view_dir, 1);
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
