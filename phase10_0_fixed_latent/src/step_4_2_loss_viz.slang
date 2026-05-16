// Phase 4.2: Loss Visualization — Equivalent to step_04_loss
// ============================================================
// Step 4 最终形态。计算并可视化 reference vs prediction 的平方误差。
// 功能等价于参考代码 step_04_loss.slang。
//
// 包含: render() + loss() + downsample3/downsample1
//       MaterialParameters

import slangpy;
import brdf;

struct MaterialParameters
{
    Tensor<float3, 2> albedo; Tensor<float3, 2> normal; Tensor<float, 2> roughness;
    float3 get_albedo(int2 p) { return albedo.getv(p); }
    float3 get_normal(int2 p) { return normalize(normal.getv(p)); }
    float get_roughness(int2 p) { return roughness.getv(p); }
};

float3 render(int2 pixel, MaterialParameters mat, float3 light_dir, float3 view_dir)
{
    float3 brdf = eval_brdf(mat.get_albedo(pixel), normalize(light_dir),
        normalize(view_dir), mat.get_normal(pixel), mat.get_roughness(pixel), 0.0, 1.0);
    return brdf * 1.5;
}

float3 loss(int2 pixel, float3 reference, MaterialParameters mat, float3 light_dir, float3 view_dir)
{
    float3 color = render(pixel, mat, light_dir, view_dir);
    float3 error = color - reference;
    return error * error;
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
