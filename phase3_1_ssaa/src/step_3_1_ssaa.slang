// Phase 3.1: SSAA Pipeline — Equivalent to step_03_supersample
// =============================================================
// Step 3 最终形态。Supersampling Anti-Aliasing 管线:
//   对每个输出像素, 在材质空间采样 4x4 个子像素并平均,
//   等价于高分辨率渲染后降采样 (SSAA)。
//
// 包含: render() + ssaa_render() + downsample3/downsample1
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

// SSAA render: for each output pixel, average 4x4 sub-pixel samples.
// This is equivalent to rendering at 4x resolution then downsampling 4x.
float3 ssaa_render(int2 pixel, MaterialParameters mat, float3 light_dir, float3 view_dir)
{
    int2 base = pixel * 4;
    float3 sum = 0.0;
    for (int dx = 0; dx < 4; dx++)
    {
        for (int dy = 0; dy < 4; dy++)
        {
            sum += render(base + int2(dx, dy), mat, light_dir, view_dir);
        }
    }
    return sum / 16.0;
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
