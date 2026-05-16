// Phase 4.0: Per-Pixel Difference
// =================================
// Compute per-pixel absolute difference between two BRDF renders.
// Output is grayscale abs error: mean(|color - reference|).
//
// Contains: render() + abs_diff() + downsample

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

// Absolute per-pixel difference (as grayscale float3)
float3 abs_diff(int2 pixel, float3 reference, MaterialParameters mat, float3 light_dir, float3 view_dir)
{
    float3 color = render(pixel, mat, light_dir, view_dir);
    float3 err = abs(color - reference);
    float gray = (err.x + err.y + err.z) / 3.0;
    return float3(gray, gray, gray);
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
