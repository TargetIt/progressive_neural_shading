// Phase 2.1: GPU Downsampling
// ==============================
// 从 Phase 2.0 (Python 端降采样) 演进: 在 GPU 上并行降采样。
//
// 新增概念:
//   - GPU 并行降采样: 每个输出像素取 4 个输入像素的平均
//   - 在 Slang 中操作 Tensor (GPU 并行)
//   - 对比 Python vs GPU: GPU 速度快 100-1000x
//
// 对比 Phase 2.0:
//   - 2.0: np.mean(reshaped) on CPU
//   - 2.1: downsample3() on GPU (并行)

import slangpy;
import brdf;

struct MaterialParameters
{
    Tensor<float3, 2> albedo;
    Tensor<float3, 2> normal;
    Tensor<float, 2> roughness;
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

// GPU 降采样: 对 float3 的 3 通道纹理做 2x2 box filter
float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;  // 平均值 = 和 / 4
}

// GPU 降采样: 对单通道纹理
float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
