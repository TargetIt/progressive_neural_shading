// Phase 1.2: Full BRDF with Textures — Equivalent to step_01_basicprogram
// ==========================================================================
// 从 Phase 1.1 (硬编码材质) 演进: 加入纹理采样。
// 这是 Step 1 的最终形态, 功能等价于参考代码的 step_01_basicprogram。
//
// 新增概念:
//   - Tensor<T, N>: GPU 纹理 (2D 数组)
//   - Texture sampling: .getv(pixel) 读取像素位置的纹理值
//   - MaterialParameters struct: 封装材质纹理
//   - Disney BRDF: 完整的基于物理的 BRDF 模型 (import brdf)
//
// 对比 Phase 1.1:
//   - 1.1: albedo = float3(0.2, 0.4, 1.0)   ← 硬编码
//   - 1.2: albedo = albedo_map.getv(pixel)    ← 纹理采样

import slangpy;
import brdf;

// 材质参数结构体: 封装所有纹理
struct MaterialParameters
{
    Tensor<float3, 2> albedo;     // 反照率纹理 (RGB)
    Tensor<float3, 2> normal;     // 法线贴图 (RGB, 需要归一化)
    Tensor<float, 2> roughness;   // 粗糙度纹理 (单通道)

    // 采样反照率
    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }

    // 采样并归一化法线
    float3 get_normal(int2 pixel)
    {
        return normalize(normal.getv(pixel));
    }

    // 采样粗糙度
    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }
};

// 渲染完整分辨率的 BRDF 光照
// pixel: 当前像素坐标 (由 slangpy 自动分配)
// material: 材质纹理
// light_dir: 光源方向
// view_dir: 观察方向
float3 render(int2 pixel, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float light_intensity = 1.5;

    // 计算 Disney BRDF
    float3 brdf_eval = eval_brdf(
        material.get_albedo(pixel),     // 从纹理采样反照率
        normalize(light_dir),           // 光源方向
        normalize(view_dir),            // 观察方向
        material.get_normal(pixel),     // 法线 (归一化)
        material.get_roughness(pixel),  // 粗糙度
        0.0,                            // metallic (非金属)
        1.0                             // specular
    );

    return brdf_eval * light_intensity;
}
