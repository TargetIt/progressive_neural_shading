// Phase 1.1: Albedo BRDF — Lighting with Hardcoded Color
// =========================================================
// 从 Phase 1.0 (纯红色) 演进: 加入 BRDF 光照模型。
// 区别: 不使用纹理, albedo 是硬编码的蓝色。
//
// 新增概念:
//   - BRDF (Bidirectional Reflectance Distribution Function): 双向反射分布函数
//     描述光线在材质表面的反射行为
//   - light_dir: 光源方向
//   - view_dir: 观察方向
//   - normal: 表面法线 (假设平面, 统一指向上方)
//   - float3 normalize(v): 向量归一化

import slangpy;

// BRDF 评估函数
// 使用简单的 diffuse (Lambertian) + specular 模型
float3 eval_brdf(
    float3 albedo,      // 材质颜色 (反射率)
    float3 light_dir,   // 光源方向 (指向光源)
    float3 view_dir,    // 观察方向 (指向相机)
    float3 normal,      // 表面法线
    float roughness     // 粗糙度 (0=镜面, 1=粗糙)
)
{
    // Lambertian diffuse: 漫反射分量
    // N·L = 法线与光源方向的夹角余弦, max(0,.) 避免负值
    float NdotL = max(0.0, dot(normal, light_dir));
    float3 diffuse = albedo * NdotL;

    // 简单的 specular highlight (Blinn-Phong 风格)
    float3 half_vec = normalize(light_dir + view_dir);
    float NdotH = max(0.0, dot(normal, half_vec));
    float specular = pow(NdotH, 1.0 / max(roughness, 0.01));

    // 组合漫反射和高光
    return diffuse + specular * 0.5;
}

// 每个像素执行一次
float3 render(int2 pixel, float3 light_dir, float3 view_dir)
{
    // 硬编码材质参数 (无纹理)
    float3 albedo = float3(0.2, 0.4, 1.0);  // 蓝色
    float roughness = 0.3;                    // 较光滑
    float3 normal = float3(0.0, 0.0, 1.0);   // 平面法线 (指向相机)

    // 光源强度
    float light_intensity = 1.5;

    // 计算 BRDF 光照
    float3 brdf = eval_brdf(
        albedo,
        normalize(light_dir),
        normalize(view_dir),
        normal,
        roughness
    );

    return brdf * light_intensity;
}
