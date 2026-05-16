// Phase 1.0: Hello Slang — Minimal Shader
// ==========================================
// 最简单的 Slang 着色器: 每个像素返回纯红色。
// 理解: .slang 文件结构, import, 函数签名, GPU 并行执行模型
//
// GPU 并行模型:
//   render() 函数对每个像素并行执行一次
//   pixel 参数自动从 (0,0) 遍历到 (width-1, height-1)
//
// 概念:
//   - float3 = 3 分量浮点向量 (RGB)
//   - int2 = 2 分量整数向量 (像素坐标 x, y)
//   - slangpy 负责: 编译 .slang → GPU 可执行代码, 分配线程网格

import slangpy;

// 每个像素执行一次, 返回该像素的颜色
// pixel: 当前像素坐标 (x, y)
// _result: 输出 Tensor, 由 slangpy 自动分配
float3 render(int2 pixel)
{
    // 返回纯红色 (R=1, G=0, B=0)
    return float3(1.0, 0.0, 0.0);
}
