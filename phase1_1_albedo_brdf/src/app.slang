// Phase 1.0: Minimal blit helper
// 最简单的 blit: 把 Tensor 拷贝到屏幕输出纹理
import slangpy;

void blit(
    int2 pixel,
    int2 size,
    int2 offset,
    Tensor<float3, 2> input,
    RWTexture2D<float4> output,
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    uint[2] shape = input.shape;
    float2 uv = (float2(pixel) + 0.5f) / float2(size);
    float3 col = input.getv(uint2(float2(shape[1], shape[0]) * uv));

    output[offset + pixel] = float4(col, 1.0f);
}
