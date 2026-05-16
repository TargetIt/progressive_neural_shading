// Phase 1.2: Blit helper with ACES tonemapping
import slangpy;

float3 tonemap_aces_film(float3 x)
{
    float a = 2.51; float b = 0.03; float c = 2.43;
    float d = 0.59; float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

void blit(
    int2 pixel, int2 size, int2 offset,
    bool tonemap,
    Tensor<float3, 2> input,
    RWTexture2D<float4> output,
)
{
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    uint[2] shape = input.shape;
    float2 uv = (float2(pixel) + 0.5f) / float2(size);
    float3 col = input.getv(uint2(float2(shape[1], shape[0]) * uv));
    col = abs(col);
    if (tonemap) col = tonemap_aces_film(col);
    if (any(isnan(col.x)))
        output[offset + pixel] = float4(1, 0, 1, 1);
    else
        output[offset + pixel] = float4(col, 1.0f);
}
