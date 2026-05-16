// Phase 9.0: Sin Encoding
// ========================
// Demonstrates single-frequency sin positional encoding.
// Encodes UV coordinates using sin at a single frequency.
// Outputs the encoded value directly as RGB color -> range [-1, 1].

import slangpy;

// Single-frequency sin encoding
// Encodes a 2D coordinate into a 3-channel output
float3 sin_encode(float2 uv, float frequency)
{
    float scale = frequency * 2.0f * 3.14159265358979323846f;
    float3 encoded;
    encoded[0] = sin(uv.x * scale);
    encoded[1] = sin(uv.y * scale);
    encoded[2] = sin((uv.x + uv.y) * scale * 0.5f);
    return encoded;
}

float3 render(int2 pixel, int2 resolution, float frequency)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return sin_encode(uv, frequency);
}
