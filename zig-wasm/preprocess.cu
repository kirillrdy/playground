#include "preprocess.h"

#include <cuda_runtime.h>

__device__ inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__global__ void nv12_to_rgb_nchw_kernel(
    const unsigned char *src_y,
    const unsigned char *src_uv,
    int src_pitch_y,
    int src_pitch_uv,
    float *dst,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int y_idx = y * src_pitch_y + x;
    const int uv_x = (x / 2) * 2;
    const int uv_y = y / 2;
    const int uv_idx = uv_y * src_pitch_uv + uv_x;

    const float Y = static_cast<float>(src_y[y_idx]);
    const float U = static_cast<float>(src_uv[uv_idx + 0]) - 128.0f;
    const float V = static_cast<float>(src_uv[uv_idx + 1]) - 128.0f;

    float r = (Y + 1.402f * V) / 255.0f;
    float g = (Y - 0.344136f * U - 0.714136f * V) / 255.0f;
    float b = (Y + 1.772f * U) / 255.0f;

    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);

    const int plane = width * height;
    const int idx = y * width + x;
    dst[idx] = r;
    dst[plane + idx] = g;
    dst[(2 * plane) + idx] = b;
}

extern "C" void launch_nv12_to_rgb_nchw(
    const unsigned char *src_y,
    const unsigned char *src_uv,
    int src_pitch_y,
    int src_pitch_uv,
    float *dst,
    int width,
    int height,
    void *stream
) {
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nv12_to_rgb_nchw_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        src_y, src_uv, src_pitch_y, src_pitch_uv, dst, width, height
    );
}
