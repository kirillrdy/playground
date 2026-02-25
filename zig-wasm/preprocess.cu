#include "preprocess.h"

#include <cuda_runtime.h>

__device__ inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__global__ void nv12_resize_to_rgb_nchw_kernel(
    const unsigned char *src_y,
    const unsigned char *src_uv,
    int src_pitch_y,
    int src_pitch_uv,
    float *dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) return;

    // Nearest-neighbor sampling from source frame to model input size.
    const int src_x = (x * src_width) / dst_width;
    const int src_row = (y * src_height) / dst_height;

    const int y_idx = src_row * src_pitch_y + src_x;
    const int uv_x = (src_x / 2) * 2;
    const int uv_y = src_row / 2;
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

    const int plane = dst_width * dst_height;
    const int idx = y * dst_width + x;
    dst[idx] = r;
    dst[plane + idx] = g;
    dst[(2 * plane) + idx] = b;
}

extern "C" void launch_nv12_resize_to_rgb_nchw(
    const unsigned char *src_y,
    const unsigned char *src_uv,
    int src_pitch_y,
    int src_pitch_uv,
    float *dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    void *stream
) {
    const dim3 block(16, 16);
    const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    nv12_resize_to_rgb_nchw_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        src_y, src_uv, src_pitch_y, src_pitch_uv, dst, src_width, src_height, dst_width, dst_height
    );
}
