#ifndef PREPROCESS_H
#define PREPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_nv12_resize_to_rgb_nchw(
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
);

#ifdef __cplusplus
}
#endif

#endif
