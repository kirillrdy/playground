#ifndef SAM_MACOS_BRIDGE_H
#define SAM_MACOS_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void (*on_open_file)(const char *path);
    void (*on_open_video)(const char *path);
    void (*on_video_play_pause)(void);
    void (*on_video_seek)(double seconds);
    void (*on_video_step)(void);
    void (*on_precache_video)(const char *text);
    void (*on_sample_click)(void);
    void (*on_mode_change)(int mode); // 1 = add, 0 = cut
    void (*on_clear_points)(void);
    void (*on_find_text)(const char *text);
    void (*on_cancel_query)(void);
    void (*on_canvas_click)(float norm_x, float norm_y, int is_positive);
    void (*on_select_mask)(int mask_index);
} SamCallbacks;

typedef struct {
    float score;
    float coverage;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
} SamMaskInfo;

typedef struct {
    uint8_t *rgb;
    int width;
    int height;
    double pts_seconds;
} SamVideoFrame;

int sam_macos_init(const SamCallbacks *callbacks);
void sam_macos_run(void);
void sam_macos_set_window_title(const char *title);
void sam_macos_set_status(const char *text);
void sam_macos_set_image(const uint8_t *rgba_pixels, int width, int height);
void sam_macos_set_masks(int count, const SamMaskInfo *masks, int best_index, int selected_index);
void sam_macos_set_busy(int is_busy);
void sam_macos_set_video_mode(int active, int playing);
void sam_macos_set_video_timeline(double duration, double position);
void sam_macos_set_precache_progress(int state, double fraction, size_t frames);
void sam_macos_set_query_active(int active);
void *sam_macos_video_open(const char *path, double start_seconds);
double sam_macos_video_duration(void *reader);
int sam_macos_video_next(void *reader, SamVideoFrame *frame);
void sam_macos_video_free_frame(SamVideoFrame *frame);
void sam_macos_video_close(void *reader);
int sam_macos_file_exists(const char *path);
const char *sam_macos_get_home(void);
void sam_macos_dispatch_main(void (*fn)(void *ctx), void *ctx);

#ifdef __cplusplus
}
#endif

#endif // SAM_MACOS_BRIDGE_H
