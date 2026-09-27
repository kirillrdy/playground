#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <spawn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct {
    unsigned char *rgb;
    int width;
    int height;
    double pts_seconds;
} SamVideoFrame;

typedef struct {
    int fd, width, height;
    pid_t pid, pts_pid;
    double duration, fps, start;
    size_t index;
    FILE *pts_file;
    double pts_buffer[64];
    size_t pts_len;
} SamVideoReader;

extern char **environ;

static pid_t launch(char *const argv[], int *read_fd) {
    int fds[2];
    if (pipe(fds)) return -1;
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions)) { close(fds[0]); close(fds[1]); return -1; }
    posix_spawn_file_actions_addclose(&actions, fds[0]);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    int result = posix_spawnp(&pid, argv[0], &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[1]);
    if (result) { close(fds[0]); return -1; }
    *read_fd = fds[0];
    return pid;
}

static char *capture(char *const argv[]) {
    int fd;
    pid_t pid = launch(argv, &fd);
    if (pid < 0) return NULL;
    size_t size = 4096, len = 0;
    char *data = malloc(size);
    if (!data) { close(fd); waitpid(pid, NULL, 0); return NULL; }
    for (;;) {
        if (len + 4096 + 1 > size) {
            size *= 2;
            char *next = realloc(data, size);
            if (!next) { free(data); close(fd); kill(pid, SIGTERM); waitpid(pid, NULL, 0); return NULL; }
            data = next;
        }
        ssize_t n = read(fd, data + len, size - len - 1);
        if (n > 0) len += (size_t)n;
        else if (n == 0) break;
        else if (errno != EINTR) break;
    }
    close(fd);
    int status;
    waitpid(pid, &status, 0);
    data[len] = 0;
    if (!WIFEXITED(status) || WEXITSTATUS(status)) { free(data); return NULL; }
    return data;
}

void sam_linux_video_close(SamVideoReader *r) {
    if (!r) return;
    if (r->fd >= 0) close(r->fd);
    if (r->pid > 0) { kill(r->pid, SIGTERM); waitpid(r->pid, NULL, 0); }
    if (r->pts_file) fclose(r->pts_file);
    if (r->pts_pid > 0) { kill(r->pts_pid, SIGTERM); waitpid(r->pts_pid, NULL, 0); }
    free(r);
}

SamVideoReader *sam_linux_video_open(const char *path, double start) {
    char *meta_argv[] = {"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate:format=duration", "-of", "default=noprint_wrappers=1", (char *)path, NULL};
    char *meta = capture(meta_argv);
    if (!meta) return NULL;
    SamVideoReader *r = calloc(1, sizeof(*r));
    if (!r) { free(meta); return NULL; }
    r->fd = -1;
    r->fps = 30;
    for (char *line = strtok(meta, "\n"); line; line = strtok(NULL, "\n")) {
        if (!strncmp(line, "width=", 6)) r->width = atoi(line + 6);
        else if (!strncmp(line, "height=", 7)) r->height = atoi(line + 7);
        else if (!strncmp(line, "duration=", 9)) r->duration = atof(line + 9);
        else if (!strncmp(line, "avg_frame_rate=", 15)) {
            double num = 0, den = 0;
            if (sscanf(line + 15, "%lf/%lf", &num, &den) == 2 && den > 0 && num > 0) r->fps = num / den;
        }
    }
    free(meta);
    if (r->width <= 0 || r->height <= 0 || (uint64_t)r->width * (uint64_t)r->height > 100000000) { sam_linux_video_close(r); return NULL; }
    r->start = start > 0 && isfinite(start) ? start : 0;
    char interval_buf[64];
    snprintf(interval_buf, sizeof(interval_buf), "%.9f%%", r->start);
    char *pts_argv[] = {"ffprobe", "-v", "error", "-read_intervals", interval_buf, "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", (char *)path, NULL};
    int pts_fd;
    r->pts_pid = launch(pts_argv, &pts_fd);
    if (r->pts_pid > 0) {
        r->pts_file = fdopen(pts_fd, "r");
        if (!r->pts_file) { close(pts_fd); kill(r->pts_pid, SIGTERM); waitpid(r->pts_pid, NULL, 0); r->pts_pid = 0; }
    } else r->pts_pid = 0;
    char seek_buf[64];
    snprintf(seek_buf, sizeof(seek_buf), "%.9f", r->start);
    char *ffmpeg_argv[] = {"ffmpeg", "-v", "error", "-nostdin", "-ss", seek_buf, "-i", (char *)path, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", NULL};
    r->pid = launch(ffmpeg_argv, &r->fd);
    if (r->pid < 0) { r->pid = 0; sam_linux_video_close(r); return NULL; }
    return r;
}

double sam_linux_video_duration(SamVideoReader *r) { return r->duration; }

static double next_pts(SamVideoReader *r) {
    if (r->pts_file) {
        char line[128];
        while (r->pts_len < 64 && fgets(line, sizeof(line), r->pts_file)) {
            char *end;
            double pts = strtod(line, &end);
            if (end == line || !isfinite(pts) || pts + 0.000001 < r->start) continue;
            r->pts_buffer[r->pts_len++] = pts;
        }
        if (r->pts_len) {
            size_t first = 0;
            for (size_t i = 1; i < r->pts_len; i++) {
                if (r->pts_buffer[i] < r->pts_buffer[first]) first = i;
            }
            double pts = r->pts_buffer[first];
            r->pts_buffer[first] = r->pts_buffer[--r->pts_len];
            return pts;
        }
    }
    return r->start + (double)r->index / r->fps;
}

int sam_linux_video_next(SamVideoReader *r, SamVideoFrame *frame) {
    size_t size = (size_t)r->width * (size_t)r->height * 3;
    unsigned char *rgb = malloc(size);
    if (!rgb) return -1;
    size_t done = 0;
    while (done < size) {
        ssize_t n = read(r->fd, rgb + done, size - done);
        if (n > 0) done += (size_t)n;
        else if (n == 0) { free(rgb); return done ? -1 : 0; }
        else if (errno != EINTR) { free(rgb); return -1; }
    }
    frame->rgb = rgb;
    frame->width = r->width;
    frame->height = r->height;
    frame->pts_seconds = next_pts(r);
    r->index++;
    return 1;
}
void sam_linux_video_free_frame(SamVideoFrame *frame) { free(frame->rgb); frame->rgb = NULL; }
