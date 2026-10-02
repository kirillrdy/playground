const std = @import("std");
const sam3 = @import("sam3");
const render = sam3.render;
const zigimg = @import("zigimg");
const zimo = @import("zimo");
const log = @import("log");
const vdb = @import("vdb");
const QueryTab = vdb.query_tab.QueryTab;
const here = zimo.bind(@This(), @embedFile("app.zig"));

const SamCallbacks = extern struct {
    on_open_file: ?*const fn (path: [*:0]const u8) callconv(.c) void,
    on_open_video: ?*const fn (path: [*:0]const u8) callconv(.c) void,
    on_video_play_pause: ?*const fn () callconv(.c) void,
    on_video_seek: ?*const fn (seconds: f64) callconv(.c) void,
    on_video_step: ?*const fn () callconv(.c) void,
    on_sample_click: ?*const fn () callconv(.c) void,
    on_mode_change: ?*const fn (mode: c_int) callconv(.c) void,
    on_clear_points: ?*const fn () callconv(.c) void,
    on_find_text: ?*const fn (text: [*:0]const u8) callconv(.c) void,
    on_cancel_query: ?*const fn () callconv(.c) void,
    on_clear_query: ?*const fn () callconv(.c) void,
    on_select_query: ?*const fn (index: usize) callconv(.c) void,
    on_close_query: ?*const fn (index: usize) callconv(.c) void,
    on_reap_queries: ?*const fn () callconv(.c) void,
    on_new_query: ?*const fn () callconv(.c) void,
    on_edit_query: ?*const fn (text: [*:0]const u8) callconv(.c) void,
    on_canvas_click: ?*const fn (norm_x: f32, norm_y: f32, is_positive: c_int) callconv(.c) void,
    on_select_mask: ?*const fn (mask_index: c_int) callconv(.c) void,
};

const SamMaskInfo = extern struct {
    score: f32,
    coverage: f32,
    red: u8,
    green: u8,
    blue: u8,
};

const MaskColor = struct { red: u8, green: u8, blue: u8 };
const mask_colors = [_]MaskColor{
    .{ .red = 0, .green = 220, .blue = 100 },
    .{ .red = 0, .green = 180, .blue = 255 },
    .{ .red = 255, .green = 175, .blue = 0 },
    .{ .red = 220, .green = 90, .blue = 255 },
    .{ .red = 255, .green = 80, .blue = 110 },
    .{ .red = 255, .green = 225, .blue = 80 },
    .{ .red = 70, .green = 220, .blue = 210 },
    .{ .red = 160, .green = 155, .blue = 255 },
};

fn maskColor(index: usize) MaskColor {
    return mask_colors[index % mask_colors.len];
}

const VideoFrame = extern struct {
    rgb: ?[*]u8,
    width: c_int,
    height: c_int,
    pts_seconds: f64,
};

extern fn sam_macos_init(callbacks: *const SamCallbacks) c_int;
extern fn sam_macos_run() void;
extern fn sam_macos_set_window_title(title: [*:0]const u8) void;
extern fn sam_macos_set_status(text: [*:0]const u8) void;
extern fn sam_macos_set_image(rgba_pixels: ?[*]const u8, width: c_int, height: c_int) void;
extern fn sam_macos_set_masks(count: c_int, masks: ?[*]const SamMaskInfo, best_index: c_int, selected_index: c_int) void;
extern fn sam_macos_set_busy(is_busy: c_int) void;
extern fn sam_macos_file_exists(path: [*:0]const u8) c_int;
extern fn sam_macos_get_home() ?[*:0]const u8;
extern fn sam_macos_set_video_mode(active: c_int, playing: c_int) void;
extern fn sam_macos_set_video_timeline(duration: f64, position: f64) void;
extern fn sam_macos_set_precache_progress(state: c_int, fraction: f64, frames: usize) void;
extern fn sam_macos_set_query_active(active: c_int) void;
extern fn sam_macos_set_query_tabs(labels: [*:0]const u8, selected: c_int) void;
extern fn sam_macos_set_query_text(sql: [*:0]const u8) void;
extern fn sam_macos_video_open(path: [*:0]const u8, start_seconds: f64) ?*anyopaque;
extern fn sam_macos_video_duration(reader: *anyopaque) f64;
extern fn sam_macos_video_fps(reader: *anyopaque) f64;
extern fn sam_macos_video_next(reader: *anyopaque, frame: *VideoFrame) c_int;
extern fn sam_macos_video_free_frame(frame: *VideoFrame) void;
extern fn sam_macos_video_close(reader: *anyopaque) void;
extern fn sam_macos_dispatch_main(func: *const fn (?*anyopaque) callconv(.c) void, ctx: ?*anyopaque) void;

const MacosVideoReader = struct {
    path: [:0]const u8,
    duration: f64,
    fps: f64,
    reader: ?*anyopaque = null,
    current_index: usize = 0,
    current_frame: ?VideoFrame = null,

    pub fn init(path: [:0]const u8) MacosVideoReader {
        const handle = sam_macos_video_open(path.ptr, 0);
        const dur = if (handle) |h| sam_macos_video_duration(h) else 0;
        return .{
            .path = path,
            .duration = dur,
            .fps = if (handle) |h| sam_macos_video_fps(h) else 30.0,
            .reader = handle,
            .current_index = 0,
            .current_frame = null,
        };
    }

    pub fn deinit(self: *MacosVideoReader) void {
        if (self.current_frame) |*vf| {
            sam_macos_video_free_frame(vf);
            self.current_frame = null;
        }
        if (self.reader) |r| {
            sam_macos_video_close(r);
            self.reader = null;
        }
    }

    pub fn asReader(self: *MacosVideoReader) vdb.engine.VideoReader {
        return .{
            .ptr = self,
            .vtable = &.{
                .totalFrames = totalFramesImpl,
                .seekToFrame = seekToFrameImpl,
                .nextFrame = nextFrameImpl,
            },
        };
    }

    fn totalFramesImpl(ctx: *anyopaque) usize {
        const self: *MacosVideoReader = @ptrCast(@alignCast(ctx));
        return @intFromFloat(@max(self.duration * self.fps, 1.0));
    }

    fn seekToFrameImpl(ctx: *anyopaque, frame_idx: usize) anyerror!void {
        const self: *MacosVideoReader = @ptrCast(@alignCast(ctx));
        if (self.current_frame) |*vf| {
            sam_macos_video_free_frame(vf);
            self.current_frame = null;
        }
        if (self.reader) |r| sam_macos_video_close(r);
        const pts = @as(f64, @floatFromInt(frame_idx)) / self.fps;
        self.reader = if (pts >= self.duration) null else sam_macos_video_open(self.path.ptr, pts);
        self.current_index = frame_idx;
    }

    fn nextFrameImpl(ctx: *anyopaque) anyerror!?vdb.types.FrameRef {
        const self: *MacosVideoReader = @ptrCast(@alignCast(ctx));
        if (self.reader == null) return null;
        if (self.current_frame) |*vf| {
            sam_macos_video_free_frame(vf);
            self.current_frame = null;
        }

        var vf: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
        const ret = sam_macos_video_next(self.reader.?, &vf);
        if (ret <= 0) return null;

        self.current_frame = vf;
        const idx = self.current_index;
        self.current_index += 1;

        return vdb.types.FrameRef{
            .index = idx,
            .pts_seconds = vf.pts_seconds,
            .width = @intCast(vf.width),
            .height = @intCast(vf.height),
            .rgb = vf.rgb,
        };
    }
};

const max_points = 32;

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    mutex: std.Io.Mutex = .init,
    model_mutex: std.Io.Mutex = .init,
    index_mutex: std.Io.Mutex = .init,
    is_busy: bool = false,

    image: ?zigimg.Image = null,
    frame: []u8 = &.{},

    points: [max_points]sam3.Point = undefined,
    points_len: usize = 0,
    click_mode_add: bool = true,

    masks: ?sam3.Masks = null,
    coverages: []f32 = &.{},
    selected_mask: i32 = -1,
    best_mask_idx: i32 = -1,

    video_path: ?[:0]u8 = null,
    video_thread: ?std.Thread = null,
    video_active: bool = false,
    video_stop: std.atomic.Value(bool) = .init(false),
    video_playing: std.atomic.Value(bool) = .init(false),
    video_phrase: [256]u8 = undefined,
    video_phrase_len: usize = 0,
    video_overlay_prompts: vdb.overlay.Prompts = .{},
    video_duration: f64 = 0,
    video_position: f64 = 0,
    video_previewed: bool = false,
    video_seek_target: ?f64 = null,
    video_step_requested: bool = false,
    query_tabs: std.ArrayList(*QueryTab) = .empty,
    closed_queries: std.ArrayList(*QueryTab) = .empty,
    selected_query: ?usize = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) App {
        return .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
        };
    }

    pub fn deinit(self: *App) void {
        for (self.query_tabs.items) |tab| tab.query_cancel.store(true, .release);
        for (self.query_tabs.items) |tab| {
            if (tab.query_thread) |thread| thread.join();
            tab.query_thread = null;
        }
        for (self.closed_queries.items) |tab| tab.deinit();
        self.closed_queries.deinit(self.allocator);
        self.stopVideo();
        for (self.query_tabs.items) |tab| tab.deinit();
        self.query_tabs.deinit(self.allocator);
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        if (self.image) |*img| img.deinit(self.allocator);
        self.allocator.free(self.frame);
        self.allocator.free(self.coverages);
    }

    pub fn start(self: *App) !void {
        g_app = self;

        const callbacks: SamCallbacks = .{
            .on_open_file = &cOpenFile,
            .on_open_video = &cOpenVideo,
            .on_video_play_pause = &cVideoPlayPause,
            .on_video_seek = &cVideoSeek,
            .on_video_step = &cVideoStep,
            .on_sample_click = &cSampleClick,
            .on_mode_change = &cModeChange,
            .on_clear_points = &cClearPoints,
            .on_find_text = &cFindText,
            .on_cancel_query = &cCancelQuery,
            .on_clear_query = &cClearQuery,
            .on_select_query = &cSelectQuery,
            .on_close_query = &cCloseQuery,
            .on_reap_queries = &cReapQueries,
            .on_new_query = &cNewQuery,
            .on_edit_query = &cEditQuery,
            .on_canvas_click = &cCanvasClick,
            .on_select_mask = &cSelectMask,
        };

        if (sam_macos_init(&callbacks) != 0) {
            return error.MacosUiInitFailed;
        }

        self.newQuery();

        sam_macos_run();
    }

    fn openImageFromPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        self.stopVideo();
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            log.info(self.io, "Failed to read image file {s}: {t}", .{ path, err });
            sam_macos_set_status("Could not open that file.");
            return;
        };
        defer self.allocator.free(file_bytes);

        self.openImageFromBytes(file_bytes);
    }

    fn openImageFromBytes(self: *App, bytes: []const u8) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        var decoded = sam3.decodeImage(self.allocator, bytes) catch |err| {
            log.info(self.io, "Failed to decode image: {t}", .{err});
            sam_macos_set_status("That file is not an image this can decode.");
            return;
        };

        if (self.image) |*old| old.deinit(self.allocator);
        self.allocator.free(self.frame);
        if (self.masks) |*m| m.deinit();
        self.masks = null;

        self.points_len = 0;
        self.selected_mask = -1;
        self.best_mask_idx = -1;

        self.image = decoded;
        self.frame = self.allocator.alloc(u8, decoded.width * decoded.height * 4) catch {
            decoded.deinit(self.allocator);
            self.image = null;
            self.frame = &.{};
            sam_macos_set_status("Out of memory for frame buffer.");
            return;
        };

        self.renderComposite(-1);
        sam_macos_set_image(self.frame.ptr, @intCast(decoded.width), @intCast(decoded.height));
        sam_macos_set_masks(0, null, 0, -1);

        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrintZ(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }) catch "Image loaded.";
        sam_macos_set_status(msg);
    }

    fn stopVideo(self: *App) void {
        self.video_stop.store(true, .release);
        if (self.video_thread) |thread| thread.join();
        self.video_thread = null;
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);
        if (self.video_path) |path| self.allocator.free(path);
        self.video_path = null;
        self.video_active = false;
        self.video_phrase_len = 0;
        self.video_overlay_prompts = .{};
        self.video_playing.store(false, .release);
        self.video_duration = 0;
        self.video_position = 0;
        self.video_previewed = false;
        self.video_seek_target = null;
        self.video_step_requested = false;
        sam_macos_set_video_mode(0, 0);
        sam_macos_set_video_timeline(0, 0);
        sam_macos_set_window_title("SAM 3 — Visual Database");
    }

    fn openVideoFromPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        self.stopVideo();
        const owned = self.allocator.dupeZ(u8, path) catch {
            sam_macos_set_status("Out of memory opening video.");
            return;
        };
        self.mutex.lock(self.io) catch {
            self.allocator.free(owned);
            return;
        };
        self.video_path = owned;
        if (self.selectedTab()) |tab| {
            if (!tab.has_run and !std.mem.eql(u8, tab.query_path.?, owned)) {
                const source = self.allocator.dupeZ(u8, owned) catch null;
                if (source) |path_copy| {
                    self.allocator.free(tab.query_path.?);
                    tab.query_path = path_copy;
                }
            }
        }
        self.video_stop.store(false, .release);
        self.video_active = true;
        self.video_seek_target = 0;
        if (self.image) |*old| old.deinit(self.allocator);
        self.image = null;
        self.allocator.free(self.frame);
        self.frame = &.{};
        if (self.masks) |*old| old.deinit();
        self.masks = null;
        self.points_len = 0;
        self.selected_mask = -1;
        self.best_mask_idx = -1;
        self.restoreQueryOverlay();
        self.mutex.unlock(self.io);
        sam_macos_set_image(null, 0, 0);
        sam_macos_set_masks(0, null, 0, -1);
        sam_macos_set_video_mode(1, 0);

        // Update window title with video file name
        const basename = std.fs.path.basename(path);
        var title_buf: [256]u8 = undefined;
        const title_str = std.fmt.bufPrintZ(&title_buf, "SAM 3 — Visual Database — {s}", .{basename}) catch "SAM 3 — Visual Database";
        sam_macos_set_window_title(title_str);

        // Try auto-loading sidecar index if exists
        var index_loaded = false;
        var indexed_concepts_count: usize = 0;
        if (std.fmt.allocPrint(self.allocator, "{s}.vdb", .{path})) |sidecar| {
            defer self.allocator.free(sidecar);
            if (vdb.index.InvertedIndex.loadFromFile(self.allocator, sidecar)) |loaded| {
                var loaded_idx = loaded;
                defer loaded_idx.deinit();
                indexed_concepts_count = loaded_idx.classes.count();
                index_loaded = true;
                log.info(self.io, "Auto-loaded visual index for \"{s}\" ({d} concepts) from {s}", .{ path, indexed_concepts_count, sidecar });
            } else |_| {}
        } else |_| {}

        if (index_loaded) {
            var sbuf: [160]u8 = undefined;
            const smsg = std.fmt.bufPrintZ(&sbuf, "Video opened. Loaded visual index with {d} concept(s) from .vdb sidecar.", .{indexed_concepts_count}) catch "Video opened with index.";
            sam_macos_set_status(smsg);
        } else {
            sam_macos_set_status("Video opened. Enter a word or SQL query and press Find/Query.");
        }
        self.video_thread = std.Thread.spawn(.{}, runVideoWorker, .{self}) catch {
            self.mutex.lock(self.io) catch return;
            self.allocator.free(self.video_path.?);
            self.video_path = null;
            self.video_active = false;
            self.mutex.unlock(self.io);
            sam_macos_set_video_mode(0, 0);
            sam_macos_set_status("Could not start video playback.");
            return;
        };
    }

    fn handleVideoPlayPause(self: *App) void {
        if (!self.video_active) return;
        self.mutex.lock(self.io) catch return;
        const has_phrase = self.video_phrase_len > 0;
        const has_matches = self.isQueryVideo() and self.queryMatches().len > 0;
        self.mutex.unlock(self.io);
        if (!has_phrase and !has_matches) {
            sam_macos_set_status("Enter a visual query or concept before playing video.");
            return;
        }
        const playing = !self.video_playing.load(.acquire);
        self.video_playing.store(playing, .release);
        sam_macos_set_video_mode(1, @intFromBool(playing));
        if (!playing) {
            sam_macos_set_status("Video paused.");
        } else if (has_matches) {
            sam_macos_set_status("Playing query matches…");
        }
    }

    fn handleVideoSeek(self: *App, seconds: f64) void {
        if (!self.video_active or !std.math.isFinite(seconds)) return;
        self.mutex.lock(self.io) catch return;
        if (seconds == 0 and self.isQueryVideo() and self.queryMatches().len > 0) {
            self.selectedTab().?.query_match_idx = 0;
            const target_frame = self.queryMatches()[0];
            const target_sec = self.selectedTab().?.matchTime(self.selectedTab().?.query_match_idx);
            self.video_seek_target = target_sec;
            self.video_step_requested = true;
            const total = self.queryMatches().len;
            self.mutex.unlock(self.io);
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrintZ(&buf, "Restarted at query match 1 of {d} (Frame #{d} at {d:.2}s).", .{ total, target_frame, target_sec }) catch "Query match 1.";
            sam_macos_set_status(msg);
            return;
        }
        const target = std.math.clamp(seconds, 0, self.video_duration);
        const duration = self.video_duration;
        self.video_seek_target = target;
        self.video_step_requested = true;

        if (self.isQueryVideo() and self.queryMatches().len > 0) {
            var closest_idx: usize = 0;
            var closest_diff: f64 = 1e9;
            for (self.selectedTab().?.query_match_pts.items, 0..) |f_sec, i| {
                const diff = @abs(f_sec - target);
                if (diff < closest_diff) {
                    closest_diff = diff;
                    closest_idx = i;
                }
            }
            self.selectedTab().?.query_match_idx = closest_idx;
        }

        self.mutex.unlock(self.io);
        sam_macos_set_video_timeline(duration, target);
    }

    fn handleVideoStep(self: *App) void {
        if (!self.video_active or self.video_playing.load(.acquire)) return;
        self.mutex.lock(self.io) catch return;
        if (self.isQueryVideo() and self.queryMatches().len > 0) {
            self.selectedTab().?.query_match_idx = (self.selectedTab().?.query_match_idx + 1) % self.queryMatches().len;
            const target_frame = self.queryMatches()[self.selectedTab().?.query_match_idx];
            const target_sec = self.selectedTab().?.matchTime(self.selectedTab().?.query_match_idx);
            self.video_seek_target = target_sec;
            self.video_step_requested = true;
            const match_num = self.selectedTab().?.query_match_idx + 1;
            const total = self.queryMatches().len;
            self.mutex.unlock(self.io);
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrintZ(&buf, "Query match {d} of {d} (Frame #{d} at {d:.2}s).", .{ match_num, total, target_frame, target_sec }) catch "Next query match.";
            sam_macos_set_status(msg);
            return;
        }
        const has_phrase = self.video_phrase_len > 0;
        if (has_phrase) self.video_step_requested = true;
        self.mutex.unlock(self.io);
        if (!has_phrase) {
            sam_macos_set_status("Enter a word and press Find before stepping video.");
            return;
        }
        self.video_playing.store(false, .release);
        sam_macos_set_video_mode(1, 0);
    }

    fn runVideoWorker(self: *App) void {
        var reader: ?*anyopaque = null;
        defer if (reader) |handle| sam_macos_video_close(handle);
        var previous_pts: ?f64 = null;
        var previous_display: ?std.Io.Timestamp = null;
        var retry_required = false;
        var open_at: f64 = 0;

        while (!self.video_stop.load(.acquire)) {
            self.mutex.lock(self.io) catch return;
            const seek = self.video_seek_target;
            self.video_seek_target = null;
            const step = self.video_step_requested;
            self.video_step_requested = false;
            self.mutex.unlock(self.io);
            if (seek) |target| {
                if (reader) |handle| sam_macos_video_close(handle);
                reader = null;
                open_at = target;
                previous_pts = null;
                previous_display = null;
            }
            if (reader == null and (!retry_required or self.video_playing.load(.acquire) or step or seek != null)) {
                reader = sam_macos_video_open(self.video_path.?.ptr, open_at);
                retry_required = true;
                if (reader == null) {
                    self.video_playing.store(false, .release);
                    sam_macos_set_video_mode(1, 0);
                    sam_macos_set_status("Could not decode this video.");
                } else {
                    const duration = sam_macos_video_duration(reader.?);
                    self.mutex.lock(self.io) catch return;
                    self.video_duration = duration;
                    self.mutex.unlock(self.io);
                    sam_macos_set_video_timeline(duration, open_at);
                }
            }
            if ((!self.video_playing.load(.acquire) and !step and seek == null) or reader == null) {
                std.Io.sleep(self.io, .fromMilliseconds(20), .awake) catch {};
                continue;
            }

            var video_frame: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
            const next = sam_macos_video_next(reader.?, &video_frame);
            if (next <= 0) {
                sam_macos_video_close(reader.?);
                reader = null;
                retry_required = true;
                self.video_playing.store(false, .release);
                sam_macos_set_video_mode(1, 0);
                sam_macos_set_status(if (next == 0) "End of video. Press Play to replay." else "Video decoding failed. Press Play to retry.");
                previous_pts = null;
                previous_display = null;
                open_at = 0;
                continue;
            }
            defer sam_macos_video_free_frame(&video_frame);
            const width: usize = @intCast(video_frame.width);
            const height: usize = @intCast(video_frame.height);
            const pixels_len = std.math.mul(usize, width, height) catch continue;
            const rgb_len = std.math.mul(usize, pixels_len, 3) catch continue;
            if (seek != null) {
                const preview = self.allocator.alloc(u8, std.math.mul(usize, pixels_len, 4) catch continue) catch continue;
                defer self.allocator.free(preview);
                const rgb = video_frame.rgb.?;
                for (0..pixels_len) |i| {
                    preview[i * 4] = rgb[i * 3];
                    preview[i * 4 + 1] = rgb[i * 3 + 1];
                    preview[i * 4 + 2] = rgb[i * 3 + 2];
                    preview[i * 4 + 3] = 255;
                }
                sam_macos_set_image(preview.ptr, @intCast(width), @intCast(height));
                sam_macos_set_masks(0, null, 0, -1);
                self.mutex.lock(self.io) catch return;
                self.video_position = video_frame.pts_seconds;
                self.video_previewed = true;
                const duration = self.video_duration;
                self.mutex.unlock(self.io);
                sam_macos_set_video_timeline(duration, video_frame.pts_seconds);
            }
            const pixels = self.allocator.dupe(u8, video_frame.rgb.?[0..rgb_len]) catch continue;
            var decoded = zigimg.Image.fromRawPixelsOwned(width, height, pixels, .rgb24) catch {
                self.allocator.free(pixels);
                continue;
            };
            var decoded_owned = true;
            defer if (decoded_owned) decoded.deinit(self.allocator);

            var phrase_buf: [256]u8 = undefined;
            self.mutex.lock(self.io) catch return;
            const has_query_matches = self.queryMatches().len > 0;
            const overlay_prompts = self.video_overlay_prompts;
            const phrase_len = self.video_phrase_len;
            @memcpy(phrase_buf[0..phrase_len], self.video_phrase[0..phrase_len]);
            self.mutex.unlock(self.io);
            if (phrase_len == 0 and overlay_prompts.count == 0 and !has_query_matches and !step and seek == null) continue;
            const phrase = phrase_buf[0..phrase_len];

            const started = std.Io.Timestamp.now(self.io, .awake);
            var masks = self.videoOverlayMasks(sam3.RgbImage.fromImage(decoded), phrase, &overlay_prompts, video_frame.pts_seconds) catch |err| {
                log.info(self.io, "Video frame lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
                sam_macos_set_status("Video frame inference failed.");
                self.video_playing.store(false, .release);
                sam_macos_set_video_mode(1, 0);
                continue;
            };
            defer if (masks) |*m| m.deinit();
            const lookup_elapsed = started.untilNow(self.io, .awake);

            if (!step) {
                if (previous_pts) |pts| {
                    const interval = video_frame.pts_seconds - pts;
                    if (interval > 0 and interval < 1 and std.math.isFinite(interval)) {
                        const target_ns: i96 = @intFromFloat(interval * 1e9);
                        const elapsed_ns = previous_display.?.untilNow(self.io, .awake).nanoseconds;
                        if (target_ns > elapsed_ns) {
                            std.Io.sleep(self.io, .fromNanoseconds(target_ns - elapsed_ns), .awake) catch {};
                        }
                    }
                }
            }
            if (self.video_stop.load(.acquire)) break;
            self.mutex.lock(self.io) catch return;
            const stale = self.video_seek_target != null;
            self.mutex.unlock(self.io);
            if (stale or (!self.video_playing.load(.acquire) and !step)) continue;

            const rgba_len = std.math.mul(usize, pixels_len, 4) catch continue;
            const next_frame = self.allocator.alloc(u8, rgba_len) catch continue;
            self.mutex.lock(self.io) catch {
                self.allocator.free(next_frame);
                return;
            };
            if (self.image) |*old| old.deinit(self.allocator);
            self.allocator.free(self.frame);
            if (self.masks) |*old| old.deinit();
            self.image = decoded;
            decoded_owned = false;
            self.frame = next_frame;
            self.masks = masks;
            masks = null;
            self.points_len = 0;
            const mask_count = if (self.masks) |m| m.count else 0;
            if (self.coverages.len != mask_count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, mask_count) catch &.{};
            }
            self.best_mask_idx = if (mask_count == 0) -1 else blk: {
                const m = self.masks.?;
                break :blk @intCast(render.scoreMasks(m.logits, m.scores, m.count, m.width, m.height, self.coverages));
            };
            self.selected_mask = self.best_mask_idx;
            self.video_position = video_frame.pts_seconds;
            self.renderComposite(self.selected_mask);
            sam_macos_set_image(self.frame.ptr, @intCast(width), @intCast(height));
            if (self.masks) |m| self.showMaskButtons(m) else sam_macos_set_masks(0, null, 0, -1);
            const duration = self.video_duration;
            self.mutex.unlock(self.io);
            sam_macos_set_video_timeline(duration, video_frame.pts_seconds);

            previous_pts = video_frame.pts_seconds;
            previous_display = std.Io.Timestamp.now(self.io, .awake);
            const seconds = if (std.math.isFinite(video_frame.pts_seconds)) @max(video_frame.pts_seconds, 0) else 0;
            const position_ms: u64 = @intFromFloat(@min(seconds * 1000, 1.0e15));
            var time_buf: [32]u8 = undefined;
            const timecode = std.fmt.bufPrint(&time_buf, "{d}:{d:0>2}.{d:0>3}", .{
                position_ms / 60_000,
                position_ms / 1_000 % 60,
                position_ms % 1_000,
            }) catch "0:00.000";
            var status_buf: [256]u8 = undefined;
            const status = if (phrase_len == 0 and overlay_prompts.count == 0)
                std.fmt.bufPrintZ(&status_buf, "At {s}: frame shown.", .{timecode}) catch "Frame shown."
            else if (self.masks != null)
                std.fmt.bufPrintZ(&status_buf, "At {s}: {d} match(es) for “{s}” in {f}", .{
                    timecode, mask_count, phrase, lookup_elapsed,
                }) catch "Video frame processed."
            else
                std.fmt.bufPrintZ(&status_buf, "At {s}: waiting for cached “{s}” result.", .{ timecode, phrase }) catch "Frame shown without a cached result.";
            sam_macos_set_status(status);
            if (self.masks != null) log.info(self.io, "at {s}: \"{s}\" -> {d} object(s) in {f}", .{
                timecode, phrase, mask_count, lookup_elapsed,
            });

            // If video playback is running and query results exist, advance through matched frames!
            if (self.video_playing.load(.acquire)) {
                self.mutex.lock(self.io) catch return;
                const total_matches = if (self.isQueryVideo()) self.queryMatches().len else 0;
                if (total_matches > 0) {
                    if (self.selectedTab().?.query_match_idx + 1 < total_matches) {
                        self.selectedTab().?.query_match_idx += 1;
                    } else {
                        self.selectedTab().?.query_match_idx = 0;
                    }
                    const next_match_frame = self.queryMatches()[self.selectedTab().?.query_match_idx];
                    const next_sec = self.selectedTab().?.matchTime(self.selectedTab().?.query_match_idx);
                    self.video_seek_target = next_sec;
                    self.video_step_requested = true;
                    const cur_num = self.selectedTab().?.query_match_idx + 1;
                    self.mutex.unlock(self.io);

                    std.Io.sleep(self.io, .fromMilliseconds(250), .awake) catch {};

                    var match_buf: [160]u8 = undefined;
                    const match_msg = std.fmt.bufPrintZ(&match_buf, "Playing query match {d} of {d} (Frame #{d} at {d:.2}s)", .{
                        cur_num, total_matches, next_match_frame, next_sec,
                    }) catch "Playing query match.";
                    sam_macos_set_status(match_msg);
                    continue;
                }
                self.mutex.unlock(self.io);
            }
        }
    }

    fn cachedFrameQuery(self: *App, image: sam3.RgbImage, phrase: []const u8) ![]f32 {
        const args = .{
            self.allocator,
            [_][]const u8{
                sam3.assets.concept_vision_encoder.sha256,
                sam3.assets.concept_text_encoder.sha256,
                sam3.assets.concept_decoder.sha256,
                sam3.assets.concept_tokenizer_json.sha256,
                sam3.assets.concept_vision_encoder_data.sha256,
                sam3.assets.concept_text_encoder_data.sha256,
            },
            image,
            phrase,
            @as(f32, 0.5),
        };
        return here.call(.computeQuery, args);
    }

    fn videoQuery(self: *App, image: sam3.RgbImage, phrase: []const u8, pts: f64) !?[]f32 {
        self.mutex.lock(self.io) catch return error.VideoLockFailed;
        if (self.selectedTab()) |tab| {
            if (self.isQueryVideo() and tab.precache_active.load(.acquire)) {
                const scanned_phrase = std.mem.eql(u8, phrase, tab.precache_phrase[0..tab.precache_phrase_len]);
                if (!scanned_phrase or !std.math.isFinite(pts) or pts > tab.precache_scanned_until) {
                    self.mutex.unlock(self.io);
                    return null;
                }
            }
        }
        self.mutex.unlock(self.io);
        self.model_mutex.lock(self.io) catch return error.ModelLockFailed;
        defer self.model_mutex.unlock(self.io);
        return try self.cachedFrameQuery(image, phrase);
    }

    fn videoOverlayMasks(self: *App, image: sam3.RgbImage, phrase: []const u8, prompts: *const vdb.overlay.Prompts, pts: f64) !?sam3.Masks {
        var combined: ?sam3.Masks = null;
        errdefer if (combined) |*m| m.deinit();
        const count = if (prompts.count > 0) prompts.count else @as(usize, if (phrase.len > 0) 1 else 0);
        for (0..count) |i| {
            const prompt = if (prompts.count > 0) prompts.get(i) else phrase;
            const values = (try self.videoQuery(image, prompt, pts)) orelse continue;
            defer self.allocator.free(values);
            var masks = unpackMasks(self.allocator, values) catch |err| {
                self.mutex.lock(self.io) catch return error.VideoLockFailed;
                const indexing = if (self.selectedTab()) |tab| self.isQueryVideo() and tab.precache_active.load(.acquire) else false;
                self.mutex.unlock(self.io);
                if (indexing) continue;
                return err;
            };
            if (combined) |*existing| {
                defer masks.deinit();
                if (existing.width != masks.width or existing.height != masks.height) return error.IncompatibleMaskDimensions;
                const scores = try self.allocator.alloc(f32, existing.scores.len + masks.scores.len);
                errdefer self.allocator.free(scores);
                const logits = try self.allocator.alloc(f32, existing.logits.len + masks.logits.len);
                @memcpy(scores[0..existing.scores.len], existing.scores);
                @memcpy(scores[existing.scores.len..], masks.scores);
                @memcpy(logits[0..existing.logits.len], existing.logits);
                @memcpy(logits[existing.logits.len..], masks.logits);
                self.allocator.free(existing.scores);
                self.allocator.free(existing.logits);
                existing.scores = scores;
                existing.logits = logits;
                existing.count += masks.count;
                existing.object_score = @max(existing.object_score, masks.object_score);
            } else {
                combined = masks;
            }
        }
        return combined;
    }

    fn sam3SegmentBridge(ctx: *anyopaque, allocator: std.mem.Allocator, frame: vdb.types.FrameRef, prompt: []const u8) anyerror!vdb.types.MaskRef {
        _ = allocator;
        const self: *App = @ptrCast(@alignCast(ctx));
        return self.evaluateSam3Frame(frame, prompt);
    }

    fn evaluateSam3Frame(self: *App, frame: vdb.types.FrameRef, prompt: []const u8) !vdb.types.MaskRef {
        if (frame.rgb == null or frame.width == 0 or frame.height == 0) {
            return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
        }
        const pixel_count = std.math.mul(usize, frame.width, frame.height) catch return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
        const bytes_len = std.math.mul(usize, pixel_count, 3) catch return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
        const rgb_img = sam3.RgbImage{
            .pixels = frame.rgb.?[0..bytes_len],
            .width = frame.width,
            .height = frame.height,
        };
        self.model_mutex.lock(self.io) catch return error.ModelLockFailed;
        defer self.model_mutex.unlock(self.io);
        const maybe_vals: ?[]f32 = try self.cachedFrameQuery(rgb_img, prompt);
        if (maybe_vals) |vals| {
            defer self.allocator.free(vals);
            if (vals.len >= 4) {
                const count: usize = @as(u32, @bitCast(vals[0]));
                if (count > 0) {
                    const obj_score = vals[3];
                    const best_score = if (vals.len >= 5) @max(obj_score, vals[4]) else obj_score;
                    return .{
                        .score = best_score,
                        .coverage = 0.2,
                        .width = frame.width,
                        .height = frame.height,
                    };
                }
            }
        }
        return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
    }

    fn saveVideoIndex(self: *App, tab: *QueryTab, builder: *vdb.index.IndexBuilder) !void {
        try self.index_mutex.lock(self.io);
        defer self.index_mutex.unlock(self.io);
        const sidecar = try std.fmt.allocPrint(self.allocator, "{s}.vdb", .{tab.query_path.?});
        defer self.allocator.free(sidecar);
        if (vdb.index.InvertedIndex.loadFromFile(self.allocator, sidecar)) |loaded| {
            var latest = loaded;
            defer latest.deinit();
            try builder.importIndex(&latest);
        } else |_| {}
        var built = try builder.build();
        defer built.deinit();
        try built.saveToFile(sidecar);
    }

    fn runPrecache(self: *App, tab: *QueryTab, source_path: [:0]const u8, phrase: []const u8) void {
        self.mutex.lock(self.io) catch return;
        if (tab.query_cancel.load(.acquire)) {
            self.mutex.unlock(self.io);
            return;
        }
        tab.precache_phrase_len = @min(phrase.len, tab.precache_phrase.len);
        @memcpy(tab.precache_phrase[0..tab.precache_phrase_len], phrase[0..tab.precache_phrase_len]);
        if (self.isSelectedQuery(tab)) {
            self.video_overlay_prompts = .{};
            self.video_phrase_len = tab.precache_phrase_len;
            @memcpy(self.video_phrase[0..self.video_phrase_len], tab.precache_phrase[0..tab.precache_phrase_len]);
        }
        tab.precache_scanned_until = -1;
        tab.precache_active.store(true, .release);
        self.mutex.unlock(self.io);
        defer tab.precache_active.store(false, .release);
        self.refreshQueryTabs();
        defer self.refreshQueryTabs();
        const reader = sam_macos_video_open(source_path.ptr, 0) orelse {
            self.setQueryStatus(tab, "Could not decode video for pre-caching.");
            return;
        };
        defer sam_macos_video_close(reader);
        const duration = sam_macos_video_duration(reader);
        var frames: usize = 0;
        const started = std.Io.Timestamp.now(self.io, .awake);
        var last_ui_update = started;
        var last_log = started;
        var fraction: f64 = 0;

        var index_builder = vdb.index.IndexBuilder.init(self.allocator);
        defer index_builder.deinit();

        log.info(self.io, "pre-caching and indexing video for \"{s}\" ({d:.2} s)", .{ phrase, duration });
        while (!tab.query_cancel.load(.acquire)) {
            var frame: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
            const next = sam_macos_video_next(reader, &frame);
            if (next <= 0) {
                if (next < 0) {
                    self.setQueryStatus(tab, "Video decoding failed during pre-cache.");
                } else {
                    self.saveVideoIndex(tab, &index_builder) catch |err| {
                        log.info(self.io, "Could not save video index: {t}", .{err});
                        self.setQueryStatus(tab, "Could not save video index.");
                        return;
                    };
                    self.mutex.lock(self.io) catch return;
                    tab.precache_progress = 1;
                    tab.frames_processed = frames;
                    self.mutex.unlock(self.io);
                    self.refreshQueryTabs();
                    var status_buf: [200]u8 = undefined;
                    const status = std.fmt.bufPrintZ(&status_buf, "Pre-cached and indexed {d} frames for “{s}”. Saved to .vdb sidecar.", .{ frames, phrase }) catch "Video pre-cache complete.";
                    self.setQueryStatus(tab, status);
                    log.info(self.io, "pre-cached and indexed {d} frames for \"{s}\" in {f}", .{ frames, phrase, started.untilNow(self.io, .awake) });
                }
                return;
            }
            defer sam_macos_video_free_frame(&frame);
            if (frame.width <= 0 or frame.height <= 0 or frame.rgb == null) continue;
            const width: usize = @intCast(frame.width);
            const height: usize = @intCast(frame.height);
            const rgb_len = std.math.mul(usize, std.math.mul(usize, width, height) catch continue, 3) catch continue;
            const pixels = self.allocator.dupe(u8, frame.rgb.?[0..rgb_len]) catch {
                self.setQueryStatus(tab, "Out of memory during video pre-cache.");
                return;
            };
            var decoded = zigimg.Image.fromRawPixelsOwned(width, height, pixels, .rgb24) catch {
                self.allocator.free(pixels);
                self.setQueryStatus(tab, "Could not prepare video frame for pre-cache.");
                return;
            };
            defer decoded.deinit(self.allocator);
            const args = .{
                self.allocator,
                [_][]const u8{
                    sam3.assets.concept_vision_encoder.sha256,
                    sam3.assets.concept_text_encoder.sha256,
                    sam3.assets.concept_decoder.sha256,
                    sam3.assets.concept_tokenizer_json.sha256,
                    sam3.assets.concept_vision_encoder_data.sha256,
                    sam3.assets.concept_text_encoder_data.sha256,
                },
                sam3.RgbImage.fromImage(decoded),
                phrase,
                @as(f32, 0.5),
            };
            const frame_started = std.Io.Timestamp.now(self.io, .awake);
            self.model_mutex.lock(self.io) catch return;
            if (tab.query_cancel.load(.acquire)) {
                self.model_mutex.unlock(self.io);
                break;
            }
            const result = here.call(.computeQuery, args);
            self.model_mutex.unlock(self.io);
            const values = result catch |err| {
                log.info(self.io, "Video pre-cache failed at frame {d}: {t}: {s}", .{ frames, err, sam3.onnx.lastError() });
                self.setQueryStatus(tab, "Video pre-cache inference failed.");
                return;
            };
            defer self.allocator.free(values);

            if (values.len >= 4) {
                const count: usize = @as(u32, @bitCast(values[0]));
                if (count > 0) {
                    const obj_score = values[3];
                    const best_score = if (values.len >= 5) @max(obj_score, values[4]) else obj_score;
                    const pts_ms: u32 = if (std.math.isFinite(frame.pts_seconds) and frame.pts_seconds >= 0)
                        @intFromFloat(frame.pts_seconds * 1000.0)
                    else
                        @intCast(frames * 33);
                    index_builder.addDetection(
                        @intCast(frames),
                        pts_ms,
                        phrase,
                        best_score,
                        .{ .x = 0, .y = 0, .w = 1, .h = 1 },
                    ) catch {};
                }
            }
            frames += 1;
            if (std.math.isFinite(frame.pts_seconds)) {
                self.mutex.lock(self.io) catch return;
                tab.precache_scanned_until = @max(tab.precache_scanned_until, frame.pts_seconds);
                self.mutex.unlock(self.io);
            }
            if (duration > 0 and std.math.isFinite(frame.pts_seconds)) {
                fraction = @max(fraction, std.math.clamp(frame.pts_seconds / duration, 0, 0.9999));
            }
            const now = std.Io.Timestamp.now(self.io, .awake);
            if (frames == 1 or last_ui_update.untilNow(self.io, .awake).nanoseconds >= 100_000_000) {
                self.mutex.lock(self.io) catch return;
                tab.precache_progress = fraction;
                tab.frames_processed = frames;
                self.mutex.unlock(self.io);
                self.refreshQueryTabs();
                last_ui_update = now;
            }
            if (frames == 1 or last_log.untilNow(self.io, .awake).nanoseconds >= 1_000_000_000) {
                log.info(self.io, "pre-cache frame {d} at {d:.2}/{d:.2} s ({d:.2}%) in {f}", .{
                    frames, frame.pts_seconds, duration, fraction * 100, frame_started.untilNow(self.io, .awake),
                });
                last_log = now;
            }
        }
        if (tab.query_cancel.load(.acquire)) {
            self.setQueryStatus(tab, "Video pre-cache cancelled.");
            log.info(self.io, "pre-cache cancelled after {d} frames", .{frames});
        }
    }

    fn handleCanvasClick(self: *App, norm_x: f32, norm_y: f32, is_positive: c_int) void {
        if (self.video_active or self.is_busy or self.image == null or self.points_len >= max_points) return;

        self.mutex.lock(self.io) catch return;
        self.points[self.points_len] = .{
            .x = std.math.clamp(norm_x, 0.0, 1.0),
            .y = std.math.clamp(norm_y, 0.0, 1.0),
            .label = if (is_positive != 0) .positive else .negative,
        };
        self.points_len += 1;

        self.renderComposite(-1);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
        self.mutex.unlock(self.io);

        self.is_busy = true;
        sam_macos_set_busy(1);
        sam_macos_set_status("Segmenting… the first click on an image also runs the vision encoder.");

        const thread = std.Thread.spawn(.{}, runSegmentWorker, .{self}) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        thread.detach();
    }

    fn runSegmentWorker(self: *App) void {
        self.model_mutex.lock(self.io) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer self.model_mutex.unlock(self.io);
        const started = std.Io.Timestamp.now(self.io, .awake);

        var embedding = self.ensureEmbedding(false) catch |err| {
            log.info(self.io, "Vision encoder failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Vision encoder failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer embedding.deinit();

        const decode_started = std.Io.Timestamp.now(self.io, .awake);
        const masks = self.model.segment(&embedding, self.points[0..self.points_len]) catch |err| {
            log.info(self.io, "Decoder failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Segmentation failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const decode_elapsed = decode_started.untilNow(self.io, .awake);
        log.info(self.io, "{d} point(s) -> {d} masks in {f}", .{
            self.points_len,
            masks.count,
            decode_elapsed,
        });

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (self.coverages.len != masks.count) {
            self.allocator.free(self.coverages);
            self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
        }
        self.best_mask_idx = @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
        self.selected_mask = self.best_mask_idx;

        self.renderComposite(self.selected_mask);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));

        self.showMaskButtons(masks);

        const elapsed = started.untilNow(self.io, .awake);
        var status_buf: [128]u8 = undefined;
        const status = std.fmt.bufPrintZ(&status_buf, "{d} point(s) -> {d} masks in {f}", .{
            self.points_len,
            masks.count,
            elapsed,
        }) catch "Segmentation complete";
        sam_macos_set_status(status);

        self.is_busy = false;
        sam_macos_set_busy(0);
    }

    fn isQuerySql(text: []const u8) bool {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (std.ascii.startsWithIgnoreCase(trimmed, "SELECT")) return true;
        if (std.ascii.startsWithIgnoreCase(trimmed, "CREATE")) return true;
        if (std.ascii.startsWithIgnoreCase(trimmed, "WHERE")) return true;
        return false;
    }

    fn finishQueryStatus(self: *App, tab: *QueryTab) void {
        self.mutex.lock(self.io) catch return;
        const cancelling = std.mem.startsWith(u8, tab.status[0..tab.status_len], "Cancelling");
        self.mutex.unlock(self.io);
        if (cancelling) self.setQueryStatus(tab, "Query cancelled.");
    }

    fn setQueryStatus(self: *App, tab: *QueryTab, status: []const u8) void {
        self.mutex.lock(self.io) catch return;
        tab.status_len = @min(status.len, tab.status.len);
        @memcpy(tab.status[0..tab.status_len], status[0..tab.status_len]);
        if (self.selectedTab() == tab) {
            const message = self.allocator.dupeZ(u8, status) catch {
                self.mutex.unlock(self.io);
                return;
            };
            defer self.allocator.free(message);
            sam_macos_set_status(message);
        }
        self.mutex.unlock(self.io);
        self.refreshQueryTabs();
    }

    fn editQuery(self: *App, text: []const u8) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);
        if (self.selectedTab()) |tab| tab.setDraft(text) catch {};
    }

    fn newQuery(self: *App) void {
        if (self.is_busy) return;
        const tab = QueryTab.createDraft(self.allocator, "") catch return;
        self.mutex.lock(self.io) catch {
            tab.deinit();
            return;
        };
        self.query_tabs.append(self.allocator, tab) catch {
            self.mutex.unlock(self.io);
            tab.deinit();
            return;
        };
        const index = self.query_tabs.items.len - 1;
        self.mutex.unlock(self.io);
        self.selectQuery(index);
    }

    fn closeQuery(self: *App, index: usize) void {
        if (self.is_busy or index >= self.query_tabs.items.len) return;
        self.mutex.lock(self.io) catch return;
        const tab = self.query_tabs.items[index];
        self.closed_queries.append(self.allocator, tab) catch {
            self.mutex.unlock(self.io);
            return;
        };
        tab.query_cancel.store(true, .release);
        const was_selected = self.selected_query == index;
        _ = self.query_tabs.orderedRemove(index);
        self.selected_query = vdb.query_tab.selectionAfterClose(self.selected_query, index, self.query_tabs.items.len);
        const next = self.selected_query;
        if (was_selected) {
            self.video_overlay_prompts = .{};
            self.video_phrase_len = 0;
            self.video_seek_target = null;
            self.video_step_requested = false;
            if (self.masks) |*m| {
                m.deinit();
                self.masks = null;
            }
            self.selected_mask = -1;
            self.best_mask_idx = -1;
            self.renderComposite(-1);
        }
        self.mutex.unlock(self.io);
        if (was_selected) {
            if (next) |selected| {
                self.selectQuery(selected);
            } else {
                sam_macos_set_query_text("");
                sam_macos_set_status("Query closed.");
                sam_macos_set_masks(0, null, 0, -1);
                if (self.image) |img| sam_macos_set_image(self.frame.ptr, @intCast(img.width), @intCast(img.height));
            }
        }
        if (self.query_tabs.items.len == 0) self.newQuery();
        self.refreshQueryTabs();
        vdb.query_tab.reapClosed(&self.closed_queries);
    }

    fn selectQuery(self: *App, index: usize) void {
        if (self.is_busy or index >= self.query_tabs.items.len) return;
        self.mutex.lock(self.io) catch return;
        self.selected_query = index;
        const tab = self.query_tabs.items[index];
        self.video_overlay_prompts = .{};
        self.video_phrase_len = 0;
        self.mutex.unlock(self.io);
        if (tab.query_path.?.len > 0) {
            self.openVideoFromPath(tab.query_path.?);
        } else {
            self.stopVideo();
            self.mutex.lock(self.io) catch return;
            if (self.image) |*img| img.deinit(self.allocator);
            self.image = null;
            self.allocator.free(self.frame);
            self.frame = &.{};
            if (self.masks) |*m| m.deinit();
            self.masks = null;
            self.mutex.unlock(self.io);
            sam_macos_set_image(null, 0, 0);
        }
        const sql = self.allocator.dupeZ(u8, tab.draft) catch return;
        defer self.allocator.free(sql);
        sam_macos_set_query_text(sql);
        self.mutex.lock(self.io) catch return;
        const status = self.allocator.dupeZ(u8, tab.status[0..tab.status_len]) catch {
            self.mutex.unlock(self.io);
            return;
        };
        self.mutex.unlock(self.io);
        defer self.allocator.free(status);
        sam_macos_set_status(status);
        self.mutex.lock(self.io) catch return;
        if (tab.query_matches.items.len > 0) {
            tab.query_match_idx = @min(tab.query_match_idx, tab.query_matches.items.len - 1);
            self.video_seek_target = tab.matchTime(tab.query_match_idx);
            self.video_step_requested = true;
        }
        self.mutex.unlock(self.io);
        self.refreshQueryTabs();
    }

    fn refreshQueryTabs(self: *App) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);
        var labels: std.ArrayList(u8) = .empty;
        defer labels.deinit(self.allocator);
        for (self.query_tabs.items, 0..) |tab, i| {
            const state = if (!tab.has_run) "New" else if (tab.query_active.load(.acquire))
                if (tab.query_cancel.load(.acquire)) "Cancelling" else "Running"
            else if (tab.query_cancel.load(.acquire)) "Cancelled" else "Done";
            const label = std.fmt.allocPrint(self.allocator, "Query {d}: {s} ({d})", .{ i + 1, state, tab.query_matches.items.len }) catch return;
            defer self.allocator.free(label);
            if (i > 0) labels.append(self.allocator, '\n') catch return;
            labels.appendSlice(self.allocator, label) catch return;
        }
        const labels_z = self.allocator.dupeZ(u8, labels.items) catch return;
        defer self.allocator.free(labels_z);
        sam_macos_set_query_tabs(labels_z, if (self.selected_query) |index| @intCast(index) else -1);
        const selected = self.selectedTab();
        sam_macos_set_query_active(@intFromBool(if (selected) |tab| tab.query_active.load(.acquire) else false));
        if (selected) |tab| {
            sam_macos_set_precache_progress(@intFromBool(tab.precache_active.load(.acquire)), tab.precache_progress, tab.frames_processed);
        } else {
            sam_macos_set_precache_progress(0, 0, 0);
        }
    }

    fn selectedTab(self: *const App) ?*QueryTab {
        return self.query_tabs.items[self.selected_query orelse return null];
    }

    fn isQueryVideo(self: *const App) bool {
        return (self.selectedTab() orelse return false).matchesVideo(self.video_path);
    }

    fn isSelectedQuery(self: *const App, tab: *QueryTab) bool {
        return self.selectedTab() == tab and tab.matchesVideo(self.video_path);
    }

    fn queryMatches(self: *const App) []const u32 {
        if (!self.isQueryVideo()) return &.{};
        return self.selectedTab().?.query_matches.items;
    }

    fn selectedQueryActive(self: *const App) bool {
        return (self.selectedTab() orelse return false).query_active.load(.acquire);
    }

    fn restoreQueryOverlay(self: *App) void {
        if (!self.isQueryVideo()) return;
        const tab = self.selectedTab().?;
        self.video_overlay_prompts = tab.query_prompts;
        const phrase = if (tab.query_prompts.count > 0)
            tab.query_prompts.get(0)
        else
            tab.precache_phrase[0..tab.precache_phrase_len];
        self.video_phrase_len = phrase.len;
        @memcpy(self.video_phrase[0..phrase.len], phrase);
    }

    fn handleCancelQuery(self: *App) void {
        const tab = self.selectedTab() orelse return;
        if (tab.query_active.load(.acquire)) {
            tab.query_cancel.store(true, .release);
            self.setQueryStatus(tab, "Cancelling query…");
        }
    }

    fn handleClearQuery(self: *App) void {
        self.handleCancelQuery();
        self.mutex.lock(self.io) catch return;
        if (self.selectedTab()) |tab| {
            tab.query_prompts = .{};
            tab.precache_phrase_len = 0;
            tab.query_matches.clearRetainingCapacity();
            tab.query_match_pts.clearRetainingCapacity();
            tab.query_match_idx = 0;
        }
        self.video_overlay_prompts = .{};
        self.video_phrase_len = 0;
        if (self.masks) |*m| {
            m.deinit();
            self.masks = null;
        }
        self.selected_mask = -1;
        self.best_mask_idx = -1;
        self.renderComposite(-1);
        self.mutex.unlock(self.io);
        if (self.image) |img| sam_macos_set_image(self.frame.ptr, @intCast(img.width), @intCast(img.height));
        sam_macos_set_masks(0, null, 0, -1);
        sam_macos_set_status("Query cleared.");
    }

    fn resolveVideoPath(allocator: std.mem.Allocator, input_path: []const u8) ?[]const u8 {
        const trimmed = std.mem.trim(u8, input_path, " \t\r\n'\"");
        if (trimmed.len == 0) return null;

        // 1. Direct check
        if (allocator.dupeZ(u8, trimmed)) |zpath| {
            defer allocator.free(zpath);
            if (sam_macos_file_exists(zpath.ptr) != 0) {
                return allocator.dupe(u8, trimmed) catch null;
            }
        } else |_| {}

        // 2. Expand ~/
        const maybe_home = if (sam_macos_get_home()) |h| std.mem.span(h) else null;
        if (std.mem.startsWith(u8, trimmed, "~/")) {
            if (maybe_home) |home| {
                if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed[2..] })) |expanded| {
                    if (allocator.dupeZ(u8, expanded)) |zpath| {
                        defer allocator.free(zpath);
                        if (sam_macos_file_exists(zpath.ptr) != 0) {
                            return expanded;
                        }
                    } else |_| {}
                    allocator.free(expanded);
                } else |_| {}
            }
        }

        // 3. Check in home directory (~/<trimmed>)
        if (maybe_home) |home| {
            if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed })) |in_home| {
                if (allocator.dupeZ(u8, in_home)) |zpath| {
                    defer allocator.free(zpath);
                    if (sam_macos_file_exists(zpath.ptr) != 0) {
                        return in_home;
                    }
                } else |_| {}
                allocator.free(in_home);
            } else |_| {}
        }

        // 4. Check in current working directory (./<trimmed>)
        if (std.fmt.allocPrint(allocator, "./{s}", .{trimmed})) |in_cwd| {
            if (allocator.dupeZ(u8, in_cwd)) |zpath| {
                defer allocator.free(zpath);
                if (sam_macos_file_exists(zpath.ptr) != 0) {
                    return in_cwd;
                }
            } else |_| {}
            allocator.free(in_cwd);
        } else |_| {}

        return null;
    }

    fn handleQuery(self: *App, raw_query: []const u8) void {
        if (self.is_busy) return;
        const trimmed = std.mem.trim(u8, raw_query, " \t\r\n");

        // Attempt to extract source video path if specified in query
        var parsed_source_path: ?[]const u8 = null;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();

        var p = vdb.parser.Parser.init(arena.allocator(), trimmed);
        if (p.parse()) |stmt| {
            switch (stmt) {
                .select_stmt => |sel| {
                    switch (sel.source) {
                        .file_path => |fp| parsed_source_path = fp,
                        .call => |c| parsed_source_path = c.path,
                    }
                },
                .create_index => |ci| {
                    if (ci.source_file.len > 0) {
                        parsed_source_path = ci.source_file;
                    }
                },
            }
        } else |_| {}

        if (parsed_source_path == null) {
            if (self.selectedTab()) |tab| self.setQueryStatus(tab, "Include FROM 'video.mp4' in this tab's query.");
            return;
        }

        // If a video path is specified in the query, resolve and open it
        if (parsed_source_path) |source_file| {
            if (resolveVideoPath(self.allocator, source_file)) |resolved| {
                defer self.allocator.free(resolved);
                const is_same_video = if (self.video_active and self.video_path != null)
                    std.mem.eql(u8, self.video_path.?, resolved)
                else
                    false;

                if (!is_same_video) {
                    self.openVideoFromPath(resolved);
                }
            } else {
                var err_buf: [256]u8 = undefined;
                const err_msg = std.fmt.bufPrintZ(&err_buf, "Could not find video file: “{s}”", .{source_file}) catch "Video not found.";
                if (self.selectedTab()) |current| self.setQueryStatus(current, err_msg);
                return;
            }
        }

        if (!self.video_active or self.video_path == null) {
            if (self.selectedTab()) |current| self.setQueryStatus(current, "Could not load the video specified in FROM.");
            return;
        }
        const tab = QueryTab.create(self.allocator, trimmed, self.video_path.?) catch return;
        self.mutex.lock(self.io) catch {
            tab.deinit();
            return;
        };
        if (self.selected_query) |selected| {
            const old = self.query_tabs.items[selected];
            self.closed_queries.append(self.allocator, old) catch {
                self.mutex.unlock(self.io);
                tab.deinit();
                return;
            };
            old.query_cancel.store(true, .release);
            self.query_tabs.items[selected] = tab;
        } else {
            self.query_tabs.append(self.allocator, tab) catch {
                self.mutex.unlock(self.io);
                tab.deinit();
                return;
            };
            self.selected_query = self.query_tabs.items.len - 1;
        }
        self.video_overlay_prompts = .{};
        self.video_phrase_len = 0;
        self.mutex.unlock(self.io);
        self.setQueryStatus(tab, "Planning query in the background…");
        self.refreshQueryTabs();
        tab.query_thread = std.Thread.spawn(.{}, runQueryWorker, .{ self, tab }) catch {
            tab.query_active.store(false, .release);
            self.setQueryStatus(tab, "Could not start query worker.");
            self.refreshQueryTabs();
            return;
        };
    }

    fn runQueryWorker(self: *App, tab: *QueryTab) void {
        defer {
            tab.query_active.store(false, .release);
            self.finishQueryStatus(tab);
            self.refreshQueryTabs();
            tab.worker_finished.store(true, .release);
        }
        const query = tab.sql;
        if (tab.query_cancel.load(.acquire)) return;
        const source_path = tab.query_path.?;
        var database = vdb.Database.init(self.allocator);
        defer database.deinit();
        const sidecar = std.fmt.allocPrint(self.allocator, "{s}.vdb", .{source_path}) catch return;
        defer self.allocator.free(sidecar);
        {
            self.index_mutex.lock(self.io) catch return;
            defer self.index_mutex.unlock(self.io);
            if (vdb.index.InvertedIndex.loadFromFile(self.allocator, sidecar)) |loaded| {
                var loaded_idx = loaded;
                database.registerIndex(source_path, loaded_idx) catch {
                    loaded_idx.deinit();
                    return;
                };
            } else |_| {}
        }

        const started = std.Io.Timestamp.now(self.io, .awake);

        const trimmed = std.mem.trim(u8, query, " \t\r\n");

        if (std.ascii.startsWithIgnoreCase(trimmed, "CREATE")) {
            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            var p = vdb.parser.Parser.init(arena.allocator(), trimmed);
            if (p.parse()) |stmt| {
                if (stmt == .create_index) {
                    const target_prompt = stmt.create_index.prompt orelse stmt.create_index.model_name;
                    log.info(self.io, "Executing visual CREATE INDEX on concept \"{s}\"", .{target_prompt});
                    var status_buf: [160]u8 = undefined;
                    const status_msg = std.fmt.bufPrintZ(&status_buf, "Creating visual index for “{s}”…", .{target_prompt}) catch "Creating visual index…";
                    self.setQueryStatus(tab, status_msg);
                    self.runPrecache(tab, source_path, target_prompt);
                    return;
                }
            } else |err| {
                log.info(self.io, "Failed to parse CREATE INDEX: {t}", .{err});
                var err_buf: [160]u8 = undefined;
                const err_msg = std.fmt.bufPrintZ(&err_buf, "CREATE INDEX syntax error: {t}", .{err}) catch "Syntax error.";
                self.setQueryStatus(tab, err_msg);
                return;
            }
        }

        const final_query = vdb.query_input.normalize(self.allocator, trimmed, source_path) catch |err| {
            log.info(self.io, "Query normalization failed: {t}", .{err});
            var err_buf: [160]u8 = undefined;
            const err_msg = std.fmt.bufPrintZ(&err_buf, "Query syntax error: {t}", .{err}) catch "Query syntax error.";
            self.setQueryStatus(tab, err_msg);
            return;
        };
        defer self.allocator.free(final_query);

        log.info(self.io, "Executing visual query: {s}", .{final_query});

        const overlay_prompts = vdb.overlay.Prompts.fromSql(self.allocator, final_query) catch |err| {
            log.info(self.io, "Query overlay planning failed: {t}", .{err});
            self.setQueryStatus(tab, "Could not parse query overlay prompts.");
            return;
        };
        self.mutex.lock(self.io) catch return;
        if (tab.query_cancel.load(.acquire)) {
            self.mutex.unlock(self.io);
            return;
        }
        tab.query_prompts = overlay_prompts;
        if (self.isSelectedQuery(tab)) {
            self.video_overlay_prompts = overlay_prompts;
            self.video_phrase_len = 0;
            if (overlay_prompts.count > 0) {
                const prompt = overlay_prompts.get(0);
                self.video_phrase_len = prompt.len;
                @memcpy(self.video_phrase[0..prompt.len], prompt);
            }
        }
        self.mutex.unlock(self.io);

        const QueryStreamer = struct {
            app: *App,
            tab: *QueryTab,
            first_match_emitted: bool = false,
            last_ui_update: std.Io.Timestamp,

            pub fn onRow(streamer_ctx: *anyopaque, row: *const vdb.types.Row) anyerror!void {
                const streamer: *@This() = @ptrCast(@alignCast(streamer_ctx));
                const app = streamer.app;

                // Find frame in row
                var match_idx: ?usize = null;
                var match_pts: f64 = 0;
                for (row.values) |v| {
                    if (v == .frame_type) {
                        match_idx = v.frame_type.index;
                        match_pts = v.frame_type.pts_seconds;
                        break;
                    }
                }
                const idx = match_idx orelse return;

                app.mutex.lock(app.io) catch return;
                if (streamer.tab.query_cancel.load(.acquire)) {
                    app.mutex.unlock(app.io);
                    return error.QueryCancelled;
                }
                streamer.tab.appendMatch(@intCast(idx), match_pts) catch |err| {
                    app.mutex.unlock(app.io);
                    return err;
                };
                const total = streamer.tab.query_matches.items.len;

                if (!streamer.first_match_emitted) {
                    streamer.first_match_emitted = true;
                    if (app.isSelectedQuery(streamer.tab)) {
                        streamer.tab.query_match_idx = 0;
                        app.video_seek_target = match_pts;
                        app.video_step_requested = true;
                    }
                    app.mutex.unlock(app.io);

                    // INSTANT UI update for first frame!
                    var status_buf: [256]u8 = undefined;
                    const status_msg = std.fmt.bufPrintZ(&status_buf, "First match found instantly! Frame #{d} at {d:.2}s. Streaming visual query results…", .{
                        idx,
                        match_pts,
                    }) catch "First frame matched!";
                    app.setQueryStatus(streamer.tab, status_msg);
                    log.info(app.io, "Instantly streamed first match: Frame #{d} at {d:.2}s", .{ idx, match_pts });
                } else {
                    app.mutex.unlock(app.io);

                    // Periodically update UI with progress so user sees live match count
                    const now = std.Io.Timestamp.now(app.io, .awake);
                    if (streamer.last_ui_update.untilNow(app.io, .awake).nanoseconds > 200_000_000 or total % 50 == 0) {
                        streamer.last_ui_update = now;
                        var status_buf: [160]u8 = undefined;
                        const status_msg = std.fmt.bufPrintZ(&status_buf, "Streaming query: {d} matches found… scanning… Press Cancel Query to stop.", .{total}) catch "Streaming query…";
                        app.setQueryStatus(streamer.tab, status_msg);
                    }
                }
            }
        };

        var streamer = QueryStreamer{
            .app = self,
            .tab = tab,
            .last_ui_update = started,
        };
        const stream_cb: vdb.engine.RowCallback = .{
            .ctx = &streamer,
            .onRow = QueryStreamer.onRow,
        };

        var video_adapter = MacosVideoReader.init(source_path);
        defer video_adapter.deinit();

        database.engine_inst.sam3 = .{
            .ptr = self,
            .segmentFn = sam3SegmentBridge,
        };

        var result = database.executeQuery(final_query, video_adapter.asReader(), &tab.query_cancel, stream_cb) catch |err| {
            if (err == error.QueryCancelled) {
                log.info(self.io, "Visual database query cancelled.", .{});
                self.mutex.lock(self.io) catch return;
                const matches_so_far = tab.query_matches.items.len;
                self.mutex.unlock(self.io);
                var cancel_buf: [160]u8 = undefined;
                const cancel_msg = if (matches_so_far > 0)
                    std.fmt.bufPrintZ(&cancel_buf, "Query cancelled. Kept {d} matches found so far.", .{matches_so_far}) catch "Query cancelled."
                else
                    "Visual query cancelled.";
                self.setQueryStatus(tab, cancel_msg);
                return;
            }
            log.info(self.io, "Query execution error: {t}", .{err});
            var err_buf: [160]u8 = undefined;
            const err_msg = std.fmt.bufPrintZ(&err_buf, "Query error: {t}", .{err}) catch "Query execution failed.";
            self.setQueryStatus(tab, err_msg);
            return;
        };
        defer result.deinit();

        const elapsed = started.untilNow(self.io, .awake);

        self.mutex.lock(self.io) catch return;
        const total_matches = tab.query_matches.items.len;
        const first_frame = if (total_matches > 0) tab.query_matches.items[0] else 0;
        const first_frame_target = if (total_matches > 0) tab.matchTime(0) else null;
        self.mutex.unlock(self.io);

        if (total_matches > 0) {
            var status_buf: [256]u8 = undefined;
            const first_sec = first_frame_target.?;
            const status_msg = std.fmt.bufPrintZ(&status_buf, "Query complete: {d} match(es) in {f}. First match: Frame #{d} at {d:.2}s.", .{
                total_matches,
                elapsed,
                first_frame,
                first_sec,
            }) catch "Query completed.";
            self.setQueryStatus(tab, status_msg);
        } else {
            var status_buf: [160]u8 = undefined;
            const status_msg = std.fmt.bufPrintZ(&status_buf, "Query returned 0 matching frames in {f}.", .{elapsed}) catch "0 matches.";
            self.setQueryStatus(tab, status_msg);
        }
    }

    fn handleFindText(self: *App, text: [*:0]const u8) void {
        self.handleQuery(std.mem.span(text));
    }

    fn runLookupWorker(ctx: anytype) void {
        defer {
            ctx.app.allocator.free(ctx.phrase);
            ctx.app.allocator.destroy(ctx);
        }
        const self = ctx.app;
        const phrase = ctx.phrase;
        self.model_mutex.lock(self.io) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer self.model_mutex.unlock(self.io);
        const started = std.Io.Timestamp.now(self.io, .awake);

        const lookup_started = std.Io.Timestamp.now(self.io, .awake);
        const values = here.call(.computeQuery, .{
            self.allocator,
            [_][]const u8{
                sam3.assets.concept_vision_encoder.sha256,
                sam3.assets.concept_text_encoder.sha256,
                sam3.assets.concept_decoder.sha256,
                sam3.assets.concept_tokenizer_json.sha256,
                sam3.assets.concept_vision_encoder_data.sha256,
                sam3.assets.concept_text_encoder_data.sha256,
            },
            sam3.RgbImage.fromImage(self.image.?),
            phrase,
            @as(f32, 0.5),
        }) catch |err| {
            log.info(self.io, "Text lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer self.allocator.free(values);
        const masks = unpackMasks(self.allocator, values) catch |err| {
            log.info(self.io, "Cached text lookup failed: {t}", .{err});
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const lookup_elapsed = lookup_started.untilNow(self.io, .awake);
        log.info(self.io, "\"{s}\" -> {d} object(s) in {f}", .{
            phrase,
            masks.count,
            lookup_elapsed,
        });

        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        self.masks = masks;

        if (masks.count == 0) {
            self.selected_mask = -1;
            self.best_mask_idx = -1;
            self.renderComposite(-1);
            sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
            sam_macos_set_masks(0, null, 0, -1);

            var msg_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrintZ(&msg_buf, "No objects matched “{s}”.", .{phrase}) catch "No objects matched.";
            sam_macos_set_status(msg);
        } else {
            if (self.coverages.len != masks.count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
            }
            self.best_mask_idx = @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
            self.selected_mask = self.best_mask_idx;

            self.renderComposite(self.selected_mask);
            sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));

            self.showMaskButtons(masks);

            const elapsed = started.untilNow(self.io, .awake);
            var status_buf: [256]u8 = undefined;
            const status = std.fmt.bufPrintZ(&status_buf, "{d} object(s) matched “{s}” in {f}", .{
                masks.count,
                phrase,
                elapsed,
            }) catch "Search complete";
            sam_macos_set_status(status);
        }

        self.is_busy = false;
        sam_macos_set_busy(0);
    }

    fn handleSelectMask(self: *App, mask_index: c_int) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        self.selected_mask = mask_index;
        self.renderComposite(self.selected_mask);
        sam_macos_set_image(self.frame.ptr, @intCast(self.image.?.width), @intCast(self.image.?.height));
    }

    fn handleClearPoints(self: *App) void {
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        self.points_len = 0;
        if (self.masks) |*m| {
            m.deinit();
            self.masks = null;
        }
        self.selected_mask = -1;
        self.best_mask_idx = -1;
        self.renderComposite(-1);
        if (self.image) |img| {
            sam_macos_set_image(self.frame.ptr, @intCast(img.width), @intCast(img.height));
        }
        sam_macos_set_masks(0, null, 0, -1);
        sam_macos_set_status("Points cleared.");
    }

    fn ensureEmbedding(self: *App, comptime concept: bool) !if (concept) sam3.TextImageEmbedding else sam3.PointEmbedding {
        const img = self.image orelse return error.NoImageLoaded;
        const started = std.Io.Timestamp.now(self.io, .awake);
        const rgb = sam3.RgbImage.fromImage(img);
        const embedding = if (concept) try self.model.encodeForText(rgb) else try self.model.encodePoints(rgb);
        log.info(self.io, "{s}encoded {d}x{d} in {f}", .{
            if (concept) "concept-" else "",
            img.width,
            img.height,
            started.untilNow(self.io, .awake),
        });
        return embedding;
    }

    fn renderComposite(self: *App, mask_index: i32) void {
        const img = self.image orelse return;
        if (mask_index < 0 or self.masks == null) {
            render.compositeRgba(self.allocator, img, self.frame, null, 0, 0, self.points[0..self.points_len]);
            return;
        }

        render.compositeRgba(self.allocator, img, self.frame, null, 0, 0, &.{});
        const masks = self.masks.?;
        const selected: usize = @intCast(mask_index);
        for (0..masks.count) |i| {
            if (i != selected) self.overlayMask(img, masks, i, 0.25);
        }
        if (selected < masks.count) self.overlayMask(img, masks, selected, 0.5);
        self.drawPointMarkers(img);
    }

    fn showMaskButtons(self: *App, masks: sam3.Masks) void {
        const infos = self.allocator.alloc(SamMaskInfo, masks.count) catch return;
        defer self.allocator.free(infos);
        for (infos, 0..) |*info, i| {
            const color = maskColor(i);
            info.* = .{
                .score = masks.scores[i],
                .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
                .red = color.red,
                .green = color.green,
                .blue = color.blue,
            };
        }
        sam_macos_set_masks(@intCast(infos.len), infos.ptr, self.best_mask_idx, self.selected_mask);
    }

    fn overlayMask(self: *App, img: zigimg.Image, masks: sam3.Masks, index: usize, alpha: f32) void {
        const plane = masks.plane(index);
        const color = maskColor(index);
        const ratio_x = @as(f32, @floatFromInt(masks.width)) / @as(f32, @floatFromInt(img.width));
        const ratio_y = @as(f32, @floatFromInt(masks.height)) / @as(f32, @floatFromInt(img.height));
        for (0..img.height) |y| {
            const source_y = ratio_y * (@as(f32, @floatFromInt(y)) + 0.5) - 0.5;
            const y0 = clampMaskIndex(source_y, masks.height);
            const y1 = @min(y0 + 1, masks.height - 1);
            const wy = @max(0.0, source_y - @as(f32, @floatFromInt(y0)));
            for (0..img.width) |x| {
                const source_x = ratio_x * (@as(f32, @floatFromInt(x)) + 0.5) - 0.5;
                const x0 = clampMaskIndex(source_x, masks.width);
                const x1 = @min(x0 + 1, masks.width - 1);
                const wx = @max(0.0, source_x - @as(f32, @floatFromInt(x0)));
                const top = std.math.lerp(plane[y0 * masks.width + x0], plane[y0 * masks.width + x1], wx);
                const bottom = std.math.lerp(plane[y1 * masks.width + x0], plane[y1 * masks.width + x1], wx);
                if (std.math.lerp(top, bottom, wy) <= 0) continue;
                const offset = (y * img.width + x) * 4;
                self.frame[offset] = blendChannel(self.frame[offset], color.red, alpha);
                self.frame[offset + 1] = blendChannel(self.frame[offset + 1], color.green, alpha);
                self.frame[offset + 2] = blendChannel(self.frame[offset + 2], color.blue, alpha);
            }
        }
    }

    fn drawPointMarkers(self: *App, img: zigimg.Image) void {
        for (self.points[0..self.points_len]) |point| {
            const center_x: isize = @intFromFloat(point.x * @as(f32, @floatFromInt(img.width)));
            const center_y: isize = @intFromFloat(point.y * @as(f32, @floatFromInt(img.height)));
            const color: MaskColor = if (point.label == .positive)
                .{ .red = 0, .green = 255, .blue = 0 }
            else
                .{ .red = 255, .green = 0, .blue = 0 };
            for (0..15) |row| {
                for (0..15) |col| {
                    const dx: isize = @as(isize, @intCast(col)) - 7;
                    const dy: isize = @as(isize, @intCast(row)) - 7;
                    if (dx * dx + dy * dy > 49) continue;
                    const x = center_x + dx;
                    const y = center_y + dy;
                    if (x < 0 or y < 0 or x >= img.width or y >= img.height) continue;
                    const offset = (@as(usize, @intCast(y)) * img.width + @as(usize, @intCast(x))) * 4;
                    self.frame[offset] = color.red;
                    self.frame[offset + 1] = color.green;
                    self.frame[offset + 2] = color.blue;
                }
            }
        }
    }
};

fn clampMaskIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0) return 0;
    return @min(@as(usize, @intFromFloat(@floor(coordinate))), limit - 1);
}

fn blendChannel(original: u8, tint: u8, alpha: f32) u8 {
    return @intFromFloat(@as(f32, @floatFromInt(original)) * (1 - alpha) + @as(f32, @floatFromInt(tint)) * alpha);
}

pub fn computeQuery(
    allocator: std.mem.Allocator,
    model_ids: [6][]const u8,
    image: sam3.RgbImage,
    phrase: []const u8,
    min_score: f32,
) ![]f32 {
    const model = (g_app orelse return error.NoActiveApp).model;
    const text_features = try here.call(.computeTextFeatures, .{
        allocator,
        [_][]const u8{
            model_ids[1],
            model_ids[5],
            model_ids[3],
        },
        phrase,
    });
    defer allocator.free(text_features);
    var embedding = try model.encodeForText(image);
    defer embedding.deinit();
    var masks = try model.findWithTextFeatures(&embedding, phrase, text_features, .{ .min_score = min_score });
    defer masks.deinit();

    const header_and_scores = try std.math.add(usize, 4, masks.scores.len);
    const len = try std.math.add(usize, header_and_scores, masks.logits.len);
    const values = try allocator.alloc(f32, len);
    values[0] = @bitCast(@as(u32, @intCast(masks.count)));
    values[1] = @bitCast(@as(u32, @intCast(masks.width)));
    values[2] = @bitCast(@as(u32, @intCast(masks.height)));
    values[3] = masks.object_score;
    @memcpy(values[4..][0..masks.scores.len], masks.scores);
    @memcpy(values[header_and_scores..], masks.logits);
    return values;
}

pub fn computeTextFeatures(
    allocator: std.mem.Allocator,
    model_ids: [3][]const u8,
    phrase: []const u8,
) ![]f32 {
    _ = allocator;
    _ = model_ids;
    const model = (g_app orelse return error.NoActiveApp).model;
    return model.encodeTextFeatures(phrase);
}

fn unpackMasks(allocator: std.mem.Allocator, values: []const f32) !sam3.Masks {
    if (values.len < 4) return error.InvalidCachedQuery;
    const count: usize = @as(u32, @bitCast(values[0]));
    const width: usize = @as(u32, @bitCast(values[1]));
    const height: usize = @as(u32, @bitCast(values[2]));
    if (width == 0 or height == 0) return error.InvalidCachedQuery;
    const pixels = std.math.mul(usize, width, height) catch return error.InvalidCachedQuery;
    const logits_len = std.math.mul(usize, count, pixels) catch return error.InvalidCachedQuery;
    const header_and_scores = std.math.add(usize, 4, count) catch return error.InvalidCachedQuery;
    const expected = std.math.add(usize, header_and_scores, logits_len) catch return error.InvalidCachedQuery;
    if (values.len != expected) return error.InvalidCachedQuery;

    const scores = try allocator.dupe(f32, values[4..header_and_scores]);
    errdefer allocator.free(scores);
    const logits = try allocator.dupe(f32, values[header_and_scores..]);
    return .{
        .allocator = allocator,
        .scores = scores,
        .logits = logits,
        .count = count,
        .width = width,
        .height = height,
        .object_score = values[3],
    };
}

var g_app: ?*App = null;

fn cOpenFile(path: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| {
        app.openImageFromPath(std.mem.span(path));
    }
}

fn cOpenVideo(path: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| app.openVideoFromPath(std.mem.span(path));
}

fn cVideoPlayPause() callconv(.c) void {
    if (g_app) |app| app.handleVideoPlayPause();
}

fn cVideoSeek(seconds: f64) callconv(.c) void {
    if (g_app) |app| app.handleVideoSeek(seconds);
}

fn cVideoStep() callconv(.c) void {
    if (g_app) |app| app.handleVideoStep();
}

fn cSampleClick() callconv(.c) void {
    if (g_app) |app| {
        app.openImageFromPath(app.example_path);
    }
}

fn cModeChange(mode: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.click_mode_add = (mode == 1);
    }
}

fn cClearPoints() callconv(.c) void {
    if (g_app) |app| {
        app.handleClearPoints();
    }
}

fn cFindText(text: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| {
        app.handleFindText(text);
    }
}

fn cSelectQuery(index: usize) callconv(.c) void {
    if (g_app) |app| app.selectQuery(index);
}

fn cNewQuery() callconv(.c) void {
    if (g_app) |app| app.newQuery();
}

fn cEditQuery(text: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| app.editQuery(std.mem.span(text));
}

fn cCloseQuery(index: usize) callconv(.c) void {
    if (g_app) |app| app.closeQuery(index);
}

fn cReapQueries() callconv(.c) void {
    if (g_app) |app| vdb.query_tab.reapClosed(&app.closed_queries);
}

fn cCancelQuery() callconv(.c) void {
    if (g_app) |app| {
        app.handleCancelQuery();
    }
}

fn cClearQuery() callconv(.c) void {
    if (g_app) |app| {
        app.handleClearQuery();
    }
}

fn cCanvasClick(norm_x: f32, norm_y: f32, is_positive: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.handleCanvasClick(norm_x, norm_y, is_positive);
    }
}

fn cSelectMask(mask_index: c_int) callconv(.c) void {
    if (g_app) |app| {
        app.handleSelectMask(mask_index);
    }
}
