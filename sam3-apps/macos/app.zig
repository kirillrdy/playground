const std = @import("std");
const sam3 = @import("sam3");
const render = sam3.render;
const zigimg = @import("zigimg");
const zimo = @import("zimo");
const log = @import("log");
const vdb = @import("vdb");
const here = zimo.bind(@This(), @embedFile("app.zig"));

const SamCallbacks = extern struct {
    on_open_file: ?*const fn (path: [*:0]const u8) callconv(.c) void,
    on_open_video: ?*const fn (path: [*:0]const u8) callconv(.c) void,
    on_video_play_pause: ?*const fn () callconv(.c) void,
    on_video_seek: ?*const fn (seconds: f64) callconv(.c) void,
    on_video_step: ?*const fn () callconv(.c) void,
    on_precache_video: ?*const fn (text: [*:0]const u8) callconv(.c) void,
    on_sample_click: ?*const fn () callconv(.c) void,
    on_mode_change: ?*const fn (mode: c_int) callconv(.c) void,
    on_clear_points: ?*const fn () callconv(.c) void,
    on_find_text: ?*const fn (text: [*:0]const u8) callconv(.c) void,
    on_cancel_query: ?*const fn () callconv(.c) void,
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
extern fn sam_macos_set_status(text: [*:0]const u8) void;
extern fn sam_macos_set_image(rgba_pixels: ?[*]const u8, width: c_int, height: c_int) void;
extern fn sam_macos_set_masks(count: c_int, masks: ?[*]const SamMaskInfo, best_index: c_int, selected_index: c_int) void;
extern fn sam_macos_set_busy(is_busy: c_int) void;
extern fn sam_macos_set_video_mode(active: c_int, playing: c_int) void;
extern fn sam_macos_set_video_timeline(duration: f64, position: f64) void;
extern fn sam_macos_set_precache_progress(state: c_int, fraction: f64, frames: usize) void;
extern fn sam_macos_set_query_active(active: c_int) void;
extern fn sam_macos_video_open(path: [*:0]const u8, start_seconds: f64) ?*anyopaque;
extern fn sam_macos_video_duration(reader: *anyopaque) f64;
extern fn sam_macos_video_next(reader: *anyopaque, frame: *VideoFrame) c_int;
extern fn sam_macos_video_free_frame(frame: *VideoFrame) void;
extern fn sam_macos_video_close(reader: *anyopaque) void;
extern fn sam_macos_dispatch_main(func: *const fn (?*anyopaque) callconv(.c) void, ctx: ?*anyopaque) void;

const MacosVideoReader = struct {
    path: [:0]const u8,
    duration: f64,
    reader: ?*anyopaque = null,
    current_index: usize = 0,
    current_frame: ?VideoFrame = null,

    pub fn init(path: [:0]const u8) MacosVideoReader {
        const handle = sam_macos_video_open(path.ptr, 0);
        const dur = if (handle) |h| sam_macos_video_duration(h) else 0;
        return .{
            .path = path,
            .duration = dur,
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
        return @intFromFloat(@max(self.duration * 30.0, 1.0));
    }

    fn seekToFrameImpl(ctx: *anyopaque, frame_idx: usize) anyerror!void {
        const self: *MacosVideoReader = @ptrCast(@alignCast(ctx));
        if (self.current_frame) |*vf| {
            sam_macos_video_free_frame(vf);
            self.current_frame = null;
        }
        if (self.reader) |r| sam_macos_video_close(r);
        const pts = @as(f64, @floatFromInt(frame_idx)) * 0.0333;
        self.reader = sam_macos_video_open(self.path.ptr, pts);
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
    video_duration: f64 = 0,
    video_position: f64 = 0,
    video_previewed: bool = false,
    video_seek_target: ?f64 = null,
    video_step_requested: bool = false,
    precache_thread: ?std.Thread = null,
    precache_active: std.atomic.Value(bool) = .init(false),
    precache_cancel: std.atomic.Value(bool) = .init(false),
    precache_phrase: [256]u8 = undefined,
    precache_phrase_len: usize = 0,
    precache_scanned_until: f64 = -1,

    vdb_database: vdb.Database,
    query_matches: std.ArrayList(u32) = .empty,
    query_match_idx: usize = 0,
    query_thread: ?std.Thread = null,
    query_active: std.atomic.Value(bool) = .init(false),
    query_cancel: std.atomic.Value(bool) = .init(false),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) App {
        return .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
            .vdb_database = vdb.Database.init(allocator),
        };
    }

    pub fn deinit(self: *App) void {
        self.stopVideo();
        self.vdb_database.deinit();
        self.query_matches.deinit(self.allocator);
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
            .on_precache_video = &cPrecacheVideo,
            .on_sample_click = &cSampleClick,
            .on_mode_change = &cModeChange,
            .on_clear_points = &cClearPoints,
            .on_find_text = &cFindText,
            .on_cancel_query = &cCancelQuery,
            .on_canvas_click = &cCanvasClick,
            .on_select_mask = &cSelectMask,
        };

        if (sam_macos_init(&callbacks) != 0) {
            return error.MacosUiInitFailed;
        }

        // Open sample image by default to give user an instant experience
        self.openImageFromPath(self.example_path);

        sam_macos_run();
    }

    fn openImageFromPath(self: *App, path: []const u8) void {
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
        self.precache_cancel.store(true, .release);
        self.query_cancel.store(true, .release);
        if (self.video_thread) |thread| thread.join();
        if (self.precache_thread) |thread| thread.join();
        if (self.query_thread) |thread| thread.join();
        self.video_thread = null;
        self.precache_thread = null;
        self.query_thread = null;
        if (self.video_path) |path| self.allocator.free(path);
        self.video_path = null;
        self.video_active = false;
        self.video_phrase_len = 0;
        self.video_playing.store(false, .release);
        self.video_duration = 0;
        self.video_position = 0;
        self.video_previewed = false;
        self.video_seek_target = null;
        self.video_step_requested = false;
        self.precache_active.store(false, .release);
        self.precache_phrase_len = 0;
        self.precache_scanned_until = -1;
        self.query_active.store(false, .release);
        self.query_matches.clearRetainingCapacity();
        self.query_match_idx = 0;
        sam_macos_set_video_mode(0, 0);
        sam_macos_set_video_timeline(0, 0);
        sam_macos_set_precache_progress(0, 0, 0);
        sam_macos_set_query_active(0);
    }

    fn openVideoFromPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        self.stopVideo();
        self.video_path = self.allocator.dupeZ(u8, path) catch {
            sam_macos_set_status("Out of memory opening video.");
            return;
        };
        self.video_stop.store(false, .release);
        self.precache_cancel.store(false, .release);
        self.video_active = true;
        self.mutex.lock(self.io) catch return;
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
        self.mutex.unlock(self.io);
        sam_macos_set_image(null, 0, 0);
        sam_macos_set_masks(0, null, 0, -1);
        sam_macos_set_video_mode(1, 0);

        // Try auto-loading sidecar index if exists
        var index_loaded = false;
        var indexed_concepts_count: usize = 0;
        if (std.fmt.allocPrint(self.allocator, "{s}.vdb", .{path})) |sidecar| {
            defer self.allocator.free(sidecar);
            if (vdb.index.InvertedIndex.loadFromFile(self.allocator, sidecar)) |loaded_idx| {
                indexed_concepts_count = loaded_idx.classes.count();
                self.vdb_database.registerIndex(path, loaded_idx) catch |err| {
                    log.info(self.io, "Failed to register loaded index: {t}", .{err});
                };
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
            self.allocator.free(self.video_path.?);
            self.video_path = null;
            self.video_active = false;
            sam_macos_set_video_mode(0, 0);
            sam_macos_set_status("Could not start video playback.");
            return;
        };
    }

    fn handleVideoPlayPause(self: *App) void {
        if (!self.video_active) return;
        self.mutex.lock(self.io) catch return;
        const has_phrase = self.video_phrase_len > 0;
        self.mutex.unlock(self.io);
        if (!has_phrase) {
            sam_macos_set_status("Enter a word and press Find before playing video.");
            return;
        }
        const playing = !self.video_playing.load(.acquire);
        self.video_playing.store(playing, .release);
        sam_macos_set_video_mode(1, @intFromBool(playing));
        if (!playing) sam_macos_set_status("Video paused.");
    }

    fn handleVideoSeek(self: *App, seconds: f64) void {
        if (!self.video_active or !std.math.isFinite(seconds)) return;
        self.mutex.lock(self.io) catch return;
        if (seconds == 0 and self.query_matches.items.len > 0) {
            self.query_match_idx = 0;
            const target_frame = self.query_matches.items[0];
            const target_sec = @as(f64, @floatFromInt(target_frame)) * 0.0333;
            self.video_seek_target = target_sec;
            self.video_step_requested = true;
            const total = self.query_matches.items.len;
            self.mutex.unlock(self.io);
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrintZ(&buf, "Restarted at query match 1 of {d} (Frame #{d} at {d:.2}s).", .{ total, target_frame, target_sec }) catch "Query match 1.";
            sam_macos_set_status(msg);
            return;
        }
        const target = std.math.clamp(seconds, 0, self.video_duration);
        const duration = self.video_duration;
        self.video_seek_target = target;
        self.video_step_requested = self.video_phrase_len > 0;
        self.mutex.unlock(self.io);
        sam_macos_set_video_timeline(duration, target);
    }

    fn handleVideoStep(self: *App) void {
        if (!self.video_active or self.video_playing.load(.acquire)) return;
        self.mutex.lock(self.io) catch return;
        if (self.query_matches.items.len > 0) {
            self.query_match_idx = (self.query_match_idx + 1) % self.query_matches.items.len;
            const target_frame = self.query_matches.items[self.query_match_idx];
            const target_sec = @as(f64, @floatFromInt(target_frame)) * 0.0333;
            self.video_seek_target = target_sec;
            self.video_step_requested = true;
            const match_num = self.query_match_idx + 1;
            const total = self.query_matches.items.len;
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

    fn handlePrecacheVideo(self: *App, text: [*:0]const u8) void {
        if (!self.video_active) return;
        if (self.precache_active.load(.acquire)) {
            self.precache_cancel.store(true, .release);
            sam_macos_set_status("Cancelling pre-cache after this frame…");
            return;
        }
        if (self.precache_thread) |thread| {
            thread.join();
            self.precache_thread = null;
        }
        const phrase = std.mem.span(text);
        if (phrase.len == 0) {
            sam_macos_set_status("Enter a word to pre-cache this video.");
            return;
        }
        self.mutex.lock(self.io) catch return;
        self.precache_phrase_len = @min(phrase.len, self.precache_phrase.len);
        @memcpy(self.precache_phrase[0..self.precache_phrase_len], phrase[0..self.precache_phrase_len]);
        self.video_phrase_len = self.precache_phrase_len;
        @memcpy(self.video_phrase[0..self.video_phrase_len], self.precache_phrase[0..self.precache_phrase_len]);
        self.precache_cancel.store(false, .release);
        self.precache_scanned_until = -1;
        self.precache_active.store(true, .release);
        self.mutex.unlock(self.io);
        sam_macos_set_precache_progress(1, 0, 0);
        sam_macos_set_status("Pre-caching video frames…");
        self.precache_thread = std.Thread.spawn(.{}, runPrecacheWorker, .{self}) catch {
            self.precache_active.store(false, .release);
            sam_macos_set_precache_progress(0, 0, 0);
            sam_macos_set_status("Could not start video pre-cache.");
            return;
        };
    }

    fn runPrecacheWorker(self: *App) void {
        self.runPrecache(self.precache_phrase[0..self.precache_phrase_len]);
        self.precache_active.store(false, .release);
        sam_macos_set_precache_progress(2, 0, 0);
        if (!self.video_stop.load(.acquire) and !self.video_playing.load(.acquire)) {
            self.mutex.lock(self.io) catch return;
            self.video_seek_target = self.video_position;
            self.video_step_requested = true;
            self.mutex.unlock(self.io);
        }
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
            const phrase_len = self.video_phrase_len;
            @memcpy(phrase_buf[0..phrase_len], self.video_phrase[0..phrase_len]);
            self.mutex.unlock(self.io);
            if (phrase_len == 0) continue;
            const phrase = phrase_buf[0..phrase_len];

            const started = std.Io.Timestamp.now(self.io, .awake);
            const maybe_values = self.videoQuery(sam3.RgbImage.fromImage(decoded), phrase, video_frame.pts_seconds) catch |err| {
                log.info(self.io, "Video frame lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
                sam_macos_set_status("Video frame inference failed.");
                self.video_playing.store(false, .release);
                sam_macos_set_video_mode(1, 0);
                continue;
            };
            defer if (maybe_values) |values| self.allocator.free(values);
            var masks: ?sam3.Masks = null;
            if (maybe_values) |values| {
                masks = unpackMasks(self.allocator, values) catch blk: {
                    // The pre-cache worker may still be writing this entry.
                    if (self.precache_active.load(.acquire)) break :blk null;
                    sam_macos_set_status("Cached video frame is invalid.");
                    continue;
                };
            }
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
            const status = if (self.masks != null)
                std.fmt.bufPrintZ(&status_buf, "At {s}: {d} match(es) for “{s}” in {f}", .{
                    timecode, mask_count, phrase, lookup_elapsed,
                }) catch "Video frame processed."
            else
                std.fmt.bufPrintZ(&status_buf, "At {s}: waiting for cached “{s}” result.", .{ timecode, phrase }) catch "Frame shown without a cached result.";
            sam_macos_set_status(status);
            if (self.masks != null) log.info(self.io, "at {s}: \"{s}\" -> {d} object(s) in {f}", .{
                timecode, phrase, mask_count, lookup_elapsed,
            });
        }
    }

    fn videoQuery(self: *App, image: sam3.RgbImage, phrase: []const u8, pts: f64) !?[]f32 {
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
        if (self.precache_active.load(.acquire)) {
            self.mutex.lock(self.io) catch return error.VideoLockFailed;
            const scanned_until = self.precache_scanned_until;
            const scanned_phrase = std.mem.eql(u8, phrase, self.precache_phrase[0..self.precache_phrase_len]);
            self.mutex.unlock(self.io);
            if (!scanned_phrase or !std.math.isFinite(pts) or pts > scanned_until) return null;
            return try here.call(.computeQuery, args);
        }
        self.model_mutex.lock(self.io) catch return error.ModelLockFailed;
        defer self.model_mutex.unlock(self.io);
        return try here.call(.computeQuery, args);
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
        const maybe_vals = try self.videoQuery(rgb_img, prompt, frame.pts_seconds);
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

    fn runPrecache(self: *App, phrase: []const u8) void {
        const reader = sam_macos_video_open(self.video_path.?.ptr, 0) orelse {
            sam_macos_set_status("Could not decode video for pre-caching.");
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

        if (self.vdb_database.getIndex(self.video_path.?)) |existing| {
            index_builder.importIndex(existing) catch |err| {
                log.info(self.io, "Could not import existing index: {t}", .{err});
            };
        }

        log.info(self.io, "pre-caching and indexing video for \"{s}\" ({d:.2} s)", .{ phrase, duration });
        while (!self.video_stop.load(.acquire) and !self.precache_cancel.load(.acquire)) {
            var frame: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
            const next = sam_macos_video_next(reader, &frame);
            if (next <= 0) {
                if (next < 0) {
                    sam_macos_set_status("Video decoding failed during pre-cache.");
                } else {
                    if (index_builder.build()) |built_idx| {
                        self.vdb_database.registerIndex(self.video_path.?, built_idx) catch |err| {
                            log.info(self.io, "Failed to register index in database: {t}", .{err});
                        };
                        if (std.fmt.allocPrint(self.allocator, "{s}.vdb", .{self.video_path.?})) |sidecar| {
                            defer self.allocator.free(sidecar);
                            if (self.vdb_database.getIndex(self.video_path.?)) |reg_idx| {
                                reg_idx.saveToFile(sidecar) catch |err| {
                                    log.info(self.io, "Failed to save index to {s}: {t}", .{ sidecar, err });
                                };
                            }
                        } else |_| {}
                    } else |err| {
                        log.info(self.io, "Failed to build inverted index: {t}", .{err});
                    }
                    sam_macos_set_precache_progress(1, 1, frames);
                    var status_buf: [200]u8 = undefined;
                    const status = std.fmt.bufPrintZ(&status_buf, "Pre-cached and indexed {d} frames for “{s}”. Saved to .vdb sidecar.", .{ frames, phrase }) catch "Video pre-cache complete.";
                    sam_macos_set_status(status);
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
                sam_macos_set_status("Out of memory during video pre-cache.");
                return;
            };
            var decoded = zigimg.Image.fromRawPixelsOwned(width, height, pixels, .rgb24) catch {
                self.allocator.free(pixels);
                sam_macos_set_status("Could not prepare video frame for pre-cache.");
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
            const result = here.call(.computeQuery, args);
            self.model_mutex.unlock(self.io);
            const values = result catch |err| {
                log.info(self.io, "Video pre-cache failed at frame {d}: {t}: {s}", .{ frames, err, sam3.onnx.lastError() });
                sam_macos_set_status("Video pre-cache inference failed.");
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
                self.precache_scanned_until = @max(self.precache_scanned_until, frame.pts_seconds);
                self.mutex.unlock(self.io);
            }
            if (duration > 0 and std.math.isFinite(frame.pts_seconds)) {
                fraction = @max(fraction, std.math.clamp(frame.pts_seconds / duration, 0, 0.9999));
            }
            const now = std.Io.Timestamp.now(self.io, .awake);
            if (frames == 1 or last_ui_update.untilNow(self.io, .awake).nanoseconds >= 100_000_000) {
                sam_macos_set_precache_progress(1, fraction, frames);
                last_ui_update = now;
            }
            if (frames == 1 or last_log.untilNow(self.io, .awake).nanoseconds >= 1_000_000_000) {
                log.info(self.io, "pre-cache frame {d} at {d:.2}/{d:.2} s ({d:.2}%) in {f}", .{
                    frames, frame.pts_seconds, duration, fraction * 100, frame_started.untilNow(self.io, .awake),
                });
                last_log = now;
            }
        }
        if (!self.video_stop.load(.acquire)) {
            sam_macos_set_status("Video pre-cache cancelled.");
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

    fn handleCancelQuery(self: *App) void {
        if (self.query_active.load(.acquire)) {
            self.query_cancel.store(true, .release);
            sam_macos_set_status("Cancelling visual database query…");
        }
    }

    fn handleQuery(self: *App, raw_query: []const u8) void {
        if (!self.video_active or self.video_path == null) {
            sam_macos_set_status("Open a video first before executing visual SQL queries.");
            return;
        }
        if (self.query_active.load(.acquire)) {
            self.handleCancelQuery();
            return;
        }
        if (self.query_thread) |thread| {
            thread.join();
            self.query_thread = null;
        }
        if (self.is_busy) return;

        self.query_cancel.store(false, .release);
        self.query_active.store(true, .release);
        sam_macos_set_query_active(1);

        self.is_busy = true;
        sam_macos_set_busy(1);
        sam_macos_set_status("Planning visual database query with index pushdown…");

        const QueryContext = struct {
            app: *App,
            query: []const u8,
        };

        const ctx = self.allocator.create(QueryContext) catch {
            self.query_active.store(false, .release);
            sam_macos_set_query_active(0);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const query_copy = self.allocator.dupe(u8, raw_query) catch {
            self.allocator.destroy(ctx);
            self.query_active.store(false, .release);
            sam_macos_set_query_active(0);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        ctx.* = .{ .app = self, .query = query_copy };

        self.query_thread = std.Thread.spawn(.{}, runQueryWorker, .{ctx}) catch {
            self.allocator.free(query_copy);
            self.allocator.destroy(ctx);
            self.query_active.store(false, .release);
            sam_macos_set_query_active(0);
            self.is_busy = false;
            sam_macos_set_busy(0);
            sam_macos_set_status("Could not spawn query thread.");
            return;
        };
    }

    fn runQueryWorker(ctx: anytype) void {
        defer {
            ctx.app.query_active.store(false, .release);
            sam_macos_set_query_active(0);
            ctx.app.is_busy = false;
            sam_macos_set_busy(0);
            ctx.app.allocator.free(ctx.query);
            ctx.app.allocator.destroy(ctx);
        }
        const self = ctx.app;
        const query = ctx.query;
        const started = std.Io.Timestamp.now(self.io, .awake);

        // Normalize query if needed
        var query_buf: [2048]u8 = undefined;
        var final_query: []const u8 = query;
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
                    sam_macos_set_status(status_msg);
                    self.runPrecache(target_prompt);
                    return;
                }
            } else |err| {
                log.info(self.io, "Failed to parse CREATE INDEX: {t}", .{err});
                var err_buf: [160]u8 = undefined;
                const err_msg = std.fmt.bufPrintZ(&err_buf, "CREATE INDEX syntax error: {t}", .{err}) catch "Syntax error.";
                sam_macos_set_status(err_msg);
                return;
            }
        }

        if (std.ascii.startsWithIgnoreCase(trimmed, "WHERE")) {
            final_query = std.fmt.bufPrint(&query_buf, "SELECT frame, sam3(frame, \"matched\") FROM \"{s}\" {s}", .{
                self.video_path.?,
                trimmed,
            }) catch query;
        } else if (std.ascii.indexOfIgnoreCase(trimmed, "FROM") == null) {
            if (std.ascii.indexOfIgnoreCase(trimmed, "WHERE")) |where_idx| {
                final_query = std.fmt.bufPrint(&query_buf, "{s} FROM \"{s}\" {s}", .{
                    trimmed[0..where_idx],
                    self.video_path.?,
                    trimmed[where_idx..],
                }) catch query;
            } else {
                final_query = std.fmt.bufPrint(&query_buf, "{s} FROM \"{s}\"", .{
                    trimmed,
                    self.video_path.?,
                }) catch query;
            }
        }

        log.info(self.io, "Executing visual query: {s}", .{final_query});

        // Extract prompt if sam3(frame, "...") is in query
        if (std.ascii.indexOfIgnoreCase(final_query, "sam3")) |sam_idx| {
            if (std.mem.indexOfPos(u8, final_query, sam_idx, "\"")) |q1| {
                if (std.mem.indexOfPos(u8, final_query, q1 + 1, "\"")) |q2| {
                    const prompt = final_query[q1 + 1 .. q2];
                    self.mutex.lock(self.io) catch return;
                    self.video_phrase_len = @min(prompt.len, self.video_phrase.len);
                    @memcpy(self.video_phrase[0..self.video_phrase_len], prompt[0..self.video_phrase_len]);
                    self.mutex.unlock(self.io);
                }
            } else if (std.mem.indexOfPos(u8, final_query, sam_idx, "'")) |q1| {
                if (std.mem.indexOfPos(u8, final_query, q1 + 1, "'")) |q2| {
                    const prompt = final_query[q1 + 1 .. q2];
                    self.mutex.lock(self.io) catch return;
                    self.video_phrase_len = @min(prompt.len, self.video_phrase.len);
                    @memcpy(self.video_phrase[0..self.video_phrase_len], prompt[0..self.video_phrase_len]);
                    self.mutex.unlock(self.io);
                }
            }
        }

        self.mutex.lock(self.io) catch return;
        self.query_matches.clearRetainingCapacity();
        self.query_match_idx = 0;
        self.mutex.unlock(self.io);

        const QueryStreamer = struct {
            app: *App,
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
                app.query_matches.append(app.allocator, @intCast(idx)) catch {};
                const total = app.query_matches.items.len;

                if (!streamer.first_match_emitted) {
                    streamer.first_match_emitted = true;
                    app.query_match_idx = 0;
                    app.video_seek_target = match_pts;
                    app.video_step_requested = true;
                    app.mutex.unlock(app.io);

                    // INSTANT UI update for first frame!
                    var status_buf: [256]u8 = undefined;
                    const status_msg = std.fmt.bufPrintZ(&status_buf, "First match found instantly! Frame #{d} at {d:.2}s. Streaming visual query results…", .{
                        idx,
                        match_pts,
                    }) catch "First frame matched!";
                    sam_macos_set_status(status_msg);
                    log.info(app.io, "Instantly streamed first match: Frame #{d} at {d:.2}s", .{ idx, match_pts });
                } else {
                    app.mutex.unlock(app.io);

                    // Periodically update UI with progress so user sees live match count
                    const now = std.Io.Timestamp.now(app.io, .awake);
                    if (streamer.last_ui_update.untilNow(app.io, .awake).nanoseconds > 200_000_000 or total % 50 == 0) {
                        streamer.last_ui_update = now;
                        var status_buf: [160]u8 = undefined;
                        const status_msg = std.fmt.bufPrintZ(&status_buf, "Streaming query: {d} matches found… scanning… Press Cancel Query to stop.", .{total}) catch "Streaming query…";
                        sam_macos_set_status(status_msg);
                    }
                }
            }
        };

        var streamer = QueryStreamer{
            .app = self,
            .last_ui_update = started,
        };
        const stream_cb: vdb.engine.RowCallback = .{
            .ctx = &streamer,
            .onRow = QueryStreamer.onRow,
        };

        var video_adapter = MacosVideoReader.init(self.video_path.?);
        defer video_adapter.deinit();

        self.vdb_database.engine_inst.sam3 = .{
            .ptr = self,
            .segmentFn = sam3SegmentBridge,
        };

        var result = self.vdb_database.executeQuery(final_query, video_adapter.asReader(), &self.query_cancel, stream_cb) catch |err| {
            if (err == error.QueryCancelled) {
                log.info(self.io, "Visual database query cancelled.", .{});
                self.mutex.lock(self.io) catch return;
                const matches_so_far = self.query_matches.items.len;
                self.mutex.unlock(self.io);
                var cancel_buf: [160]u8 = undefined;
                const cancel_msg = if (matches_so_far > 0)
                    std.fmt.bufPrintZ(&cancel_buf, "Query cancelled. Kept {d} matches found so far.", .{matches_so_far}) catch "Query cancelled."
                else
                    "Visual query cancelled.";
                sam_macos_set_status(cancel_msg);
                return;
            }
            log.info(self.io, "Query execution error: {t}", .{err});
            var err_buf: [160]u8 = undefined;
            const err_msg = std.fmt.bufPrintZ(&err_buf, "Query error: {t}", .{err}) catch "Query execution failed.";
            sam_macos_set_status(err_msg);
            return;
        };
        defer result.deinit();

        const elapsed = started.untilNow(self.io, .awake);

        self.mutex.lock(self.io) catch return;
        const total_matches = self.query_matches.items.len;
        const first_frame_target = if (total_matches > 0) @as(f64, @floatFromInt(self.query_matches.items[0])) * 0.0333 else null;
        self.mutex.unlock(self.io);

        if (total_matches > 0) {
            var status_buf: [256]u8 = undefined;
            const first_sec = first_frame_target.?;
            const status_msg = std.fmt.bufPrintZ(&status_buf, "Query complete: {d} match(es) in {f}. Showing match 1 (Frame #{d} at {d:.2}s). Press Next Frame to step.", .{
                total_matches,
                elapsed,
                self.query_matches.items[0],
                first_sec,
            }) catch "Query completed.";
            sam_macos_set_status(status_msg);
        } else {
            var status_buf: [160]u8 = undefined;
            const status_msg = std.fmt.bufPrintZ(&status_buf, "Query returned 0 matching frames in {f}.", .{elapsed}) catch "0 matches.";
            sam_macos_set_status(status_msg);
        }
    }

    fn handleFindText(self: *App, text: [*:0]const u8) void {
        const phrase = std.mem.span(text);
        if (phrase.len == 0) return;

        if (isQuerySql(phrase)) {
            self.handleQuery(phrase);
            return;
        }

        if (self.video_active) {
            self.mutex.lock(self.io) catch return;
            if (self.video_previewed and self.video_seek_target == null) {
                self.video_seek_target = self.video_position;
            }
            self.video_phrase_len = @min(phrase.len, self.video_phrase.len);
            @memcpy(self.video_phrase[0..self.video_phrase_len], phrase[0..self.video_phrase_len]);
            const playing = self.video_playing.load(.acquire);
            self.video_step_requested = !playing;
            self.mutex.unlock(self.io);
            sam_macos_set_status(if (playing) "Updating the video prompt…" else "Processing the current video frame…");
            return;
        }
        if (self.is_busy or self.image == null) return;

        self.is_busy = true;
        sam_macos_set_busy(1);

        var status_buf: [256]u8 = undefined;
        const status = std.fmt.bufPrintZ(&status_buf, "Looking for “{s}”…", .{phrase}) catch "Searching…";
        sam_macos_set_status(status);

        const PhraseContext = struct {
            app: *App,
            phrase: []const u8,
        };
        const ctx = self.allocator.create(PhraseContext) catch {
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const phrase_copy = self.allocator.dupe(u8, phrase) catch {
            self.allocator.destroy(ctx);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        ctx.* = .{ .app = self, .phrase = phrase_copy };

        const thread = std.Thread.spawn(.{}, runLookupWorker, .{ctx}) catch {
            self.allocator.free(phrase_copy);
            self.allocator.destroy(ctx);
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        thread.detach();
    }

    fn runLookupWorker(ctx: anytype) void {
        defer {
            ctx.app.allocator.free(ctx.phrase);
            ctx.app.allocator.destroy(ctx);
        }
        const self = ctx.app;
        const phrase = ctx.phrase;
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
            if (i != selected) self.overlayMask(img, masks, i, 0.35);
        }
        if (selected < masks.count) self.overlayMask(img, masks, selected, 0.6);
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

fn cPrecacheVideo(text: [*:0]const u8) callconv(.c) void {
    if (g_app) |app| app.handlePrecacheVideo(text);
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

fn cCancelQuery() callconv(.c) void {
    if (g_app) |app| {
        app.handleCancelQuery();
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
