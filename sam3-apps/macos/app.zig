const std = @import("std");
const sam3 = @import("sam3");
const render = sam3.render;
const zigimg = @import("zigimg");
const log = @import("log");
const vdb = @import("vdb");
const core = @import("core");
const QueryTab = vdb.query_tab.QueryTab;

const MaskColor = core.render.MaskColor;
const mask_colors = core.render.mask_colors;
const maskColor = core.render.maskColor;

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
extern fn sam_macos_set_video_mode(active: c_int, playing: c_int) void;
extern fn sam_macos_set_video_timeline(duration: f64, position: f64) void;
extern fn sam_macos_set_precache_progress(state: c_int, fraction: f64, frames: usize) void;
extern fn sam_macos_set_query_table(json: [*:0]const u8) void;
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

    pub fn init(allocator: std.mem.Allocator, path: [:0]const u8) MacosVideoReader {
        _ = allocator;
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

    pub fn isValid(self: *const MacosVideoReader) bool {
        return self.reader != null;
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
    video_overlay_prompts: vdb.ast.Prompts = .{},
    video_duration: f64 = 0,
    video_position: f64 = 0,
    video_previewed: bool = false,
    video_seek_target: ?f64 = null,
    video_step_requested: bool = false,
    session: core.Session(MacosVideoReader),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) App {
        return .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
            .session = core.Session(MacosVideoReader).init(allocator, io),
        };
    }

    pub fn deinit(self: *App) void {
        self.session.deinit();
        self.stopVideo();
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        if (self.image) |*img| img.deinit(self.allocator);
        self.allocator.free(self.frame);
        self.allocator.free(self.coverages);
        core.inference.setActiveModel(null);
    }

    pub fn start(self: *App) !void {
        g_app = self;
        core.inference.setActiveModel(self.model);

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
        const msg = std.fmt.bufPrintSentinel(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }, 0) catch "Image loaded.";
        sam_macos_set_status(msg);
    }

    pub fn stopVideo(self: *App) void {
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

    pub fn openVideoFromPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        self.stopVideo();
        const owned = self.allocator.dupeSentinel(u8, path, 0) catch {
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
                const source = self.allocator.dupeSentinel(u8, owned, 0) catch null;
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
        const title_str = std.fmt.bufPrintSentinel(&title_buf, "SAM 3 — Visual Database — {s}", .{basename}, 0) catch "SAM 3 — Visual Database";
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
            const smsg = std.fmt.bufPrintSentinel(&sbuf, "Video opened. Loaded visual index with {d} concept(s) from .vdb sidecar.", .{indexed_concepts_count}, 0) catch "Video opened with index.";
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
            const msg = std.fmt.bufPrintSentinel(&buf, "Restarted at query match 1 of {d} (Frame #{d} at {d:.2}s).", .{ total, target_frame, target_sec }, 0) catch "Query match 1.";
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
            const msg = std.fmt.bufPrintSentinel(&buf, "Query match {d} of {d} (Frame #{d} at {d:.2}s).", .{ match_num, total, target_frame, target_sec }, 0) catch "Next query match.";
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
                std.fmt.bufPrintSentinel(&status_buf, "At {s}: frame shown.", .{timecode}, 0) catch "Frame shown."
            else if (self.masks != null)
                std.fmt.bufPrintSentinel(&status_buf, "At {s}: {d} match(es) for “{s}” in {f}", .{
                    timecode, mask_count, phrase, lookup_elapsed,
                }, 0) catch "Video frame processed."
            else
                std.fmt.bufPrintSentinel(&status_buf, "At {s}: waiting for cached “{s}” result.", .{ timecode, phrase }, 0) catch "Frame shown without a cached result.";
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
                    const match_msg = std.fmt.bufPrintSentinel(&match_buf, "Playing query match {d} of {d} (Frame #{d} at {d:.2}s)", .{
                        cur_num, total_matches, next_match_frame, next_sec,
                    }, 0) catch "Playing query match.";
                    sam_macos_set_status(match_msg);
                    continue;
                }
                self.mutex.unlock(self.io);
            }
        }
    }

    fn cachedFrameQuery(self: *App, image: sam3.RgbImage, phrase: []const u8) ![]f32 {
        return core.inference.cachedQuery(self.allocator, image, phrase);
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
            var masks = core.inference.unpackMasks(self.allocator, values) catch |err| {
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
        const status = std.fmt.bufPrintSentinel(&status_buf, "{d} point(s) -> {d} masks in {f}", .{
            self.points_len,
            masks.count,
            elapsed,
        }, 0) catch "Segmentation complete";
        sam_macos_set_status(status);

        self.is_busy = false;
        sam_macos_set_busy(0);
    }

    pub fn isBusy(self: *const App) bool {
        return self.is_busy;
    }

    pub fn getVideoPath(self: *const App) ?[]const u8 {
        return self.video_path;
    }

    pub fn isVideoActive(self: *const App) bool {
        return self.video_active;
    }

    pub fn selectedTab(self: *const App) ?*QueryTab {
        return self.session.selectedTab();
    }

    pub fn isQueryVideo(self: *const App) bool {
        return self.session.isQueryVideo(self.video_path);
    }

    pub fn isSelectedQuery(self: *const App, tab: *QueryTab) bool {
        return self.session.isSelectedQuery(self.video_path, tab);
    }

    pub fn queryMatches(self: *const App) []const u32 {
        return self.session.queryMatches(self.video_path);
    }

    pub fn selectedQueryActive(self: *const App) bool {
        return self.session.selectedQueryActive();
    }

    pub fn setStatus(self: *App, status: []const u8) void {
        const message = self.allocator.dupeSentinel(u8, status, 0) catch return;
        defer self.allocator.free(message);
        sam_macos_set_status(message);
    }

    pub fn setQueryText(self: *App, text: []const u8) void {
        const sql = self.allocator.dupeSentinel(u8, text, 0) catch return;
        defer self.allocator.free(sql);
        sam_macos_set_query_text(sql);
    }

    pub fn clearOverlay(self: *App) void {
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
        sam_macos_set_masks(0, null, 0, -1);
        if (self.image) |img| sam_macos_set_image(self.frame.ptr, @intCast(img.width), @intCast(img.height));
    }

    pub fn setOverlayPrompts(self: *App, prompts: vdb.ast.Prompts) void {
        self.video_overlay_prompts = prompts;
        self.video_phrase_len = 0;
        if (prompts.count > 0) {
            const prompt = prompts.get(0);
            self.video_phrase_len = prompt.len;
            @memcpy(self.video_phrase[0..prompt.len], prompt);
        }
    }

    pub fn setOverlayPhrase(self: *App, phrase: []const u8) void {
        self.video_overlay_prompts = .{};
        self.video_phrase_len = @min(phrase.len, self.video_phrase.len);
        @memcpy(self.video_phrase[0..self.video_phrase_len], phrase[0..self.video_phrase_len]);
    }

    pub fn seekVideo(self: *App, pts_seconds: f64) void {
        self.video_seek_target = pts_seconds;
        self.video_step_requested = true;
    }

    pub fn refreshQueryTabs(self: *App) void {
        const labels_z = self.session.formatTabLabels(self.allocator) catch return;
        defer self.allocator.free(labels_z);
        sam_macos_set_query_tabs(labels_z, if (self.session.selected_query) |index| @intCast(index) else -1);
        const table_json = self.session.formatTable(self.allocator) catch return;
        defer self.allocator.free(table_json);
        sam_macos_set_query_table(table_json);
        const selected = self.session.selectedTab();
        sam_macos_set_query_active(@intFromBool(if (selected) |tab| tab.query_active.load(.acquire) else false));
        if (selected) |tab| {
            sam_macos_set_precache_progress(@intFromBool(tab.precache_active.load(.acquire)), tab.precache_progress, tab.frames_processed);
        } else {
            sam_macos_set_precache_progress(0, 0, 0);
        }
    }

    fn editQuery(self: *App, text: []const u8) void {
        self.session.editQuery(text);
    }

    fn newQuery(self: *App) void {
        self.session.newQuery(self);
    }

    fn closeQuery(self: *App, index: usize) void {
        self.session.closeQuery(self, index);
    }

    fn selectQuery(self: *App, index: usize) void {
        self.session.selectQuery(self, index);
    }

    fn restoreQueryOverlay(self: *App) void {
        self.session.restoreQueryOverlay(self);
    }

    fn handleCancelQuery(self: *App) void {
        self.session.handleCancelQuery(self);
    }

    fn handleClearQuery(self: *App) void {
        self.session.handleClearQuery(self);
    }

    fn handleQuery(self: *App, raw_query: []const u8) void {
        self.session.handleQuery(self, raw_query);
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
        const values = core.inference.cachedQuery(
            self.allocator,
            sam3.RgbImage.fromImage(self.image.?),
            phrase,
        ) catch |err| {
            log.info(self.io, "Text lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer self.allocator.free(values);
        const masks = core.inference.unpackMasks(self.allocator, values) catch |err| {
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
            const msg = std.fmt.bufPrintSentinel(&msg_buf, "No objects matched “{s}”.", .{phrase}, 0) catch "No objects matched.";
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
            const status = std.fmt.bufPrintSentinel(&status_buf, "{d} object(s) matched “{s}” in {f}", .{
                masks.count,
                phrase,
                elapsed,
            }, 0) catch "Search complete";
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
        core.render.overlayMask(self.frame, img, masks, index, alpha);
    }

    fn drawPointMarkers(self: *App, img: zigimg.Image) void {
        core.render.drawPointMarkers(self.frame, img, self.points[0..self.points_len]);
    }
};

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
    if (g_app) |app| app.session.reapClosed();
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
