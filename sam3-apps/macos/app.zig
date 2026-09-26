const std = @import("std");
const sam3 = @import("sam3");
const render = sam3.render;
const zigimg = @import("zigimg");
const zimo = @import("zimo");
const here = zimo.bind(@This(), @embedFile("app.zig"));
threadlocal var active_model: ?*sam3.Model = null;
threadlocal var query_computed = false;

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
    on_canvas_click: ?*const fn (norm_x: f32, norm_y: f32, is_positive: c_int) callconv(.c) void,
    on_select_mask: ?*const fn (mask_index: c_int) callconv(.c) void,
};

const SamMaskInfo = extern struct {
    score: f32,
    coverage: f32,
};

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
extern fn sam_macos_video_open(path: [*:0]const u8, start_seconds: f64) ?*anyopaque;
extern fn sam_macos_video_duration(reader: *anyopaque) f64;
extern fn sam_macos_video_next(reader: *anyopaque, frame: *VideoFrame) c_int;
extern fn sam_macos_video_free_frame(frame: *VideoFrame) void;
extern fn sam_macos_video_close(reader: *anyopaque) void;
extern fn sam_macos_dispatch_main(func: *const fn (?*anyopaque) callconv(.c) void, ctx: ?*anyopaque) void;

const max_points = 32;

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    mutex: std.Io.Mutex = .init,
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

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) App {
        return .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
        };
    }

    pub fn deinit(self: *App) void {
        self.stopVideo();
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
            std.debug.print("Failed to read image file {s}: {t}\n", .{ path, err });
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
            std.debug.print("Failed to decode image: {t}\n", .{err});
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
        sam_macos_set_video_mode(0, 0);
        sam_macos_set_video_timeline(0, 0);
    }

    fn openVideoFromPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        self.stopVideo();
        self.video_path = self.allocator.dupeZ(u8, path) catch {
            sam_macos_set_status("Out of memory opening video.");
            return;
        };
        self.video_stop.store(false, .release);
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
        sam_macos_set_status("Video opened. Enter a word and press Find to process this frame.");
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
        active_model = self.model;
        defer active_model = null;
        var reader: ?*anyopaque = null;
        defer if (reader) |handle| sam_macos_video_close(handle);
        var frame_number: usize = 0;
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
                frame_number = 0;
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
                frame_number = 0;
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
            query_computed = false;
            const values = here.call(.computeQuery, .{
                self.allocator,
                [_][]const u8{
                    sam3.assets.concept_vision_encoder.sha256,
                    sam3.assets.concept_text_encoder.sha256,
                    sam3.assets.concept_decoder.sha256,
                    sam3.assets.concept_tokenizer_json.sha256,
                },
                sam3.RgbImage.fromImage(decoded),
                phrase,
                @as(f32, 0.5),
            }) catch |err| {
                std.debug.print("Video frame lookup failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
                sam_macos_set_status("Video frame inference failed.");
                self.video_playing.store(false, .release);
                sam_macos_set_video_mode(1, 0);
                continue;
            };
            defer self.allocator.free(values);
            const cache_hit = !query_computed;
            var masks = unpackMasks(self.allocator, values) catch {
                sam_macos_set_status("Cached video frame is invalid.");
                continue;
            };
            var masks_owned = true;
            defer if (masks_owned) masks.deinit();
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
            masks_owned = false;
            self.points_len = 0;
            if (self.coverages.len != masks.count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
            }
            self.best_mask_idx = if (masks.count == 0) -1 else @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
            self.selected_mask = self.best_mask_idx;
            self.video_position = video_frame.pts_seconds;
            self.renderComposite(self.selected_mask);
            sam_macos_set_image(self.frame.ptr, @intCast(width), @intCast(height));
            var mask_infos: [64]SamMaskInfo = undefined;
            const count = @min(masks.count, mask_infos.len);
            for (0..count) |i| mask_infos[i] = .{
                .score = masks.scores[i],
                .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
            };
            sam_macos_set_masks(@intCast(count), &mask_infos, self.best_mask_idx, self.selected_mask);
            const duration = self.video_duration;
            self.mutex.unlock(self.io);
            sam_macos_set_video_timeline(duration, video_frame.pts_seconds);

            frame_number += 1;
            previous_pts = video_frame.pts_seconds;
            previous_display = std.Io.Timestamp.now(self.io, .awake);
            var status_buf: [256]u8 = undefined;
            const status = std.fmt.bufPrintZ(&status_buf, "Frame {d}: {d} match(es) for “{s}” in {f} ({s})", .{
                frame_number, masks.count, phrase, lookup_elapsed,
                if (cache_hit) "cache hit" else "computed",
            }) catch "Video frame processed.";
            sam_macos_set_status(status);
            std.debug.print("  frame {d}: \"{s}\" -> {d} object(s) in {f} ({s})\n", .{
                frame_number, phrase, masks.count, lookup_elapsed,
                if (cache_hit) "cache hit" else "computed",
            });
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
            std.debug.print("Vision encoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Vision encoder failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer embedding.deinit();

        const decode_started = std.Io.Timestamp.now(self.io, .awake);
        const masks = self.model.segment(&embedding, self.points[0..self.points_len]) catch |err| {
            std.debug.print("Decoder failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Segmentation failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const decode_elapsed = decode_started.untilNow(self.io, .awake);
        std.debug.print("  {d} point(s) -> {d} masks in {f}\n", .{
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

        var mask_infos: [16]SamMaskInfo = undefined;
        const count = @min(masks.count, mask_infos.len);
        for (0..count) |i| {
            mask_infos[i] = .{
                .score = masks.scores[i],
                .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
            };
        }
        sam_macos_set_masks(@intCast(count), &mask_infos, self.best_mask_idx, self.selected_mask);

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

    fn handleFindText(self: *App, text: [*:0]const u8) void {
        const phrase = std.mem.span(text);
        if (phrase.len == 0) return;
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
        active_model = self.model;
        defer active_model = null;
        query_computed = false;
        const values = here.call(.computeQuery, .{
            self.allocator,
            [_][]const u8{
                sam3.assets.concept_vision_encoder.sha256,
                sam3.assets.concept_text_encoder.sha256,
                sam3.assets.concept_decoder.sha256,
                sam3.assets.concept_tokenizer_json.sha256,
            },
            sam3.RgbImage.fromImage(self.image.?),
            phrase,
            @as(f32, 0.5),
        }) catch |err| {
            std.debug.print("Text lookup failed: {t}: {s}\n", .{ err, sam3.onnx.lastError() });
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        defer self.allocator.free(values);
        const cache_hit = !query_computed;
        const masks = unpackMasks(self.allocator, values) catch |err| {
            std.debug.print("Cached text lookup failed: {t}\n", .{err});
            sam_macos_set_status("Text lookup failed");
            self.is_busy = false;
            sam_macos_set_busy(0);
            return;
        };
        const lookup_elapsed = lookup_started.untilNow(self.io, .awake);
        std.debug.print("  \"{s}\" -> {d} object(s) in {f} ({s})\n", .{
            phrase,
            masks.count,
            lookup_elapsed,
            if (cache_hit) "cache hit" else "computed",
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

            var mask_infos: [64]SamMaskInfo = undefined;
            const count = @min(masks.count, mask_infos.len);
            for (0..count) |i| {
                mask_infos[i] = .{
                    .score = masks.scores[i],
                    .coverage = if (self.coverages.len > i) self.coverages[i] else 0,
                };
            }
            sam_macos_set_masks(@intCast(count), &mask_infos, self.best_mask_idx, self.selected_mask);

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
        std.debug.print("  {s}encoded {d}x{d} in {f}\n", .{
            if (concept) "concept-" else "",
            img.width,
            img.height,
            started.untilNow(self.io, .awake),
        });
        return embedding;
    }

    fn renderComposite(self: *App, mask_index: i32) void {
        const img = self.image orelse return;
        var plane: ?[]const f32 = null;
        var mw: usize = 0;
        var mh: usize = 0;
        if (mask_index >= 0 and self.masks != null) {
            const masks = self.masks.?;
            const u_index: usize = @intCast(mask_index);
            if (u_index < masks.count) {
                plane = masks.plane(u_index);
                mw = masks.width;
                mh = masks.height;
            }
        }
        render.compositeRgba(self.allocator, img, self.frame, plane, mw, mh, self.points[0..self.points_len]);
    }
};

pub fn computeQuery(
    allocator: std.mem.Allocator,
    model_ids: [4][]const u8,
    image: sam3.RgbImage,
    phrase: []const u8,
    min_score: f32,
) ![]f32 {
    _ = model_ids;
    query_computed = true;
    const model = active_model orelse return error.NoActiveModel;
    var embedding = try model.encodeForText(image);
    defer embedding.deinit();
    var masks = try model.find(&embedding, phrase, .{ .min_score = min_score });
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
