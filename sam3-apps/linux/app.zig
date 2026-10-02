const std = @import("std");
const sam3 = @import("sam3");
const zigimg = @import("zigimg");
const wayland = @import("wayland.zig");
const font = @import("font.zig");
const zimo = @import("zimo");
const log = @import("log");
const vdb = @import("vdb");
const QueryTab = vdb.query_tab.QueryTab;
const here = zimo.bind(@This(), @embedFile("app.zig"));

fn getMonotonicMs() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(ts.nsec, 1_000_000);
}

const VideoFrame = extern struct { rgb: ?[*]u8, width: c_int, height: c_int, pts_seconds: f64 };
extern fn sam_linux_video_open(path: [*:0]const u8, start: f64) ?*anyopaque;
extern fn sam_linux_video_duration(reader: *anyopaque) f64;
extern fn sam_linux_video_fps(reader: *anyopaque) f64;
extern fn sam_linux_video_next(reader: *anyopaque, frame: *VideoFrame) c_int;
extern fn sam_linux_video_free_frame(frame: *VideoFrame) void;
extern fn sam_linux_video_close(reader: *anyopaque) void;

const LinuxVideoReader = struct {
    allocator: std.mem.Allocator,
    path: [:0]const u8,
    duration: f64,
    fps: f64,
    reader: ?*anyopaque = null,
    current_index: usize = 0,
    current_frame: ?VideoFrame = null,

    pub fn init(allocator: std.mem.Allocator, path: [:0]const u8) LinuxVideoReader {
        const owned = allocator.dupeZ(u8, path) catch path;
        const handle = sam_linux_video_open(owned.ptr, 0);
        const dur = if (handle) |h| sam_linux_video_duration(h) else 0;
        const fps = if (handle) |h| sam_linux_video_fps(h) else 30.0;
        return .{
            .allocator = allocator,
            .path = owned,
            .duration = dur,
            .fps = fps,
            .reader = handle,
            .current_index = 0,
            .current_frame = null,
        };
    }

    pub fn deinit(self: *LinuxVideoReader) void {
        if (self.current_frame) |*vf| {
            sam_linux_video_free_frame(vf);
            self.current_frame = null;
        }
        if (self.reader) |r| {
            sam_linux_video_close(r);
            self.reader = null;
        }
        self.allocator.free(self.path);
    }

    pub fn asReader(self: *LinuxVideoReader) vdb.engine.VideoReader {
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
        const self: *LinuxVideoReader = @ptrCast(@alignCast(ctx));
        return @intFromFloat(@max(self.duration * self.fps, 1.0));
    }

    fn seekToFrameImpl(ctx: *anyopaque, frame_idx: usize) anyerror!void {
        const self: *LinuxVideoReader = @ptrCast(@alignCast(ctx));
        if (self.current_frame) |*vf| {
            sam_linux_video_free_frame(vf);
            self.current_frame = null;
        }
        if (self.reader) |r| {
            sam_linux_video_close(r);
            self.reader = null;
        }
        const frame_time = 1.0 / @max(self.fps, 1.0);
        const pts = @as(f64, @floatFromInt(frame_idx)) * frame_time;
        self.reader = sam_linux_video_open(self.path.ptr, pts);
        self.current_index = frame_idx;
    }

    fn nextFrameImpl(ctx: *anyopaque) anyerror!?vdb.types.FrameRef {
        const self: *LinuxVideoReader = @ptrCast(@alignCast(ctx));
        if (self.reader == null) return null;
        if (self.current_frame) |*vf| {
            sam_linux_video_free_frame(vf);
            self.current_frame = null;
        }

        var vf: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
        const ret = sam_linux_video_next(self.reader.?, &vf);
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

fn isQuerySql(text: []const u8) bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (std.ascii.startsWithIgnoreCase(trimmed, "SELECT")) return true;
    if (std.ascii.startsWithIgnoreCase(trimmed, "CREATE")) return true;
    if (std.ascii.startsWithIgnoreCase(trimmed, "WHERE")) return true;
    return false;
}

const render = sam3.render;
const max_points = 32;
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
const BrowserEntry = struct {
    name: []u8,
    is_dir: bool,
};

pub const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *sam3.Model,
    example_path: []const u8,

    client: wayland.WaylandClient,

    mutex: std.Io.Mutex = .init,
    model_mutex: std.Io.Mutex = .init,
    index_mutex: std.Io.Mutex = .init,
    is_busy: bool = false,
    redraw_pending: std.atomic.Value(bool) = .init(true),

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
    video_seek_target: ?f64 = null,
    video_step_requested: bool = false,
    query_tabs: std.ArrayList(*QueryTab) = .empty,
    closed_queries: std.ArrayList(*QueryTab) = .empty,
    selected_query: ?usize = null,

    window_title: [256]u8 = undefined,
    window_title_len: usize = 0,

    status_text: [256]u8 = undefined,
    status_len: usize = 0,

    search_text: [1024]u8 = undefined,
    search_len: usize = 0,
    search_focused: bool = true,
    search_caret: usize = 0,
    search_scroll: usize = 0,
    search_anchor: ?usize = null,
    completion_visible: bool = false,
    completion_index: usize = 0,
    completion_cache: vdb.completion.Matches = .{},
    repeat_key: ?u32 = null,
    repeat_next_ms: i64 = 0,
    browser_open: bool = false,
    browser_path: [4096]u8 = undefined,
    browser_path_len: usize = 0,
    browser_dir: []u8 = &.{},
    browser_entries: std.ArrayList(BrowserEntry) = .empty,
    browser_scroll: usize = 0,
    shift_down: bool = false,
    ctrl_down: bool = false,
    alt_down: bool = false,
    caps_lock: bool = false,

    // Window state
    is_maximized: bool = false,
    unmaximized_width: u32 = 1000,
    unmaximized_height: u32 = 720,
    pending_width: u32 = 1000,
    pending_height: u32 = 720,
    last_titlebar_click_time: ?std.Io.Timestamp = null,

    // Layout geometry
    canvas_x: usize = 16,
    canvas_y: usize = 162,
    canvas_w: usize = 968,
    canvas_h: usize = 460,

    img_rect_x: usize = 0,
    img_rect_y: usize = 0,
    img_rect_w: usize = 0,
    img_rect_h: usize = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        model: *sam3.Model,
        example_path: []const u8,
        width: u32,
        height: u32,
    ) !App {
        const client = try wayland.WaylandClient.connect(allocator, width, height);

        var app: App = .{
            .allocator = allocator,
            .io = io,
            .model = model,
            .example_path = example_path,
            .client = client,
            .pending_width = width,
            .pending_height = height,
            .unmaximized_width = width,
            .unmaximized_height = height,
        };

        app.setWindowTitle("SAM 3 — Visual Database");
        app.setStatus("Enter SQL with FROM to choose a video.");
        return app;
    }

    pub fn deinit(self: *App) void {
        self.completion_cache.deinit(self.allocator);
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
        g_app = null;
        self.mutex.lock(self.io) catch return;
        defer self.mutex.unlock(self.io);

        if (self.masks) |*m| m.deinit();
        if (self.image) |*img| img.deinit(self.allocator);
        self.allocator.free(self.frame);
        self.allocator.free(self.coverages);
        self.clearBrowserEntries();
        self.browser_entries.deinit(self.allocator);
        if (self.browser_dir.len > 0) self.allocator.free(self.browser_dir);
        self.client.deinit();
    }

    fn setWindowTitle(self: *App, title: []const u8) void {
        const len = @min(title.len, self.window_title.len);
        @memcpy(self.window_title[0..len], title[0..len]);
        self.window_title_len = len;
        self.client.setTitle(self.window_title[0..len]) catch {};
        self.redraw_pending.store(true, .release);
    }

    fn setStatus(self: *App, text: []const u8) void {
        const len = @min(text.len, self.status_text.len);
        @memcpy(self.status_text[0..len], text[0..len]);
        self.status_len = len;
        self.redraw_pending.store(true, .release);
    }

    pub fn run(self: *App) !void {
        g_app = self;
        _ = here.id(.computeQuery);
        _ = here.id(.computeTextFeatures);
        self.newQuery();

        self.pending_width = self.client.width;
        self.pending_height = self.client.height;
        while (true) {
            vdb.query_tab.reapClosed(&self.closed_queries);
            if (self.pending_width != self.client.width or self.pending_height != self.client.height) {
                try self.client.resizeShmBuffer(self.pending_width, self.pending_height);
                self.redraw_pending.store(true, .release);
            }
            // Draw only when contents change, into a buffer released by the compositor.
            if (self.redraw_pending.load(.acquire) and try self.client.beginFrame()) {
                _ = self.redraw_pending.swap(false, .acq_rel);
                self.mutex.lock(self.io) catch return;
                self.redraw();
                self.mutex.unlock(self.io);
                try self.client.commitFrame();
            }

            // Calculate poll timeout based on next repeat event
            var timeout_ms: i32 = 16;
            if (self.repeat_key) |_| {
                const now = getMonotonicMs();
                if (now >= self.repeat_next_ms) {
                    timeout_ms = 0;
                } else {
                    const remaining: i64 = self.repeat_next_ms - now;
                    timeout_ms = @intCast(@min(@as(i64, 16), remaining));
                }
            }

            // Poll events with timeout
            const ev_opt = try self.client.pollEvent(timeout_ms);
            if (ev_opt) |ev| {
                switch (ev) {
                    .close => break,
                    .configure => |cfg| {
                        self.redraw_pending.store(true, .release);
                        self.is_maximized = cfg.maximized;
                        if (cfg.width > 0 and cfg.height > 0) {
                            if (!cfg.maximized) {
                                self.unmaximized_width = cfg.width;
                                self.unmaximized_height = cfg.height;
                            }
                            self.pending_width = cfg.width;
                            self.pending_height = cfg.height;
                        } else if (!cfg.maximized) {
                            self.pending_width = self.unmaximized_width;
                            self.pending_height = self.unmaximized_height;
                        }
                    },
                    .pointer_button => |btn| {
                        self.repeat_key = null;
                        if (btn.state == 1) { // Pressed
                            if (try self.handlePointerClick(btn.x, btn.y, btn.button, btn.serial)) {
                                break;
                            }
                            self.redraw_pending.store(true, .release);
                        }
                    },
                    .keyboard_key => |k| {
                        if (k.state == 1) {
                            self.handleKey(k.key, 1);
                            self.redraw_pending.store(true, .release);
                            if (self.isRepeatableKey(k.key) and self.client.repeat_rate > 0) {
                                self.repeat_key = k.key;
                                const now = getMonotonicMs();
                                const delay: i64 = @intCast(self.client.repeat_delay);
                                self.repeat_next_ms = now + delay;
                            } else {
                                self.repeat_key = null;
                            }
                        } else {
                            self.handleKey(k.key, 0);
                            if (self.repeat_key) |rk| {
                                if (rk == k.key) {
                                    self.repeat_key = null;
                                }
                            }
                        }
                    },
                    .keyboard_leave => {
                        self.repeat_key = null;
                    },
                    .pointer_motion => |motion| try self.updateCursor(motion.x, motion.y),
                }
            }

            // Check if key repeat timer has expired
            if (self.repeat_key) |rk| {
                const now = getMonotonicMs();
                if (now >= self.repeat_next_ms) {
                    if (self.isRepeatableKey(rk) and self.client.repeat_rate > 0) {
                        self.handleKey(rk, 1);
                        self.redraw_pending.store(true, .release);
                        const rate: i64 = @max(1, @as(i64, self.client.repeat_rate));
                        const interval: i64 = @max(1, @divTrunc(@as(i64, 1000), rate));
                        self.repeat_next_ms = now + interval;
                    } else {
                        self.repeat_key = null;
                    }
                }
            }
        }
    }

    fn openImageFromPath(self: *App, path: []const u8) bool {
        if (self.is_busy) return false;
        self.stopVideo();
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(
            self.io,
            path,
            self.allocator,
            .limited(64 * 1024 * 1024),
        ) catch |err| {
            log.info(self.io, "Failed to read image file {s}: {t}", .{ path, err });
            self.setStatus("Could not open image file.");
            return false;
        };
        defer self.allocator.free(file_bytes);

        return self.openImageFromBytes(file_bytes);
    }

    fn openImageFromBytes(self: *App, bytes: []const u8) bool {
        self.mutex.lock(self.io) catch return false;
        defer self.mutex.unlock(self.io);

        var decoded = sam3.decodeImage(self.allocator, bytes) catch |err| {
            log.info(self.io, "Failed to decode image: {t}", .{err});
            self.setStatus("That file is not an image this can decode.");
            return false;
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
            self.setStatus("Out of memory for frame buffer.");
            return false;
        };

        self.renderComposite(-1);

        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "{d} × {d} — click the object you want.", .{
            decoded.width,
            decoded.height,
        }) catch "Image loaded.";
        self.setStatus(msg);
        return true;
    }

    fn clearBrowserEntries(self: *App) void {
        for (self.browser_entries.items) |entry| self.allocator.free(entry.name);
        self.browser_entries.clearRetainingCapacity();
    }

    fn browseDir(self: *App, path: []const u8) void {
        const dir = std.Io.Dir.cwd().openDir(self.io, path, .{ .iterate = true }) catch {
            self.setStatus("Cannot open that folder.");
            return;
        };
        defer dir.close(self.io);

        const owned = self.allocator.dupe(u8, path) catch return;
        if (self.browser_dir.len > 0) self.allocator.free(self.browser_dir);
        self.browser_dir = owned;
        self.clearBrowserEntries();
        self.browser_scroll = 0;
        self.setBrowserPath(owned);

        var it = dir.iterate();
        while (it.next(self.io) catch null) |entry| {
            if (entry.name.len == 0 or entry.name[0] == '.') continue;
            const is_dir = entry.kind == .directory;
            const name = self.allocator.dupe(u8, entry.name) catch break;
            self.browser_entries.append(self.allocator, .{ .name = name, .is_dir = is_dir }) catch {
                self.allocator.free(name);
                break;
            };
        }
        std.mem.sort(BrowserEntry, self.browser_entries.items, {}, struct {
            fn lessThan(_: void, a: BrowserEntry, b: BrowserEntry) bool {
                if (a.is_dir != b.is_dir) return a.is_dir;
                return std.ascii.lessThanIgnoreCase(a.name, b.name);
            }
        }.lessThan);
    }

    fn setBrowserPath(self: *App, path: []const u8) void {
        self.browser_path_len = @min(path.len, self.browser_path.len);
        @memcpy(self.browser_path[0..self.browser_path_len], path[0..self.browser_path_len]);
    }

    fn openBrowser(self: *App) void {
        self.browser_open = true;
        const home = if (std.c.getenv("HOME")) |value| std.mem.span(value) else "/";
        self.browseDir(if (self.browser_dir.len > 0) self.browser_dir else home);
    }

    fn browserOpenPath(self: *App, path: []const u8) void {
        if (self.is_busy) return;
        const dir = std.Io.Dir.cwd().openDir(self.io, path, .{ .iterate = true }) catch null;
        if (dir) |d| {
            d.close(self.io);
            self.browseDir(path);
        } else if (isVideoPath(path)) {
            if (self.openVideoFromPath(path)) self.browser_open = false;
        } else if (self.openImageFromPath(path)) {
            self.browser_open = false;
        }
    }

    fn browserActivate(self: *App, index: usize) void {
        if (index >= self.browser_entries.items.len) return;
        const entry = self.browser_entries.items[index];
        const path = std.fs.path.join(self.allocator, &.{ self.browser_dir, entry.name }) catch return;
        defer self.allocator.free(path);
        self.browserOpenPath(path);
    }

    fn getResizeEdge(self: *App, x: usize, y: usize) u32 {
        if (self.is_maximized) return 0;
        const margin: usize = 8;
        const w = self.client.width;
        const h = self.client.height;
        if (w >= 110 and x >= w - 105 and y < 32) return 0;
        var edges: u32 = 0;
        if (y < margin) edges |= 1;
        if (y + margin >= h) edges |= 2;
        if (x < margin) edges |= 4;
        if (x + margin >= w) edges |= 8;
        return edges;
    }

    fn getResizeCursor(edges: u32) ?wayland.Cursor {
        return switch (edges) {
            1, 2 => .resize_ns,
            4, 8 => .resize_ew,
            5, 10 => .resize_nwse,
            6, 9 => .resize_nesw,
            else => null,
        };
    }

    fn handlePointerClick(self: *App, px: f32, py: f32, button: u32, serial: u32) !bool {
        const x: usize = @intFromFloat(@max(0, px));
        const y: usize = @intFromFloat(@max(0, py));
        const stride = self.client.width;

        // Interactive window resize from borders
        if (button == 0x110) {
            const edges = self.getResizeEdge(x, y);
            if (edges != 0) {
                try self.client.startInteractiveResize(serial, edges);
                return false;
            }
        }

        // Header bar / Window controls
        if (y < 32 and button == 0x110) {
            if (stride >= 110) {
                // Close button [x]
                if (x >= stride - 36 and x < stride - 10 and y >= 5 and y < 27) {
                    return true;
                }
                // Maximize / restore button [+] / [=]
                if (x >= stride - 68 and x < stride - 42 and y >= 5 and y < 27) {
                    if (self.is_maximized) {
                        try self.client.unsetMaximized();
                        self.is_maximized = false;
                        self.pending_width = self.unmaximized_width;
                        self.pending_height = self.unmaximized_height;
                    } else {
                        self.unmaximized_width = self.client.width;
                        self.unmaximized_height = self.client.height;
                        try self.client.setMaximized();
                        self.is_maximized = true;
                    }
                    return false;
                }
                // Minimize button [-]
                if (x >= stride - 100 and x < stride - 74 and y >= 5 and y < 27) {
                    try self.client.setMinimized();
                    return false;
                }
            }

            // Drag title bar to move, or double-click to toggle maximize
            const now = std.Io.Timestamp.now(self.io, .awake);
            if (self.last_titlebar_click_time) |prev| {
                const dt = prev.durationTo(now).nanoseconds;
                if (dt < 400_000_000) {
                    self.last_titlebar_click_time = null;
                    if (self.is_maximized) {
                        try self.client.unsetMaximized();
                        self.is_maximized = false;
                        self.pending_width = self.unmaximized_width;
                        self.pending_height = self.unmaximized_height;
                    } else {
                        self.unmaximized_width = self.client.width;
                        self.unmaximized_height = self.client.height;
                        try self.client.setMaximized();
                        self.is_maximized = true;
                    }
                    return false;
                }
            }
            self.last_titlebar_click_time = now;
            try self.client.startInteractiveMove(serial);
            return false;
        }

        if (self.browser_open) {
            if (button != 0x110) return false;
            const bx: usize = 16;
            const by: usize = 140;
            const bw = @min(stride -| 32, 640);
            const bh = @min(self.client.height -| 190, 420);
            if (x >= bx + bw -| 92 and x < bx + bw -| 12 and y >= by + 8 and y < by + 36) {
                self.browser_open = false;
            } else if (x >= bx + 12 and x < bx + 92 and y >= by + 8 and y < by + 36) {
                const parent = std.fs.path.dirname(self.browser_dir) orelse "/";
                self.browseDir(parent);
            } else if (x >= bx + 12 and x < bx + bw -| 12 and y >= by + 46 and y < by + 74) {
                self.browser_path_len = 0;
            } else if (y >= by + 82 and y < by + bh -| 42 and x >= bx + 12 and x < bx + bw -| 12) {
                self.browserActivate(self.browser_scroll + (y - by - 82) / 24);
            } else if (x >= bx + 12 and x < bx + 92 and y >= by + bh -| 36 and y < by + bh -| 8) {
                if (self.browser_scroll > 0) self.browser_scroll -= 1;
            } else if (x >= bx + 100 and x < bx + 180 and y >= by + bh -| 36 and y < by + bh -| 8) {
                if (self.browser_scroll + browserVisibleRows(bh) < self.browser_entries.items.len) self.browser_scroll += 1;
            } else if (x >= bx + bw -| 100 and x < bx + bw -| 12 and y >= by + bh -| 36 and y < by + bh -| 8) {
                const path = self.browser_path[0..self.browser_path_len];
                self.browserOpenPath(path);
            }
            return false;
        }

        if (button != 0x110 and button != 0x111) return false;

        const suggestions = self.queryCompletions();
        if (suggestions.len > 0 and button == 0x110 and x >= 16 and x < 16 + self.completionWidth() and y >= 136 and y < 136 + 24 * suggestions.len) {
            self.completion_index = (y - 136) / 24;
            self.acceptQueryCompletion();
            return false;
        }
        self.completion_visible = false;

        if (y >= 40 and y < 68 and button == 0x110 and x >= stride -| 40 and x < stride -| 16) {
            self.newQuery();
            return false;
        }
        if (y >= 40 and y < 68 and button == 0x110 and self.query_tabs.items.len > 0) {
            const count = self.query_tabs.items.len;
            const selected = self.selected_query orelse 0;
            if (x >= 16 and x < 40) {
                self.selectQuery(if (selected == 0) count - 1 else selected - 1);
            } else if (x >= stride -| 72 and x < stride -| 48) {
                self.selectQuery((selected + 1) % count);
            } else {
                const visible = @max(1, (stride -| 128) / 160);
                const first = @min(selected, count -| visible);
                if (x >= 48) {
                    const index = first + (x - 48) / 160;
                    if (index < count and index < first + visible) {
                        const offset = (x - 48) % 160;
                        if (offset >= 128 and offset < 152) {
                            self.closeQuery(index);
                        } else if (offset < 128) {
                            self.selectQuery(index);
                        }
                    }
                }
            }
            return false;
        }

        // SQL query input
        const q_x: usize = 16;
        const q_y: usize = 80;
        const q_h: usize = 54;
        const btn_gap: usize = 8;
        const run_btn_w: usize = 124;
        const clear_btn_w: usize = 114;
        const cancel_btn_w: usize = 110;
        const total_btns_w: usize = run_btn_w + 2 * btn_gap + cancel_btn_w + clear_btn_w;
        const q_w: usize = stride -| (q_x + total_btns_w + 24);
        const run_btn_x: usize = stride -| (total_btns_w + 16);
        const cancel_btn_x: usize = run_btn_x + run_btn_w + btn_gap;
        const clear_btn_x: usize = cancel_btn_x + cancel_btn_w + btn_gap;

        if (x >= q_x and x < q_x + q_w and y >= q_y and y < q_y + q_h and button == 0x110) {
            self.search_focused = true;
            const max_cols = if (q_w > 20) (q_w - 20) / font.font_width else 10;
            const char_col = (x -| (q_x + 10)) / font.font_width;
            const row: usize = if (y < q_y + 26) 0 else 1;
            self.search_caret = @min(self.search_len, row * max_cols + char_col);
            self.search_anchor = null;
            return false;
        }

        // Queries launch into new tabs; cancellation targets only the selected tab.
        if (x >= run_btn_x and x < run_btn_x + run_btn_w and y >= q_y and y < q_y + q_h and button == 0x110) {
            self.triggerFind();
            return false;
        }
        if (x >= cancel_btn_x and x < cancel_btn_x + cancel_btn_w and y >= q_y and y < q_y + q_h and button == 0x110) {
            self.handleCancelQuery();
            return false;
        }

        // Clear the selected query
        if (x >= clear_btn_x and x < clear_btn_x + clear_btn_w and y >= q_y and y < q_y + q_h and button == 0x110) {
            self.handleClearQuery();
            return false;
        }

        self.search_focused = false;
        self.search_anchor = null;

        // 4. Video Playback Controls (below canvas)
        const video_bar_y: usize = self.client.height -| 44;
        if (self.video_active and y >= video_bar_y and y < video_bar_y + 28 and button == 0x110) {
            if (x >= 16 and x < 86) {
                self.toggleVideoPlay();
                return false;
            }
            if (x >= 94 and x < 174) {
                self.seekVideo(0);
                return false;
            }
            if (x >= 182 and x < 286) {
                self.stepVideo();
                return false;
            }
            const s_x: usize = 294;
            const time_w: usize = 120;
            const s_w = stride -| (s_x + time_w + 16);
            if (s_w > 0 and x >= s_x and x < s_x + s_w) {
                const target = self.video_duration * @as(f64, @floatFromInt(x - s_x)) / @as(f64, @floatFromInt(s_w));
                self.seekVideo(target);
                return false;
            }
        }

        return false;
    }

    fn updateCursor(self: *App, px: f32, py: f32) !void {
        const x: usize = @intFromFloat(@max(0, px));
        const y: usize = @intFromFloat(@max(0, py));
        const edges = self.getResizeEdge(x, y);
        const q_w = self.client.width -| 404;
        const kind: wayland.Cursor = if (getResizeCursor(edges)) |c|
            c
        else if (x >= 16 and x < 16 + q_w and y >= 80 and y < 134)
            .text
        else
            .arrow;
        try self.client.setCursor(kind);
    }

    fn adjustSearchScroll(self: *App) void {
        const visible_chars: usize = 37;
        if (self.search_caret < self.search_scroll) self.search_scroll = self.search_caret;
        if (self.search_caret > self.search_scroll + visible_chars) {
            self.search_scroll = self.search_caret - visible_chars;
        }
    }

    fn selection(self: *App) ?struct { start: usize, end: usize } {
        const anchor = self.search_anchor orelse return null;
        if (anchor == self.search_caret) return null;
        return .{ .start = @min(anchor, self.search_caret), .end = @max(anchor, self.search_caret) };
    }

    fn deleteSelection(self: *App) bool {
        const selected = self.selection() orelse return false;
        std.mem.copyForwards(u8, self.search_text[selected.start..], self.search_text[selected.end..self.search_len]);
        self.search_len -= selected.end - selected.start;
        self.search_caret = selected.start;
        self.search_anchor = null;
        self.adjustSearchScroll();
        return true;
    }

    fn moveSearchCaret(self: *App, pos: usize) void {
        if (self.shift_down) {
            if (self.search_anchor == null) self.search_anchor = self.search_caret;
        } else {
            self.search_anchor = null;
        }
        self.search_caret = @min(pos, self.search_len);
        self.adjustSearchScroll();
    }

    fn isRepeatableKey(self: *const App, key: u32) bool {
        if (self.browser_open) {
            if (key == 14) return true; // Backspace
            if (key == 103 or key == 108) return true; // Up / Down
            if (!self.ctrl_down and evdevToChar(key, self.shift_down, self.caps_lock) != null) return true;
            return false;
        }
        if (self.search_focused) {
            if (!self.ctrl_down and (key == 14 or key == 111)) return true; // Backspace / Delete
            if (key == 105 or key == 106) return true; // Left / Right
            if (key == 28 and (self.shift_down or self.alt_down)) return true; // Shift/Alt+Enter (newline)
            if (!self.ctrl_down and evdevToChar(key, self.shift_down, self.caps_lock) != null) return true;
            return false;
        }
        return false;
    }

    fn queryCompletions(self: *App) vdb.completion.Matches {
        if (!self.completion_visible or !self.search_focused or self.browser_open or self.selection() != null) return .{};
        var matches = self.completion_cache;
        matches.len = @min(matches.len, 8);
        return matches;
    }

    fn refreshCompletions(self: *App) void {
        self.completion_cache.deinit(self.allocator);
        self.completion_cache = vdb.completion.complete(self.allocator, self.io, self.search_text[0..self.search_len], self.search_caret) catch .{};
    }

    fn completionWidth(self: *const App) usize {
        var width: usize = 220;
        for (self.completion_cache.items[0..@min(self.completion_cache.len, 8)]) |word| {
            width = @max(width, word.len * font.font_width + 20);
        }
        return @min(width, self.client.width -| 32);
    }

    fn acceptQueryCompletion(self: *App) void {
        const matches = self.queryCompletions();
        if (self.completion_index >= matches.len) return;
        const word = matches.items[self.completion_index];
        const new_len = self.search_len - (matches.end - matches.start) + word.len;
        if (new_len > self.search_text.len) return;
        const tail_len = self.search_len - matches.end;
        if (new_len > self.search_len) {
            std.mem.copyBackwards(u8, self.search_text[matches.start + word.len ..][0..tail_len], self.search_text[matches.end..self.search_len]);
        } else {
            std.mem.copyForwards(u8, self.search_text[matches.start + word.len ..][0..tail_len], self.search_text[matches.end..self.search_len]);
        }
        @memcpy(self.search_text[matches.start..][0..word.len], word);
        self.search_len = new_len;
        self.search_caret = matches.start + word.len;
        const directory = word.len >= 2 and word[word.len - 2] == '/' and (word[word.len - 1] == '\'' or word[word.len - 1] == '"');
        if (directory) self.search_caret -= 1;
        self.search_anchor = null;
        self.completion_visible = false;
        self.adjustSearchScroll();
        self.editQuery(self.search_text[0..self.search_len]);
        if (directory) {
            self.refreshCompletions();
            self.completion_index = 0;
            self.completion_visible = true;
        }
    }

    fn handleKey(self: *App, key: u32, state: u32) void {
        if (key == 42 or key == 54) {
            self.shift_down = state == 1;
            return;
        }
        if (key == 29 or key == 97) {
            self.ctrl_down = state == 1;
            return;
        }
        if (key == 56 or key == 100) {
            self.alt_down = state == 1;
            return;
        }
        if (state != 1) return;
        if (key == 58) {
            self.caps_lock = !self.caps_lock;
            return;
        }
        if (self.browser_open) {
            if (self.ctrl_down and key == 30) {
                self.browser_path_len = 0;
                return;
            }
            switch (key) {
                1 => self.browser_open = false, // Escape
                28 => self.browserOpenPath(self.browser_path[0..self.browser_path_len]), // Enter
                14 => self.browser_path_len -|= 1, // Backspace
                103 => { // Up
                    if (self.browser_scroll > 0) self.browser_scroll -= 1;
                },
                108 => { // Down
                    if (self.browser_scroll + browserVisibleRows(@min(self.client.height -| 190, 420)) < self.browser_entries.items.len) self.browser_scroll += 1;
                },
                else => if (!self.ctrl_down) {
                    if (evdevToChar(key, self.shift_down, self.caps_lock)) |ch| {
                        if (self.browser_path_len < self.browser_path.len) {
                            self.browser_path[self.browser_path_len] = ch;
                            self.browser_path_len += 1;
                        }
                    }
                },
            }
            return;
        }
        if (!self.search_focused) {
            if (key == 57) { // Space: Play/Pause video
                if (self.video_active) {
                    self.toggleVideoPlay();
                    return;
                }
            } else if (key == 19) { // R: Restart video
                if (self.video_active) {
                    self.seekVideo(0);
                    return;
                }
            } else if (key == 49 or key == 106) { // N or Right Arrow: Next match / Next frame
                if (self.video_active) {
                    self.stepVideo();
                    return;
                }
            } else if (key == 1) { // Escape
                if (self.selectedQueryActive()) {
                    self.handleCancelQuery();
                    return;
                }
            } else if (key == 53) { // /: Focus search
                self.search_focused = true;
                self.redraw_pending.store(true, .release);
                return;
            }
            return;
        }

        if (self.ctrl_down) {
            self.completion_visible = false;
            if (key == 30) { // Ctrl+A
                self.search_anchor = 0;
                self.search_caret = self.search_len;
                self.adjustSearchScroll();
            }
            return;
        }

        const suggestions = self.queryCompletions();
        if (suggestions.len > 0) {
            if (key == 15) { // Tab accepts; Return continues to run the query.
                self.acceptQueryCompletion();
                return;
            }
            if (key == 103 or key == 108) {
                self.completion_index = (self.completion_index + (if (key == 108) @as(usize, 1) else suggestions.len - 1)) % suggestions.len;
                return;
            }
            if (key == 1) {
                self.completion_visible = false;
                return;
            }
        }
        self.completion_visible = key == 14 or key == 111 or evdevToChar(key, self.shift_down, self.caps_lock) != null;
        self.completion_index = 0;
        switch (key) {
            28 => { // Enter
                if (self.shift_down or self.alt_down) {
                    if (self.search_len < self.search_text.len) {
                        _ = self.deleteSelection();
                        const pos = self.search_caret;
                        std.mem.copyBackwards(u8, self.search_text[pos + 1 .. self.search_len + 1], self.search_text[pos..self.search_len]);
                        self.search_text[pos] = '\n';
                        self.search_len += 1;
                        self.search_caret = pos + 1;
                    }
                } else {
                    self.triggerFind();
                }
            },
            1 => self.search_focused = false, // Escape
            105 => self.moveSearchCaret(self.search_caret -| 1), // Left
            106 => self.moveSearchCaret(self.search_caret + 1), // Right
            102 => self.moveSearchCaret(0), // Home
            107 => self.moveSearchCaret(self.search_len), // End
            14 => { // Backspace
                if (!self.deleteSelection() and self.search_caret > 0) {
                    const pos = self.search_caret - 1;
                    std.mem.copyForwards(u8, self.search_text[pos..], self.search_text[self.search_caret..self.search_len]);
                    self.search_len -= 1;
                    self.search_caret = pos;
                }
            },
            111 => { // Delete
                if (!self.deleteSelection() and self.search_caret < self.search_len) {
                    std.mem.copyForwards(u8, self.search_text[self.search_caret..], self.search_text[self.search_caret + 1 .. self.search_len]);
                    self.search_len -= 1;
                }
            },
            else => {
                if (evdevToChar(key, self.shift_down, self.caps_lock)) |ch| {
                    _ = self.deleteSelection();
                    if (self.search_len < self.search_text.len) {
                        std.mem.copyBackwards(u8, self.search_text[self.search_caret + 1 .. self.search_len + 1], self.search_text[self.search_caret..self.search_len]);
                        self.search_text[self.search_caret] = ch;
                        self.search_len += 1;
                        self.search_caret += 1;
                    }
                }
            },
        }
        self.adjustSearchScroll();
        self.editQuery(self.search_text[0..self.search_len]);
        self.refreshCompletions();
    }

    fn triggerFind(self: *App) void {
        if (self.is_busy or self.search_len == 0) return;
        self.editQuery(self.search_text[0..self.search_len]);
        self.completion_visible = false;
        self.handleQuery(self.search_text[0..self.search_len]);
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
        self.mutex.unlock(self.io);

        self.is_busy = true;
        self.setStatus("Segmenting… the first click on an image also runs the vision encoder.");

        const thread = std.Thread.spawn(.{}, runSegmentWorker, .{self}) catch {
            self.is_busy = false;
            return;
        };
        thread.detach();
    }

    fn runSegmentWorker(self: *App) void {
        self.model_mutex.lock(self.io) catch {
            self.is_busy = false;
            return;
        };
        defer self.model_mutex.unlock(self.io);
        const started = std.Io.Timestamp.now(self.io, .awake);

        var embedding = self.ensureEmbedding(false) catch |err| {
            log.info(self.io, "Vision encoder failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            self.setStatus("Vision encoder failed");
            self.is_busy = false;
            return;
        };
        defer embedding.deinit();

        const decode_started = std.Io.Timestamp.now(self.io, .awake);
        const masks = self.model.segment(&embedding, self.points[0..self.points_len]) catch |err| {
            log.info(self.io, "Decoder failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            self.setStatus("Segmentation failed");
            self.is_busy = false;
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

        const elapsed = started.untilNow(self.io, .awake);
        var status_buf: [128]u8 = undefined;
        const status = std.fmt.bufPrint(&status_buf, "{d} point(s) -> {d} masks in {f}", .{
            self.points_len,
            masks.count,
            elapsed,
        }) catch "Segmentation complete";
        self.setStatus(status);

        self.is_busy = false;
    }

    fn resolveVideoPath(allocator: std.mem.Allocator, input_path: []const u8) ?[]const u8 {
        const trimmed = std.mem.trim(u8, input_path, " \t\r\n'\"");
        if (trimmed.len == 0) return null;

        // 1. Direct check
        if (allocator.dupeZ(u8, trimmed)) |zpath| {
            defer allocator.free(zpath);
            if (std.c.access(zpath.ptr, 0) == 0) {
                return allocator.dupe(u8, trimmed) catch null;
            }
        } else |_| {}

        // 2. Expand ~/
        const maybe_home = std.c.getenv("HOME");
        if (std.mem.startsWith(u8, trimmed, "~/")) {
            if (maybe_home) |h| {
                const home = std.mem.span(h);
                if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed[2..] })) |expanded| {
                    if (allocator.dupeZ(u8, expanded)) |zpath| {
                        defer allocator.free(zpath);
                        if (std.c.access(zpath.ptr, 0) == 0) {
                            return expanded;
                        }
                    } else |_| {}
                    allocator.free(expanded);
                } else |_| {}
            }
        }

        // 3. Check in home directory (~/<trimmed>)
        if (maybe_home) |h| {
            const home = std.mem.span(h);
            if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed })) |in_home| {
                if (allocator.dupeZ(u8, in_home)) |zpath| {
                    defer allocator.free(zpath);
                    if (std.c.access(zpath.ptr, 0) == 0) {
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
                if (std.c.access(zpath.ptr, 0) == 0) {
                    return in_cwd;
                }
            } else |_| {}
            allocator.free(in_cwd);
        } else |_| {}

        return null;
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
            self.setStatus(status);
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
                self.search_len = 0;
                self.search_caret = 0;
                self.search_anchor = null;
                self.search_scroll = 0;
                self.completion_visible = false;
                self.setStatus("Query closed.");
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
            _ = self.openVideoFromPath(tab.query_path.?);
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
            self.redraw_pending.store(true, .release);
        }
        self.search_len = @min(tab.draft.len, self.search_text.len);
        @memcpy(self.search_text[0..self.search_len], tab.draft[0..self.search_len]);
        self.search_caret = self.search_len;
        self.search_anchor = null;
        self.search_focused = true;
        self.completion_visible = false;
        self.adjustSearchScroll();
        self.mutex.lock(self.io) catch return;
        self.setStatus(tab.status[0..tab.status_len]);
        self.mutex.unlock(self.io);
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
        self.redraw_pending.store(true, .release);
    }

    fn selectedIndexing(self: *const App) bool {
        return (self.selectedTab() orelse return false).precache_active.load(.acquire);
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
        self.search_len = 0;
        self.search_caret = 0;
        self.search_anchor = null;
        self.search_scroll = 0;
        self.editQuery("");
        self.setStatus("Query cleared.");
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
                    _ = self.openVideoFromPath(resolved);
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

        var video_adapter = LinuxVideoReader.init(self.allocator, source_path);
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

    fn handleFindText(self: *App, phrase: []const u8) void {
        const trimmed = std.mem.trim(u8, phrase, " \t\r\n");
        if (trimmed.len == 0) return;

        if (isQuerySql(trimmed)) {
            self.handleQuery(trimmed);
            return;
        }

        if (self.video_active) {
            self.mutex.lock(self.io) catch return;
            self.video_overlay_prompts = .{};
            self.video_phrase_len = @min(trimmed.len, self.video_phrase.len);
            @memcpy(self.video_phrase[0..self.video_phrase_len], trimmed[0..self.video_phrase_len]);
            const playing = self.video_playing.load(.acquire);
            if (!playing) {
                self.video_seek_target = self.video_position;
                self.video_step_requested = true;
            }
            self.mutex.unlock(self.io);
            self.setStatus(if (playing) "Processing video frames during playback…" else "Processing current video frame…");
            return;
        }
        self.is_busy = true;

        var status_buf: [256]u8 = undefined;
        const status = std.fmt.bufPrint(&status_buf, "Looking for “{s}”…", .{trimmed}) catch "Searching…";
        self.setStatus(status);

        const PhraseContext = struct {
            app: *App,
            phrase: []const u8,
        };
        const ctx = self.allocator.create(PhraseContext) catch {
            self.is_busy = false;
            return;
        };
        const phrase_copy = self.allocator.dupe(u8, trimmed) catch {
            self.allocator.destroy(ctx);
            self.is_busy = false;
            return;
        };
        ctx.* = .{ .app = self, .phrase = phrase_copy };

        const thread = std.Thread.spawn(.{}, runLookupWorker, .{ctx}) catch {
            self.allocator.free(phrase_copy);
            self.allocator.destroy(ctx);
            self.is_busy = false;
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
        self.model_mutex.lock(self.io) catch {
            self.is_busy = false;
            return;
        };
        defer self.model_mutex.unlock(self.io);
        const started = std.Io.Timestamp.now(self.io, .awake);

        const lookup_started = std.Io.Timestamp.now(self.io, .awake);
        const values = cachedQuery(self.allocator, sam3.RgbImage.fromImage(self.image.?), phrase) catch |err| {
            log.info(self.io, "Text lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
            self.setStatus("Lookup failed");
            self.is_busy = false;
            return;
        };
        defer self.allocator.free(values);
        const masks = unpackMasks(self.allocator, values) catch |err| {
            log.info(self.io, "Cached text lookup failed: {t}", .{err});
            self.setStatus("Cached lookup is invalid.");
            self.is_busy = false;
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

            var msg_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&msg_buf, "No objects matched “{s}”.", .{phrase}) catch "No objects matched.";
            self.setStatus(msg);
        } else {
            if (self.coverages.len != masks.count) {
                self.allocator.free(self.coverages);
                self.coverages = self.allocator.alloc(f32, masks.count) catch &.{};
            }
            self.best_mask_idx = @intCast(render.scoreMasks(masks.logits, masks.scores, masks.count, masks.width, masks.height, self.coverages));
            self.selected_mask = self.best_mask_idx;
            self.renderComposite(self.selected_mask);

            const elapsed = started.untilNow(self.io, .awake);
            var status_buf: [256]u8 = undefined;
            const msg = std.fmt.bufPrint(&status_buf, "{d} object(s) matched “{s}” in {f}", .{
                masks.count,
                phrase,
                elapsed,
            }) catch "Search complete";
            self.setStatus(msg);
        }

        self.is_busy = false;
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
        self.setStatus("Points and masks cleared.");
        self.redraw_pending.store(true, .release);
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
        self.video_playing.store(false, .release);
        self.video_phrase_len = 0;
        self.video_overlay_prompts = .{};
        self.video_duration = 0;
        self.video_position = 0;
        self.video_seek_target = null;
        self.video_step_requested = false;
        self.setWindowTitle("SAM 3 — Visual Database");
        self.redraw_pending.store(true, .release);
    }

    fn openVideoFromPath(self: *App, path: []const u8) bool {
        if (self.is_busy) return false;
        self.stopVideo();
        const owned = self.allocator.dupeZ(u8, path) catch return false;
        self.mutex.lock(self.io) catch {
            self.allocator.free(owned);
            return false;
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
        self.video_active = true;
        self.video_stop.store(false, .release);
        self.video_seek_target = 0;
        if (self.image) |*img| img.deinit(self.allocator);
        self.image = null;
        self.allocator.free(self.frame);
        self.frame = &.{};
        if (self.masks) |*m| m.deinit();
        self.masks = null;
        self.points_len = 0;
        self.selected_mask = -1;
        self.best_mask_idx = -1;
        self.restoreQueryOverlay();
        self.mutex.unlock(self.io);

        // Update window title with video file name
        const basename = std.fs.path.basename(path);
        var title_buf: [256]u8 = undefined;
        const title_str = std.fmt.bufPrint(&title_buf, "SAM 3 — Visual Database — {s}", .{basename}) catch "SAM 3 — Visual Database";
        self.setWindowTitle(title_str);

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
            const smsg = std.fmt.bufPrint(&sbuf, "Video opened. Loaded visual index with {d} concept(s) from .vdb sidecar.", .{indexed_concepts_count}) catch "Video opened with index.";
            self.setStatus(smsg);
        } else {
            self.setStatus("Video opened. Enter a word or SQL query and press Run Query.");
        }

        self.video_thread = std.Thread.spawn(.{}, runVideoWorker, .{self}) catch {
            self.stopVideo();
            self.setStatus("Could not start video decoder.");
            return false;
        };
        return true;
    }

    fn toggleVideoPlay(self: *App) void {
        const playing = !self.video_playing.load(.acquire);
        self.video_playing.store(playing, .release);
        if (!playing) self.setStatus("Video paused.");
        self.redraw_pending.store(true, .release);
    }

    fn seekVideo(self: *App, seconds: f64) void {
        if (!self.video_active or !std.math.isFinite(seconds)) return;
        self.mutex.lock(self.io) catch return;
        self.video_seek_target = std.math.clamp(seconds, 0, self.video_duration);
        self.video_step_requested = true;
        self.mutex.unlock(self.io);
        self.redraw_pending.store(true, .release);
    }

    fn stepVideo(self: *App) void {
        if (!self.video_active) return;
        self.mutex.lock(self.io) catch return;
        if (self.isQueryVideo() and self.queryMatches().len > 0) {
            self.selectedTab().?.query_match_idx = (self.selectedTab().?.query_match_idx + 1) % self.queryMatches().len;
            const target_frame = self.queryMatches()[self.selectedTab().?.query_match_idx];
            const target_sec = self.selectedTab().?.matchTime(self.selectedTab().?.query_match_idx);
            self.video_seek_target = target_sec;
            self.video_step_requested = true;
            const cur = self.selectedTab().?.query_match_idx + 1;
            const total = self.queryMatches().len;
            self.mutex.unlock(self.io);
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "Showing match {d} of {d} (Frame #{d} at {d:.2}s).", .{ cur, total, target_frame, target_sec }) catch "Next match.";
            self.setStatus(msg);
            return;
        }
        self.video_step_requested = true;
        self.mutex.unlock(self.io);
    }

    fn runVideoWorker(self: *App) void {
        var reader: ?*anyopaque = null;
        defer if (reader) |r| sam_linux_video_close(r);
        var playback_start: ?std.Io.Timestamp = null;
        var playback_first_pts: f64 = 0;
        var previous_mask_pts: ?f64 = null;
        var previous_mask_display: ?std.Io.Timestamp = null;
        var retry_required = false;
        var open_at: f64 = 0;
        while (!self.video_stop.load(.acquire)) {
            self.mutex.lock(self.io) catch return;
            const seek = self.video_seek_target;
            self.video_seek_target = null;
            const step = self.video_step_requested;
            self.video_step_requested = false;
            var phrase_buf: [256]u8 = undefined;
            const has_query_matches = self.queryMatches().len > 0;
            const overlay_prompts = self.video_overlay_prompts;
            const phrase_len = self.video_phrase_len;
            @memcpy(phrase_buf[0..phrase_len], self.video_phrase[0..phrase_len]);
            self.mutex.unlock(self.io);
            if (phrase_len == 0 and overlay_prompts.count == 0 and !has_query_matches and !step and seek == null) continue;
            const phrase = phrase_buf[0..phrase_len];
            if (seek) |target| {
                if (reader) |r| sam_linux_video_close(r);
                reader = null;
                open_at = target;
                playback_start = null;
                previous_mask_pts = null;
                previous_mask_display = null;
            }
            if (reader == null and (!retry_required or self.video_playing.load(.acquire) or step or seek != null)) {
                reader = sam_linux_video_open(self.video_path.?.ptr, open_at);
                retry_required = true;
                if (reader) |r| {
                    self.mutex.lock(self.io) catch return;
                    self.video_duration = sam_linux_video_duration(r);
                    self.mutex.unlock(self.io);
                } else {
                    self.video_playing.store(false, .release);
                    self.setStatus("Could not decode video. Install FFmpeg and ffprobe.");
                }
            }
            if (reader == null or (!self.video_playing.load(.acquire) and !step and seek == null)) {
                if (!self.video_playing.load(.acquire)) {
                    playback_start = null;
                    previous_mask_pts = null;
                    previous_mask_display = null;
                }
                std.Io.sleep(self.io, .fromMilliseconds(20), .awake) catch {};
                continue;
            }
            var raw: VideoFrame = .{ .rgb = null, .width = 0, .height = 0, .pts_seconds = 0 };
            const result = sam_linux_video_next(reader.?, &raw);
            if (result <= 0) {
                sam_linux_video_close(reader.?);
                reader = null;
                self.video_playing.store(false, .release);
                self.setStatus(if (result == 0) "End of video. Press Play to replay." else "Video decoding failed.");
                open_at = 0;
                playback_start = null;
                previous_mask_pts = null;
                previous_mask_display = null;
                continue;
            }
            defer sam_linux_video_free_frame(&raw);
            if (!self.video_playing.load(.acquire) and !step and seek == null) {
                playback_start = null;
                continue;
            }
            if (phrase_len == 0 and overlay_prompts.count == 0 and self.video_playing.load(.acquire) and !step and std.math.isFinite(raw.pts_seconds)) {
                if (playback_start == null) {
                    playback_start = std.Io.Timestamp.now(self.io, .awake);
                    playback_first_pts = raw.pts_seconds;
                } else {
                    const offset = raw.pts_seconds - playback_first_pts;
                    if (offset >= 0 and offset < 1.0e6) {
                        const target_ns: i96 = @intFromFloat(offset * 1e9);
                        var elapsed_ns = playback_start.?.untilNow(self.io, .awake).nanoseconds;
                        if (elapsed_ns > target_ns + 70_000_000) continue;
                        while (target_ns > elapsed_ns and self.video_playing.load(.acquire) and !self.video_stop.load(.acquire)) {
                            std.Io.sleep(self.io, .fromNanoseconds(@min(target_ns - elapsed_ns, 20_000_000)), .awake) catch {};
                            elapsed_ns = playback_start.?.untilNow(self.io, .awake).nanoseconds;
                        }
                        if (!self.video_playing.load(.acquire)) continue;
                    }
                }
            }
            const width: usize = @intCast(raw.width);
            const height: usize = @intCast(raw.height);
            const rgb_len = std.math.mul(usize, std.math.mul(usize, width, height) catch continue, 3) catch continue;
            const pixels = self.allocator.dupe(u8, raw.rgb.?[0..rgb_len]) catch continue;
            var decoded = zigimg.Image.fromRawPixelsOwned(width, height, pixels, .rgb24) catch {
                self.allocator.free(pixels);
                continue;
            };
            var decoded_owned = true;
            defer if (decoded_owned) decoded.deinit(self.allocator);
            const show_preview = (phrase_len == 0 and overlay_prompts.count == 0) or step or seek != null;
            if (self.video_stop.load(.acquire)) break;
            if (show_preview) {
                const frame_len = std.math.mul(usize, std.math.mul(usize, width, height) catch continue, 4) catch continue;
                const next_frame = self.allocator.alloc(u8, frame_len) catch continue;
                self.mutex.lock(self.io) catch {
                    self.allocator.free(next_frame);
                    return;
                };
                if (self.video_seek_target != null) {
                    self.mutex.unlock(self.io);
                    self.allocator.free(next_frame);
                    continue;
                }
                if (self.image) |*old| old.deinit(self.allocator);
                self.allocator.free(self.frame);
                if (self.masks) |*old| old.deinit();
                self.image = decoded;
                decoded_owned = false;
                self.frame = next_frame;
                self.masks = null;
                self.points_len = 0;
                self.selected_mask = -1;
                self.best_mask_idx = -1;
                self.video_position = raw.pts_seconds;
                self.renderComposite(-1);
                self.redraw_pending.store(true, .release);
                self.mutex.unlock(self.io);
            }
            if (phrase_len == 0 and overlay_prompts.count == 0 and !has_query_matches) {
                previous_mask_pts = null;
                previous_mask_display = null;
                var preview_status: [96]u8 = undefined;
                self.setStatus(std.fmt.bufPrint(&preview_status, "Video at {d:.2}s", .{raw.pts_seconds}) catch "Video playing.");
                continue;
            }
            playback_start = null;
            if (show_preview) self.setStatus("Processing current video frame…");
            var masks: ?sam3.Masks = null;
            defer if (masks) |*m| m.deinit();
            const lookup_started = std.Io.Timestamp.now(self.io, .awake);
            masks = self.videoOverlayMasks(sam3.RgbImage.fromImage(decoded), phrase, &overlay_prompts, raw.pts_seconds) catch |err| {
                log.info(self.io, "Video frame lookup failed: {t}: {s}", .{ err, sam3.onnx.lastError() });
                self.video_playing.store(false, .release);
                self.setStatus("Video frame inference failed.");
                continue;
            };
            const mask_result_valid = masks != null;
            const lookup_elapsed = lookup_started.untilNow(self.io, .awake);
            if (self.video_playing.load(.acquire) and !step and std.math.isFinite(raw.pts_seconds)) {
                if (previous_mask_pts) |pts| {
                    const interval = raw.pts_seconds - pts;
                    if (interval > 0 and interval < 1 and std.math.isFinite(interval)) {
                        const target_ns: i96 = @intFromFloat(interval * 1e9);
                        var elapsed_ns = previous_mask_display.?.untilNow(self.io, .awake).nanoseconds;
                        while (target_ns > elapsed_ns and self.video_playing.load(.acquire) and !self.video_stop.load(.acquire)) {
                            std.Io.sleep(self.io, .fromNanoseconds(@min(target_ns - elapsed_ns, 20_000_000)), .awake) catch {};
                            elapsed_ns = previous_mask_display.?.untilNow(self.io, .awake).nanoseconds;
                        }
                    }
                }
            }
            if (self.video_stop.load(.acquire)) break;
            const next_frame: ?[]u8 = if (show_preview) null else blk: {
                const frame_len = std.math.mul(usize, std.math.mul(usize, width, height) catch continue, 4) catch continue;
                break :blk self.allocator.alloc(u8, frame_len) catch continue;
            };
            self.mutex.lock(self.io) catch {
                if (next_frame) |frame| self.allocator.free(frame);
                return;
            };
            if (self.video_seek_target != null or (!self.video_playing.load(.acquire) and !step and seek == null)) {
                self.mutex.unlock(self.io);
                if (next_frame) |frame| self.allocator.free(frame);
                continue;
            }
            if (next_frame) |frame| {
                if (self.image) |*old| old.deinit(self.allocator);
                self.allocator.free(self.frame);
                if (self.masks) |*old| old.deinit();
                self.image = decoded;
                decoded_owned = false;
                self.frame = frame;
                self.points_len = 0;
            }
            self.masks = masks;
            masks = null;
            if (self.masks) |m| {
                if (self.coverages.len != m.count) {
                    self.allocator.free(self.coverages);
                    self.coverages = self.allocator.alloc(f32, m.count) catch &.{};
                }
                self.best_mask_idx = if (m.count == 0) -1 else @intCast(render.scoreMasks(m.logits, m.scores, m.count, m.width, m.height, self.coverages));
            } else self.best_mask_idx = -1;
            self.selected_mask = self.best_mask_idx;
            self.video_position = raw.pts_seconds;
            self.renderComposite(self.selected_mask);
            previous_mask_pts = if (std.math.isFinite(raw.pts_seconds)) raw.pts_seconds else null;
            previous_mask_display = std.Io.Timestamp.now(self.io, .awake);
            const mask_count = if (self.masks) |m| m.count else @as(usize, 0);
            self.mutex.unlock(self.io);
            const seconds = if (std.math.isFinite(raw.pts_seconds)) @max(raw.pts_seconds, 0) else 0;
            const position_ms: u64 = @intFromFloat(@min(seconds * 1000, 1.0e15));
            var time_buf: [32]u8 = undefined;
            const timecode = std.fmt.bufPrint(&time_buf, "{d}:{d:0>2}.{d:0>3}", .{
                position_ms / 60_000,
                position_ms / 1_000 % 60,
                position_ms % 1_000,
            }) catch "0:00.000";
            var status_buf: [256]u8 = undefined;
            const prompt_display = if (overlay_prompts.count > 1) "query concepts" else phrase;
            const status = if (phrase_len == 0 and overlay_prompts.count == 0)
                std.fmt.bufPrint(&status_buf, "At {s}: frame shown.", .{timecode}) catch "Frame shown."
            else if (mask_result_valid)
                std.fmt.bufPrint(&status_buf, "At {s}: {d} match(es) for “{s}” in {f}", .{ timecode, mask_count, prompt_display, lookup_elapsed }) catch "Video frame processed."
            else
                std.fmt.bufPrint(&status_buf, "At {s}: waiting for cached “{s}” result.", .{ timecode, prompt_display }) catch "Frame shown without a cached result.";
            self.setStatus(status);
            if (mask_result_valid) log.info(self.io, "at {s}: \"{s}\" -> {d} object(s) in {f}", .{
                timecode, prompt_display, mask_count, lookup_elapsed,
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
                    var match_buf: [160]u8 = undefined;
                    const match_status = std.fmt.bufPrint(&match_buf, "Match {d}/{d}: Frame #{d} ({d:.2}s)", .{
                        cur_num,
                        total_matches,
                        next_match_frame,
                        next_sec,
                    }) catch "Advancing matches…";
                    self.setStatus(match_status);
                    std.Io.sleep(self.io, .fromMilliseconds(250), .awake) catch {};
                    continue;
                }
                self.mutex.unlock(self.io);
            }
        }
    }

    fn cachedFrameQuery(self: *App, image: sam3.RgbImage, phrase: []const u8) ![]f32 {
        return cachedQuery(self.allocator, image, phrase);
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
        const reader = sam_linux_video_open(source_path.ptr, 0) orelse {
            self.setQueryStatus(tab, "Could not decode video for pre-caching.");
            return;
        };
        defer sam_linux_video_close(reader);
        const duration = sam_linux_video_duration(reader);
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
            const next = sam_linux_video_next(reader, &frame);
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
            defer sam_linux_video_free_frame(&frame);
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
            if (self.video_active and self.points_len == 0) {
                for (img.pixels.rgb24, 0..) |pixel, i| {
                    self.frame[i * 4] = pixel.r;
                    self.frame[i * 4 + 1] = pixel.g;
                    self.frame[i * 4 + 2] = pixel.b;
                    self.frame[i * 4 + 3] = 255;
                }
            } else {
                render.compositeRgba(self.allocator, img, self.frame, null, 0, 0, self.points[0..self.points_len]);
            }
        } else {
            render.compositeRgba(self.allocator, img, self.frame, null, 0, 0, &.{});
            const masks = self.masks.?;
            const selected: usize = @intCast(mask_index);
            for (0..masks.count) |i| {
                if (i != selected) self.overlayMask(img, masks, i, 0.25);
            }
            if (selected < masks.count) self.overlayMask(img, masks, selected, 0.5);
            self.drawPointMarkers(img);
        }
        self.redraw_pending.store(true, .release);
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

    fn redraw(self: *App) void {
        // The event loop holds mutex for the entire frame, including query state.
        const pixels = self.client.pixels;
        const stride = self.client.width;
        const h = self.client.height;

        // Background: #14161a
        @memset(pixels, 0x0014161a);

        // Window border (1px) when not maximized
        if (!self.is_maximized and stride > 2 and h > 2) {
            font.strokeRect(pixels, stride, 0, 0, stride, h, 0x00323742);
        }

        // 1. Header / Title bar (y: 0..32)
        font.fillRect(pixels, stride, 0, 0, stride, 32, 0x0017191e);
        font.fillRect(pixels, stride, 0, 31, stride, 1, 0x002c3038);
        const title = if (self.window_title_len > 0) self.window_title[0..self.window_title_len] else "SAM 3 — Visual Database";
        font.drawText(pixels, stride, title, 16, 7, 0x00d0d4dc);

        if (stride >= 110) {
            // Minimize [-]
            font.drawButton(pixels, stride, stride - 100, 5, 26, 22, "-", false, false, 0x0000dc64);
            // Maximize [+] / [=]
            font.drawButton(pixels, stride, stride - 68, 5, 26, 22, if (self.is_maximized) "=" else "+", false, false, 0x0000dc64);
            // Close [x]
            font.drawButton(pixels, stride, stride - 36, 5, 26, 22, "x", false, false, 0x00e05555);
        }

        const tab_count = self.query_tabs.items.len;
        if (tab_count > 0) {
            const selected = self.selected_query orelse 0;
            const visible = @max(1, (stride -| 128) / 160);
            const first = @min(selected, tab_count -| visible);
            font.drawButton(pixels, stride, 16, 40, 24, 28, "<", false, false, 0x008892a0);
            font.drawButton(pixels, stride, stride -| 72, 40, 24, 28, ">", false, false, 0x008892a0);
            for (first..@min(tab_count, first + visible)) |index| {
                const tab = self.query_tabs.items[index];
                const state = if (!tab.has_run) "new" else if (tab.query_active.load(.acquire))
                    if (tab.query_cancel.load(.acquire)) "cancelling" else "running"
                else if (tab.query_cancel.load(.acquire)) "cancelled" else "done";
                var label_buf: [48]u8 = undefined;
                const label = std.fmt.bufPrint(&label_buf, "Q{d} {s}", .{ index + 1, state }) catch "Query";
                const tab_x = 48 + (index - first) * 160;
                font.drawButton(pixels, stride, tab_x, 40, 128, 28, label, index == selected, tab.query_active.load(.acquire), 0x0000dc64);
                font.drawButton(pixels, stride, tab_x + 128, 40, 24, 28, "x", false, false, 0x008892a0);
            }
        } else {
            font.drawText(pixels, stride, "New query", 16, 46, 0x0068707c);
        }

        font.drawButton(pixels, stride, stride -| 40, 40, 24, 28, "+", false, false, 0x0000dc64);

        // SQL query console
        const q_x: usize = 16;
        const q_y: usize = 80;
        const q_h: usize = 54;
        const btn_gap: usize = 8;
        const run_btn_w: usize = 124;
        const clear_btn_w: usize = 114;
        const cancel_btn_w: usize = 110;
        const total_btns_w: usize = run_btn_w + 2 * btn_gap + cancel_btn_w + clear_btn_w;
        const q_w: usize = stride -| (q_x + total_btns_w + 24);
        const run_btn_x: usize = stride -| (total_btns_w + 16);
        const cancel_btn_x: usize = run_btn_x + run_btn_w + btn_gap;
        const clear_btn_x: usize = cancel_btn_x + cancel_btn_w + btn_gap;

        font.fillRect(pixels, stride, q_x, q_y, q_w, q_h, if (self.search_focused) 0x0023272e else 0x001c1f25);
        font.strokeRect(pixels, stride, q_x, q_y, q_w, q_h, if (self.search_focused) 0x0000dc64 else 0x00323742);

        const max_cols = if (q_w > 20) (q_w - 20) / font.font_width else 10;
        if (self.search_len == 0) {
            font.drawText(pixels, stride, "SELECT frame FROM 'video.mp4' WHERE frame_id BETWEEN 100 AND 200", q_x + 10, q_y + 10, 0x0068707c);
            font.drawText(pixels, stride, "Use FROM to choose this tab's video", q_x + 10, q_y + 30, 0x0068707c);
        } else {
            var line1_end: usize = 0;
            var line2_start: usize = 0;
            if (std.mem.indexOfScalar(u8, self.search_text[0..self.search_len], '\n')) |nl| {
                line1_end = @min(nl, max_cols);
                line2_start = nl + 1;
            } else {
                line1_end = @min(self.search_len, max_cols);
                line2_start = line1_end;
            }
            font.drawText(pixels, stride, self.search_text[0..line1_end], q_x + 10, q_y + 10, 0x00f2f4f6);
            if (line2_start < self.search_len) {
                var line2_end = self.search_len;
                if (std.mem.indexOfScalar(u8, self.search_text[line2_start..self.search_len], '\n')) |nl2| {
                    line2_end = line2_start + nl2;
                }
                const l2_len = @min(line2_end - line2_start, max_cols);
                font.drawText(pixels, stride, self.search_text[line2_start .. line2_start + l2_len], q_x + 10, q_y + 30, 0x00f2f4f6);
            }
        }
        if (self.search_focused) {
            var caret_row: usize = 0;
            var caret_col: usize = self.search_caret;
            if (std.mem.indexOfScalar(u8, self.search_text[0..self.search_len], '\n')) |nl| {
                if (self.search_caret > nl) {
                    caret_row = 1;
                    caret_col = self.search_caret - (nl + 1);
                }
            } else if (self.search_caret > max_cols) {
                caret_row = 1;
                caret_col = self.search_caret - max_cols;
            }
            caret_col = @min(caret_col, max_cols);
            const caret_x = q_x + 10 + caret_col * font.font_width;
            const caret_y = if (caret_row == 0) q_y + 8 else q_y + 28;
            font.fillRect(pixels, stride, caret_x, caret_y, 1, 18, 0x0000dc64);
        }

        const is_querying = self.selectedQueryActive();
        font.drawButton(pixels, stride, run_btn_x, q_y, run_btn_w, q_h, "Run Query", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, cancel_btn_x, q_y, cancel_btn_w, q_h, "Cancel Query", false, is_querying, if (is_querying) 0x00e05555 else 0x008892a0);
        font.drawButton(pixels, stride, clear_btn_x, q_y, clear_btn_w, q_h, "Clear Query", false, false, 0x008892a0);

        // Action toolbar
        const tb_y: usize = 146;

        const precaching = self.selectedIndexing();
        const tool_x: usize = 16;
        if (precaching) {
            font.fillRect(pixels, stride, tool_x, tb_y + 11, 100, 6, 0x002c3038);
            font.fillRect(pixels, stride, tool_x, tb_y + 11, @intFromFloat(100.0 * self.selectedTab().?.precache_progress), 6, 0x0000b8ff);
            var pct_buf: [32]u8 = undefined;
            const pct_str = std.fmt.bufPrint(&pct_buf, "Indexing {d:.1}%", .{self.selectedTab().?.precache_progress * 100.0}) catch "Indexing…";
            font.drawText(pixels, stride, pct_str, tool_x + 108, tb_y + 6, 0x0000b8ff);
        }

        // Selected query status
        font.drawText(pixels, stride, self.status_text[0..self.status_len], 16, 180, 0x00969ba5);

        // 5. Layout geometry for Canvas and Controls Below Canvas
        const video_bar_y: usize = h -| 44;
        const cy: usize = 202;
        const canvas_bottom: usize = if (self.video_active) video_bar_y -| 8 else h -| 16;
        const ch = if (canvas_bottom > cy) canvas_bottom - cy else 100;
        const cw = if (stride > 32) stride - 32 else 100;
        self.canvas_x = 16;
        self.canvas_y = cy;
        self.canvas_w = cw;
        self.canvas_h = ch;

        // 6. Video Playback Controls (Directly below canvas)
        if (self.video_active) {
            font.drawButton(pixels, stride, 16, video_bar_y, 70, 28, if (self.video_playing.load(.acquire)) "Pause" else "Play", false, false, 0x0000dc64);
            font.drawButton(pixels, stride, 94, video_bar_y, 80, 28, "Restart", false, false, 0x0000dc64);

            const next_label = if (self.isQueryVideo() and self.queryMatches().len > 0) "Next Match" else "Next Frame";
            font.drawButton(pixels, stride, 182, video_bar_y, 104, 28, next_label, false, false, 0x0000dc64);

            const s_x: usize = 294;
            const time_w: usize = 120;
            const s_w = stride -| (s_x + time_w + 16);
            const s_y = video_bar_y + 10;
            if (s_w > 20) {
                font.fillRect(pixels, stride, s_x, s_y, s_w, 8, 0x002c3038);
                const frac = if (self.video_duration > 0) std.math.clamp(self.video_position / self.video_duration, 0, 1) else 0;
                font.fillRect(pixels, stride, s_x, s_y, @intFromFloat(@as(f64, @floatFromInt(s_w)) * frac), 8, 0x0000dc64);
                const knob_x = s_x + @as(usize, @intFromFloat(@as(f64, @floatFromInt(s_w -| 8)) * frac));
                font.fillRect(pixels, stride, knob_x, s_y - 3, 8, 14, 0x00ffffff);
                if (precaching) {
                    font.fillRect(pixels, stride, s_x, s_y + 10, @intFromFloat(@as(f64, @floatFromInt(s_w)) * self.selectedTab().?.precache_progress), 3, 0x0000b8ff);
                }

                const cur_sec = @as(u32, @intFromFloat(@max(self.video_position, 0)));
                const dur_sec = @as(u32, @intFromFloat(@max(self.video_duration, 0)));
                var time_buf: [32]u8 = undefined;
                const time_str = std.fmt.bufPrint(&time_buf, "{d}:{d:0>2} / {d}:{d:0>2}", .{ cur_sec / 60, cur_sec % 60, dur_sec / 60, dur_sec % 60 }) catch "0:00 / 0:00";
                font.drawText(pixels, stride, time_str, s_x + s_w + 12, video_bar_y + 5, 0x00d0d4dc);
            }
        }

        // 7. Canvas Area
        const cx = self.canvas_x;
        font.fillRect(pixels, stride, cx, cy, cw, ch, 0x001c1f25);
        font.strokeRect(pixels, stride, cx, cy, cw, ch, 0x002c3038);

        if (self.image) |img| {
            if (img.width > 0 and img.height > 0) {
                const scale_x = @as(f32, @floatFromInt(cw)) / @as(f32, @floatFromInt(img.width));
                const scale_y = @as(f32, @floatFromInt(ch)) / @as(f32, @floatFromInt(img.height));
                const scale = @min(scale_x, scale_y);

                const dw: usize = @intFromFloat(@as(f32, @floatFromInt(img.width)) * scale);
                const dh: usize = @intFromFloat(@as(f32, @floatFromInt(img.height)) * scale);
                const dx: usize = cx + (cw - dw) / 2;
                const dy: usize = cy + (ch - dh) / 2;

                self.img_rect_x = dx;
                self.img_rect_y = dy;
                self.img_rect_w = dw;
                self.img_rect_h = dh;

                // Blit frame buffer to window
                const frame_bytes = self.frame;
                for (0..dh) |py| {
                    const src_y = (py * img.height) / dh;
                    const row_out = (dy + py) * stride;
                    for (0..dw) |px| {
                        const src_x = (px * img.width) / dw;
                        const src_idx = (src_y * img.width + src_x) * 4;
                        const r = frame_bytes[src_idx + 0];
                        const g = frame_bytes[src_idx + 1];
                        const b = frame_bytes[src_idx + 2];
                        const color: u32 = (@as(u32, r) << 16) | (@as(u32, g) << 8) | b;
                        pixels[row_out + dx + px] = color;
                    }
                }
            }
        }

        const suggestions = self.queryCompletions();
        if (suggestions.len > 0) {
            const width = self.completionWidth();
            const columns = (width -| 20) / font.font_width;
            for (suggestions.items[0..suggestions.len], 0..) |word, i| {
                const color: u32 = if (i == self.completion_index) 0x0000dc64 else 0x00f2f4f6;
                font.fillRect(pixels, stride, 16, 136 + i * 24, width, 24, if (i == self.completion_index) 0x00323742 else 0x0023272e);
                if (word.len > columns and columns > 3) {
                    font.drawText(pixels, stride, "...", 26, 140 + i * 24, color);
                    font.drawText(pixels, stride, word[word.len - (columns - 3) ..], 26 + 3 * font.font_width, 140 + i * 24, color);
                } else {
                    font.drawText(pixels, stride, word[0..@min(word.len, columns)], 26, 140 + i * 24, color);
                }
            }
        }
        if (self.browser_open) self.drawBrowser(pixels, stride, h);
    }

    fn drawBrowser(self: *App, pixels: []u32, stride: usize, height: usize) void {
        const x: usize = 16;
        const y: usize = 140;
        const w = @min(stride -| 32, 640);
        const h = @min(height -| 190, 420);
        if (w < 200 or h < 130) return;
        font.fillRect(pixels, stride, x, y, w, h, 0x0023272e);
        font.strokeRect(pixels, stride, x, y, w, h, 0x0000dc64);
        font.drawButton(pixels, stride, x + 12, y + 8, 80, 28, "Parent", false, false, 0x0000dc64);
        font.drawText(pixels, stride, "Choose image or video", x + 104, y + 14, 0x00f2f4f6);
        font.drawButton(pixels, stride, x + w - 92, y + 8, 80, 28, "Cancel", false, false, 0x0000dc64);
        font.fillRect(pixels, stride, x + 12, y + 46, w - 24, 28, 0x001c1f25);
        font.strokeRect(pixels, stride, x + 12, y + 46, w - 24, 28, 0x0000dc64);
        const path = self.browser_path[0..self.browser_path_len];
        const max_chars = (w - 42) / font.font_width;
        font.drawText(pixels, stride, if (path.len == 0) "Type an absolute path" else path[path.len -| max_chars..], x + 20, y + 52, if (path.len == 0) 0x008e949e else 0x00f2f4f6);
        const rows = browserVisibleRows(h);
        for (0..rows) |row| {
            const index = self.browser_scroll + row;
            if (index >= self.browser_entries.items.len) break;
            const entry = self.browser_entries.items[index];
            const ry = y + 82 + row * 24;
            font.fillRect(pixels, stride, x + 12, ry, w - 24, 22, if (row % 2 == 0) 0x001c1f25 else 0x0023272e);
            const max_name = (w - 60) / font.font_width;
            font.drawText(pixels, stride, if (entry.is_dir) "/" else " ", x + 18, ry + 3, 0x0000dc64);
            font.drawText(pixels, stride, entry.name[0..@min(entry.name.len, max_name)], x + 30, ry + 3, 0x00e6e8ec);
        }
        font.drawButton(pixels, stride, x + 12, y + h - 36, 80, 28, "Up", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, x + 100, y + h - 36, 80, 28, "Down", false, false, 0x0000dc64);
        font.drawButton(pixels, stride, x + w - 100, y + h - 36, 88, 28, "Open Path", false, false, 0x0000dc64);
    }
};

fn browserVisibleRows(height: usize) usize {
    return (height -| 124) / 24;
}

fn isVideoPath(path: []const u8) bool {
    const ext = std.fs.path.extension(path);
    inline for (.{ ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpeg", ".mpg", ".3gp", ".ts", ".mts", ".flv", ".wmv" }) |video_ext| {
        if (std.ascii.eqlIgnoreCase(ext, video_ext)) return true;
    }
    return false;
}

fn clampMaskIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0) return 0;
    return @min(@as(usize, @intFromFloat(@floor(coordinate))), limit - 1);
}

fn blendChannel(original: u8, tint: u8, alpha: f32) u8 {
    return @intFromFloat(@as(f32, @floatFromInt(original)) * (1 - alpha) + @as(f32, @floatFromInt(tint)) * alpha);
}

fn evdevToChar(key: u32, shift: bool, caps: bool) ?u8 {
    const ch: u8 = switch (key) {
        16 => 'q',
        17 => 'w',
        18 => 'e',
        19 => 'r',
        20 => 't',
        21 => 'y',
        22 => 'u',
        23 => 'i',
        24 => 'o',
        25 => 'p',
        30 => 'a',
        31 => 's',
        32 => 'd',
        33 => 'f',
        34 => 'g',
        35 => 'h',
        36 => 'j',
        37 => 'k',
        38 => 'l',
        44 => 'z',
        45 => 'x',
        46 => 'c',
        47 => 'v',
        48 => 'b',
        49 => 'n',
        50 => 'm',
        2 => '1',
        3 => '2',
        4 => '3',
        5 => '4',
        6 => '5',
        7 => '6',
        8 => '7',
        9 => '8',
        10 => '9',
        11 => '0',
        12 => '-',
        13 => '=',
        26 => '[',
        27 => ']',
        39 => ';',
        40 => '\'',
        41 => '`',
        43 => '\\',
        51 => ',',
        52 => '.',
        53 => '/',
        57 => ' ',
        else => return null,
    };
    if (ch >= 'a' and ch <= 'z') return if (shift != caps) ch - 32 else ch;
    if (!shift) return ch;
    return switch (ch) {
        '1' => '!',
        '2' => '@',
        '3' => '#',
        '4' => '$',
        '5' => '%',
        '6' => '^',
        '7' => '&',
        '8' => '*',
        '9' => '(',
        '0' => ')',
        '-' => '_',
        '=' => '+',
        '[' => '{',
        ']' => '}',
        ';' => ':',
        '\'' => '"',
        '`' => '~',
        '\\' => '|',
        ',' => '<',
        '.' => '>',
        '/' => '?',
        else => ch,
    };
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

fn cachedQuery(allocator: std.mem.Allocator, image: sam3.RgbImage, phrase: []const u8) ![]f32 {
    return here.call(.computeQuery, queryArgs(allocator, image, phrase));
}

fn queryArgs(allocator: std.mem.Allocator, image: sam3.RgbImage, phrase: []const u8) std.meta.ArgsTuple(@TypeOf(computeQuery)) {
    return .{
        allocator,
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
