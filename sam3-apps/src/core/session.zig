const std = @import("std");
const log = @import("log");
const zigimg = @import("zigimg");
const sam3 = @import("sam3");
const vdb = @import("vdb");
const paths = @import("paths.zig");
const indexing = @import("indexing.zig");
const inference = @import("inference.zig");

pub const QueryTab = vdb.query_tab.QueryTab;

pub fn isQuerySql(text: []const u8) bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (std.ascii.startsWithIgnoreCase(trimmed, "SELECT")) return true;
    if (std.ascii.startsWithIgnoreCase(trimmed, "CREATE")) return true;
    if (std.ascii.startsWithIgnoreCase(trimmed, "WHERE")) return true;
    return false;
}

pub fn Session(comptime VideoReader: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        io: std.Io,
        mutex: std.Io.Mutex = .init,
        index_mutex: std.Io.Mutex = .init,

        query_tabs: std.ArrayList(*QueryTab) = .empty,
        selected_query: ?usize = null,
        closed_queries: std.ArrayList(*QueryTab) = .empty,

        pub fn init(allocator: std.mem.Allocator, io: std.Io) Self {
            return .{
                .allocator = allocator,
                .io = io,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.query_tabs.items) |tab| {
                tab.query_cancel.store(true, .release);
            }
            for (self.query_tabs.items) |tab| {
                if (tab.query_thread) |thread| thread.join();
                tab.query_thread = null;
            }
            for (self.query_tabs.items) |tab| {
                tab.deinit();
            }
            self.query_tabs.deinit(self.allocator);
            for (self.closed_queries.items) |tab| {
                tab.deinit();
            }
            self.closed_queries.deinit(self.allocator);
        }

        pub fn selectedTab(self: *const Self) ?*QueryTab {
            return self.query_tabs.items[self.selected_query orelse return null];
        }

        pub fn isQueryVideo(self: *const Self, current_video_path: ?[]const u8) bool {
            return (self.selectedTab() orelse return false).matchesVideo(current_video_path);
        }

        pub fn isSelectedQuery(self: *const Self, current_video_path: ?[]const u8, tab: *QueryTab) bool {
            return self.selectedTab() == tab and tab.matchesVideo(current_video_path);
        }

        pub fn queryMatches(self: *const Self, current_video_path: ?[]const u8) []const u32 {
            if (!self.isQueryVideo(current_video_path)) return &.{};
            return self.selectedTab().?.query_matches.items;
        }

        pub fn selectedQueryActive(self: *const Self) bool {
            return (self.selectedTab() orelse return false).query_active.load(.acquire);
        }

        pub fn selectedIndexing(self: *const Self) bool {
            return (self.selectedTab() orelse return false).precache_active.load(.acquire);
        }

        pub fn formatTabLabels(self: *Self, allocator: std.mem.Allocator) ![:0]u8 {
            self.mutex.lock(self.io) catch return error.LockFailed;
            defer self.mutex.unlock(self.io);
            var labels: std.ArrayList(u8) = .empty;
            defer labels.deinit(allocator);
            for (self.query_tabs.items, 0..) |tab, i| {
                const state = if (!tab.has_run) "New" else if (tab.query_active.load(.acquire))
                    if (tab.query_cancel.load(.acquire)) "Cancelling" else "Running"
                else if (tab.query_cancel.load(.acquire)) "Cancelled" else "Done";
                const label = try std.fmt.allocPrint(allocator, "Query {d}: {s} ({d})", .{ i + 1, state, tab.resultCount() });
                defer allocator.free(label);
                if (i > 0) try labels.append(allocator, '\n');
                try labels.appendSlice(allocator, label);
            }
            return allocator.dupeSentinel(u8, labels.items, 0);
        }

        pub fn formatTable(self: *Self, allocator: std.mem.Allocator) ![:0]u8 {
            self.mutex.lock(self.io) catch return error.LockFailed;
            defer self.mutex.unlock(self.io);
            const tab = self.selectedTab();
            const json = if (tab != null and tab.?.table_mode)
                try std.json.Stringify.valueAlloc(allocator, .{ .columns = tab.?.table_columns.items, .rows = tab.?.table_rows.items }, .{})
            else
                try allocator.dupe(u8, "null");
            defer allocator.free(json);
            return allocator.dupeSentinel(u8, json, 0);
        }

        pub fn finishQueryStatus(self: *Self, host: anytype, tab: *QueryTab) void {
            self.mutex.lock(self.io) catch return;
            const cancelling = std.mem.startsWith(u8, tab.status[0..tab.status_len], "Cancelling");
            self.mutex.unlock(self.io);
            if (cancelling) self.setQueryStatus(host, tab, "Query cancelled.");
        }

        pub fn setQueryStatus(self: *Self, host: anytype, tab: *QueryTab, status: []const u8) void {
            self.mutex.lock(self.io) catch return;
            tab.status_len = @min(status.len, tab.status.len);
            @memcpy(tab.status[0..tab.status_len], status[0..tab.status_len]);
            const is_selected = self.selectedTab() == tab;
            self.mutex.unlock(self.io);
            if (is_selected) {
                host.setStatus(status);
            }
            host.refreshQueryTabs();
        }

        pub fn editQuery(self: *Self, text: []const u8) void {
            self.mutex.lock(self.io) catch return;
            defer self.mutex.unlock(self.io);
            if (self.selectedTab()) |tab| tab.setDraft(text) catch {};
        }

        pub fn newQuery(self: *Self, host: anytype) void {
            if (host.isBusy()) return;
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
            self.selectQuery(host, index);
        }

        pub fn selectQuery(self: *Self, host: anytype, index: usize) void {
            if (host.isBusy() or index >= self.query_tabs.items.len) return;
            self.mutex.lock(self.io) catch return;
            self.selected_query = index;
            const tab = self.query_tabs.items[index];
            host.clearOverlay();
            self.mutex.unlock(self.io);

            if (tab.query_path.?.len > 0) {
                _ = host.openVideoFromPath(tab.query_path.?);
            } else {
                host.stopVideo();
            }

            host.setQueryText(tab.draft);

            self.mutex.lock(self.io) catch return;
            const status = self.allocator.dupe(u8, tab.status[0..tab.status_len]) catch {
                self.mutex.unlock(self.io);
                return;
            };
            self.mutex.unlock(self.io);
            defer self.allocator.free(status);
            host.setStatus(status);

            self.mutex.lock(self.io) catch return;
            if (tab.query_matches.items.len > 0) {
                tab.query_match_idx = @min(tab.query_match_idx, tab.query_matches.items.len - 1);
                host.seekVideo(tab.matchTime(tab.query_match_idx));
            }
            self.mutex.unlock(self.io);
            host.refreshQueryTabs();
        }

        pub fn closeQuery(self: *Self, host: anytype, index: usize) void {
            if (host.isBusy() or index >= self.query_tabs.items.len) return;
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
                host.clearOverlay();
            }
            self.mutex.unlock(self.io);

            if (was_selected) {
                if (next) |selected| {
                    self.selectQuery(host, selected);
                } else {
                    host.setQueryText("");
                    host.setStatus("Query closed.");
                }
            }
            if (self.query_tabs.items.len == 0) self.newQuery(host);
            host.refreshQueryTabs();
            vdb.query_tab.reapClosed(&self.closed_queries);
        }

        pub fn reapClosed(self: *Self) void {
            vdb.query_tab.reapClosed(&self.closed_queries);
        }

        pub fn restoreQueryOverlay(self: *Self, host: anytype) void {
            if (!self.isQueryVideo(host.getVideoPath())) return;
            const tab = self.selectedTab().?;
            if (tab.query_prompts.count > 0) {
                host.setOverlayPrompts(tab.query_prompts);
            } else if (tab.precache_phrase_len > 0) {
                host.setOverlayPhrase(tab.precache_phrase[0..tab.precache_phrase_len]);
            }
        }

        pub fn handleCancelQuery(self: *Self, host: anytype) void {
            const tab = self.selectedTab() orelse return;
            if (tab.query_active.load(.acquire)) {
                tab.query_cancel.store(true, .release);
                self.setQueryStatus(host, tab, "Cancelling query…");
            }
        }

        pub fn handleClearQuery(self: *Self, host: anytype) void {
            self.handleCancelQuery(host);
            self.mutex.lock(self.io) catch return;
            if (self.selectedTab()) |tab| {
                tab.query_prompts = .{};
                tab.precache_phrase_len = 0;
                tab.clearRows();
                tab.query_matches.clearRetainingCapacity();
                tab.query_match_pts.clearRetainingCapacity();
                tab.query_match_idx = 0;
            }
            host.clearOverlay();
            self.mutex.unlock(self.io);
            host.setStatus("Query cleared.");
            host.refreshQueryTabs();
        }

        pub fn handleQuery(self: *Self, host: anytype, raw_query: []const u8) void {
            if (host.isBusy()) return;
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
                if (self.selectedTab()) |tab| self.setQueryStatus(host, tab, "Include FROM 'video.mp4' in this tab's query.");
                return;
            }

            // If a video path is specified in the query, resolve and open it
            if (parsed_source_path) |source_file| {
                if (paths.resolveVideoPath(self.allocator, source_file)) |resolved| {
                    defer self.allocator.free(resolved);
                    const current_path = host.getVideoPath();
                    const is_same_video = if (host.isVideoActive() and current_path != null)
                        std.mem.eql(u8, current_path.?, resolved)
                    else
                        false;

                    if (!is_same_video) {
                        _ = host.openVideoFromPath(resolved);
                    }
                } else {
                    var err_buf: [256]u8 = undefined;
                    const err_msg = std.fmt.bufPrint(&err_buf, "Could not find video file: “{s}”", .{source_file}) catch "Video not found.";
                    if (self.selectedTab()) |current| self.setQueryStatus(host, current, err_msg);
                    return;
                }
            }

            if (!host.isVideoActive() or host.getVideoPath() == null) {
                if (self.selectedTab()) |current| self.setQueryStatus(host, current, "Could not load the video specified in FROM.");
                return;
            }
            const tab = QueryTab.create(self.allocator, trimmed, host.getVideoPath().?) catch return;
            tab.configureResults(trimmed) catch {
                tab.deinit();
                return;
            };
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
            host.clearOverlay();
            self.mutex.unlock(self.io);
            self.setQueryStatus(host, tab, "Planning query in the background…");
            host.refreshQueryTabs();

            const Worker = struct {
                fn run(sess: *Self, h: @TypeOf(host), qtab: *QueryTab) void {
                    sess.runQueryWorker(h, qtab);
                }
            };

            tab.query_thread = std.Thread.spawn(.{}, Worker.run, .{ self, host, tab }) catch {
                tab.query_active.store(false, .release);
                self.setQueryStatus(host, tab, "Could not start query worker.");
                host.refreshQueryTabs();
                return;
            };
        }

        fn saveVideoIndex(self: *Self, tab: *QueryTab, builder: *vdb.index.IndexBuilder) !void {
            return indexing.saveVideoIndex(self.allocator, self.io, &self.index_mutex, tab.query_path.?, builder);
        }

        fn runPrecache(self: *Self, host: anytype, tab: *QueryTab, source_path: [:0]const u8, phrase: []const u8) void {
            self.mutex.lock(self.io) catch return;
            if (tab.query_cancel.load(.acquire)) {
                self.mutex.unlock(self.io);
                return;
            }
            tab.precache_phrase_len = @min(phrase.len, tab.precache_phrase.len);
            @memcpy(tab.precache_phrase[0..tab.precache_phrase_len], phrase[0..tab.precache_phrase_len]);
            if (self.isSelectedQuery(host.getVideoPath(), tab)) {
                host.setOverlayPhrase(tab.precache_phrase[0..tab.precache_phrase_len]);
            }
            tab.precache_scanned_until = -1;
            tab.precache_active.store(true, .release);
            self.mutex.unlock(self.io);
            defer tab.precache_active.store(false, .release);
            host.refreshQueryTabs();
            defer host.refreshQueryTabs();

            var video_adapter = VideoReader.init(self.allocator, source_path);
            defer video_adapter.deinit();
            if (!video_adapter.isValid()) {
                self.setQueryStatus(host, tab, "Could not decode video for pre-caching.");
                return;
            }
            const duration = video_adapter.duration;
            var reader = video_adapter.asReader();
            var frames: usize = 0;
            const started = std.Io.Timestamp.now(self.io, .awake);
            var last_ui_update = started;
            var last_log = started;
            var fraction: f64 = 0;

            var index_builder = vdb.index.IndexBuilder.init(self.allocator);
            defer index_builder.deinit();

            log.info(self.io, "pre-caching and indexing video for \"{s}\" ({d:.2} s)", .{ phrase, duration });
            while (!tab.query_cancel.load(.acquire)) {
                const frame_opt = reader.nextFrame() catch {
                    self.setQueryStatus(host, tab, "Video decoding failed during pre-cache.");
                    return;
                };
                const frame = frame_opt orelse {
                    self.saveVideoIndex(tab, &index_builder) catch |err| {
                        log.info(self.io, "Could not save video index: {t}", .{err});
                        self.setQueryStatus(host, tab, "Could not save video index.");
                        return;
                    };
                    self.mutex.lock(self.io) catch return;
                    tab.precache_progress = 1;
                    tab.frames_processed = frames;
                    self.mutex.unlock(self.io);
                    host.refreshQueryTabs();
                    var status_buf: [200]u8 = undefined;
                    const status = std.fmt.bufPrint(&status_buf, "Pre-cached and indexed {d} frames for “{s}”. Saved to .vdb sidecar.", .{ frames, phrase }) catch "Video pre-cache complete.";
                    self.setQueryStatus(host, tab, status);
                    log.info(self.io, "pre-cached and indexed {d} frames for \"{s}\" in {f}", .{ frames, phrase, started.untilNow(self.io, .awake) });
                    return;
                };

                if (frame.width <= 0 or frame.height <= 0 or frame.rgb == null) continue;
                const width = frame.width;
                const height = frame.height;
                const rgb_len = std.math.mul(usize, std.math.mul(usize, width, height) catch continue, 3) catch continue;
                const pixels = self.allocator.dupe(u8, frame.rgb.?[0..rgb_len]) catch {
                    self.setQueryStatus(host, tab, "Out of memory during video pre-cache.");
                    return;
                };
                var decoded = zigimg.Image.fromRawPixelsOwned(width, height, pixels, .rgb24) catch {
                    self.allocator.free(pixels);
                    self.setQueryStatus(host, tab, "Could not prepare video frame for pre-cache.");
                    return;
                };
                defer decoded.deinit(self.allocator);

                const frame_started = std.Io.Timestamp.now(self.io, .awake);
                host.model_mutex.lock(self.io) catch return;
                if (tab.query_cancel.load(.acquire)) {
                    host.model_mutex.unlock(self.io);
                    break;
                }
                const result = inference.cachedQuery(self.allocator, sam3.RgbImage.fromImage(decoded), phrase);
                host.model_mutex.unlock(self.io);
                const values = result catch |err| {
                    log.info(self.io, "Video pre-cache failed at frame {d}: {t}: {s}", .{ frames, err, sam3.onnx.lastError() });
                    self.setQueryStatus(host, tab, "Video pre-cache inference failed.");
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
                    host.refreshQueryTabs();
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
                self.setQueryStatus(host, tab, "Video pre-cache cancelled.");
                log.info(self.io, "pre-cache cancelled after {d} frames", .{frames});
            }
        }

        fn runQueryWorker(self: *Self, host: anytype, tab: *QueryTab) void {
            defer {
                tab.query_active.store(false, .release);
                self.finishQueryStatus(host, tab);
                host.refreshQueryTabs();
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
                        const status_msg = std.fmt.bufPrint(&status_buf, "Creating visual index for “{s}”…", .{target_prompt}) catch "Creating visual index…";
                        self.setQueryStatus(host, tab, status_msg);
                        self.runPrecache(host, tab, source_path, target_prompt);
                        return;
                    }
                } else |err| {
                    log.info(self.io, "Failed to parse CREATE INDEX: {t}", .{err});
                    var err_buf: [160]u8 = undefined;
                    const err_msg = std.fmt.bufPrint(&err_buf, "CREATE INDEX syntax error: {t}", .{err}) catch "Syntax error.";
                    self.setQueryStatus(host, tab, err_msg);
                    return;
                }
            }

            const final_query = vdb.parser.normalize(self.allocator, trimmed, source_path) catch |err| {
                log.info(self.io, "Query normalization failed: {t}", .{err});
                var err_buf: [160]u8 = undefined;
                const err_msg = std.fmt.bufPrint(&err_buf, "Query syntax error: {t}", .{err}) catch "Query syntax error.";
                self.setQueryStatus(host, tab, err_msg);
                return;
            };
            defer self.allocator.free(final_query);

            log.info(self.io, "Executing visual query: {s}", .{final_query});

            const overlay_prompts = vdb.ast.Prompts.fromSql(self.allocator, final_query) catch |err| {
                log.info(self.io, "Query overlay planning failed: {t}", .{err});
                self.setQueryStatus(host, tab, "Could not parse query overlay prompts.");
                return;
            };
            self.mutex.lock(self.io) catch return;
            if (tab.query_cancel.load(.acquire)) {
                self.mutex.unlock(self.io);
                return;
            }
            tab.query_prompts = overlay_prompts;
            if (self.isSelectedQuery(host.getVideoPath(), tab)) {
                host.setOverlayPrompts(overlay_prompts);
            }
            self.mutex.unlock(self.io);

            const HostType = @TypeOf(host);
            const QueryStreamer = struct {
                session: *Self,
                host: HostType,
                tab: *QueryTab,
                first_match_emitted: bool = false,
                last_ui_update: std.Io.Timestamp,

                pub fn onRow(streamer_ctx: *anyopaque, row: *const vdb.types.Row) anyerror!void {
                    const streamer: *@This() = @ptrCast(@alignCast(streamer_ctx));
                    const sess = streamer.session;
                    const h = streamer.host;

                    var match_idx: ?usize = null;
                    var match_pts: f64 = 0;
                    for (row.values) |v| {
                        if (v == .frame_type) {
                            match_idx = v.frame_type.index;
                            match_pts = v.frame_type.pts_seconds;
                            break;
                        }
                    }
                    if (streamer.tab.table_mode) {
                        sess.mutex.lock(sess.io) catch return;
                        if (streamer.tab.query_cancel.load(.acquire)) {
                            sess.mutex.unlock(sess.io);
                            return error.QueryCancelled;
                        }
                        streamer.tab.appendRow(row) catch |err| {
                            sess.mutex.unlock(sess.io);
                            return err;
                        };
                        const total = streamer.tab.resultCount();
                        sess.mutex.unlock(sess.io);
                        if (total == 1 or streamer.last_ui_update.untilNow(sess.io, .awake).nanoseconds > 200_000_000) {
                            streamer.last_ui_update = std.Io.Timestamp.now(sess.io, .awake);
                            var buffer: [160]u8 = undefined;
                            sess.setQueryStatus(h, streamer.tab, std.fmt.bufPrint(&buffer, "Streaming query: {d} rows…", .{total}) catch "Streaming query…");
                        }
                        return;
                    }
                    const idx = match_idx orelse return;

                    sess.mutex.lock(sess.io) catch return;
                    if (streamer.tab.query_cancel.load(.acquire)) {
                        sess.mutex.unlock(sess.io);
                        return error.QueryCancelled;
                    }
                    streamer.tab.appendMatch(@intCast(idx), match_pts) catch |err| {
                        sess.mutex.unlock(sess.io);
                        return err;
                    };
                    const total = streamer.tab.query_matches.items.len;

                    if (!streamer.first_match_emitted) {
                        streamer.first_match_emitted = true;
                        if (sess.isSelectedQuery(h.getVideoPath(), streamer.tab)) {
                            streamer.tab.query_match_idx = 0;
                            h.seekVideo(match_pts);
                        }
                        sess.mutex.unlock(sess.io);

                        var status_buf: [256]u8 = undefined;
                        const status_msg = std.fmt.bufPrint(&status_buf, "First match found instantly! Frame #{d} at {d:.2}s. Streaming visual query results…", .{
                            idx,
                            match_pts,
                        }) catch "First frame matched!";
                        sess.setQueryStatus(h, streamer.tab, status_msg);
                        log.info(sess.io, "Instantly streamed first match: Frame #{d} at {d:.2}s", .{ idx, match_pts });
                    } else {
                        sess.mutex.unlock(sess.io);

                        const now = std.Io.Timestamp.now(sess.io, .awake);
                        if (streamer.last_ui_update.untilNow(sess.io, .awake).nanoseconds > 200_000_000 or total % 50 == 0) {
                            streamer.last_ui_update = now;
                            var status_buf: [160]u8 = undefined;
                            const status_msg = std.fmt.bufPrint(&status_buf, "Streaming query: {d} matches found… scanning… Press Cancel Query to stop.", .{total}) catch "Streaming query…";
                            sess.setQueryStatus(h, streamer.tab, status_msg);
                        }
                    }
                }
            };

            var streamer = QueryStreamer{
                .session = self,
                .host = host,
                .tab = tab,
                .last_ui_update = started,
            };
            const stream_cb: vdb.engine.RowCallback = .{
                .ctx = &streamer,
                .onRow = QueryStreamer.onRow,
            };

            var video_adapter = VideoReader.init(self.allocator, source_path);
            defer video_adapter.deinit();

            const BridgeCtx = struct {
                session: *Self,
                host: HostType,
            };
            var bridge = BridgeCtx{ .session = self, .host = host };

            const SamSegmentor = struct {
                fn bridgeFn(ctx: *anyopaque, allocator: std.mem.Allocator, frame: vdb.types.FrameRef, prompt: []const u8) anyerror!vdb.types.MaskRef {
                    _ = allocator;
                    const b: *BridgeCtx = @ptrCast(@alignCast(ctx));
                    return inference.evaluateSam3Frame(b.session.allocator, b.session.io, &b.host.model_mutex, frame, prompt);
                }
            };

            database.engine_inst.sam3 = .{
                .ptr = &bridge,
                .segmentFn = SamSegmentor.bridgeFn,
            };

            var result = database.executeQuery(final_query, video_adapter.asReader(), &tab.query_cancel, stream_cb) catch |err| {
                if (err == error.QueryCancelled) {
                    log.info(self.io, "Visual database query cancelled.", .{});
                    self.mutex.lock(self.io) catch return;
                    const matches_so_far = tab.resultCount();
                    self.mutex.unlock(self.io);
                    var cancel_buf: [160]u8 = undefined;
                    const cancel_msg = if (matches_so_far > 0)
                        std.fmt.bufPrint(&cancel_buf, "Query cancelled. Kept {d} matches found so far.", .{matches_so_far}) catch "Query cancelled."
                    else
                        "Visual query cancelled.";
                    self.setQueryStatus(host, tab, cancel_msg);
                    return;
                }
                log.info(self.io, "Query execution error: {t}", .{err});
                var err_buf: [160]u8 = undefined;
                const err_msg = std.fmt.bufPrint(&err_buf, "Query error: {t}", .{err}) catch "Query execution failed.";
                self.setQueryStatus(host, tab, err_msg);
                return;
            };
            defer result.deinit();

            const elapsed = started.untilNow(self.io, .awake);

            self.mutex.lock(self.io) catch return;
            const total_matches = tab.query_matches.items.len;
            const first_frame = if (total_matches > 0) tab.query_matches.items[0] else 0;
            const first_frame_target = if (total_matches > 0) tab.matchTime(0) else null;
            self.mutex.unlock(self.io);

            if (tab.table_mode) {
                var status_buf: [160]u8 = undefined;
                const status_msg = std.fmt.bufPrint(&status_buf, "Query complete: {d} row(s) in {f}.", .{ result.rows.len, elapsed }) catch "Query completed.";
                self.setQueryStatus(host, tab, status_msg);
            } else if (total_matches > 0) {
                var status_buf: [256]u8 = undefined;
                const first_sec = first_frame_target.?;
                const status_msg = std.fmt.bufPrint(&status_buf, "Query complete: {d} match(es) in {f}. First match: Frame #{d} at {d:.2}s.", .{
                    total_matches,
                    elapsed,
                    first_frame,
                    first_sec,
                }) catch "Query completed.";
                self.setQueryStatus(host, tab, status_msg);
            } else {
                var status_buf: [160]u8 = undefined;
                const status_msg = std.fmt.bufPrint(&status_buf, "Query returned 0 matching frames in {f}.", .{elapsed}) catch "0 matches.";
                self.setQueryStatus(host, tab, status_msg);
            }
        }
    };
}

test "isQuerySql recognizes statements" {
    try std.testing.expect(isQuerySql("SELECT frame FROM 'video.mp4'"));
    try std.testing.expect(isQuerySql("select frame"));
    try std.testing.expect(isQuerySql("CREATE INDEX on sam3('dog')"));
    try std.testing.expect(isQuerySql("create index"));
    try std.testing.expect(isQuerySql("WHERE object_score > 0.5"));
    try std.testing.expect(!isQuerySql("just a regular text phrase"));
}
