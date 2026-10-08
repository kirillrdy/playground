const std = @import("std");
const ast = @import("ast.zig");
const types = @import("types.zig");

// A query owns its input, source, cancellation token, and results independently
// of the selected tab and the video currently displayed by the app.
pub const QueryTab = struct {
    allocator: std.mem.Allocator,
    sql: []u8,
    draft: []u8,
    has_run: bool = true,
    query_path: ?[:0]u8,
    query_thread: ?std.Thread = null,
    query_active: std.atomic.Value(bool) = .init(true),
    query_cancel: std.atomic.Value(bool) = .init(false),
    worker_finished: std.atomic.Value(bool) = .init(false),
    table_mode: bool = false,
    table_columns: std.ArrayList([]const u8) = .empty,
    table_rows: std.ArrayList([]const []const u8) = .empty,
    table_scroll: usize = 0,
    table_column_scroll: usize = 0,
    query_matches: std.ArrayList(u32) = .empty,
    query_match_pts: std.ArrayList(f64) = .empty,
    query_match_idx: usize = 0,
    query_prompts: ast.Prompts = .{},
    precache_active: std.atomic.Value(bool) = .init(false),
    precache_phrase: [256]u8 = undefined,
    precache_phrase_len: usize = 0,
    precache_scanned_until: f64 = -1,
    precache_progress: f64 = 0,
    frames_processed: usize = 0,
    status: [256]u8 = undefined,
    status_len: usize = 0,

    pub fn create(allocator: std.mem.Allocator, sql: []const u8, source: []const u8) !*QueryTab {
        const tab = try allocator.create(QueryTab);
        errdefer allocator.destroy(tab);
        const owned_sql = try allocator.dupe(u8, sql);
        errdefer allocator.free(owned_sql);
        const draft = try allocator.dupe(u8, sql);
        errdefer allocator.free(draft);
        const path = try allocator.dupeSentinel(u8, source, 0);
        tab.* = .{ .allocator = allocator, .sql = owned_sql, .draft = draft, .query_path = path };
        return tab;
    }

    pub fn createDraft(allocator: std.mem.Allocator, source: []const u8) !*QueryTab {
        const tab = try create(allocator, "", source);
        tab.has_run = false;
        tab.query_active.store(false, .release);
        const status = "Enter SQL and run this tab's query.";
        @memcpy(tab.status[0..status.len], status);
        tab.status_len = status.len;
        return tab;
    }

    pub fn setDraft(self: *QueryTab, text: []const u8) !void {
        if (std.mem.eql(u8, self.draft, text)) return;
        const draft = try self.allocator.dupe(u8, text);
        self.allocator.free(self.draft);
        self.draft = draft;
    }

    pub fn configureResults(self: *QueryTab, sql: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var parser = @import("parser.zig").Parser.init(arena.allocator(), sql);
        const stmt = try parser.parse();
        if (stmt != .select_stmt) return;
        self.table_mode = true;
        for (stmt.select_stmt.projections, 0..) |projection, i| {
            const expr = projection.expr.*;
            if ((expr == .column_ref and std.ascii.eqlIgnoreCase(expr.column_ref, "frame")) or
                (expr == .call and std.ascii.eqlIgnoreCase(expr.call.name, "frame"))) self.table_mode = false;
            const name = if (projection.alias) |alias| alias else switch (expr) {
                .column_ref => |name| name,
                .call => |call| call.name,
                else => try std.fmt.allocPrint(arena.allocator(), "col_{d}", .{i}),
            };
            const owned = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(owned);
            try self.table_columns.append(self.allocator, owned);
        }
    }

    pub fn appendRow(self: *QueryTab, row: *const types.Row) !void {
        const cells = try self.allocator.alloc([]const u8, row.values.len);
        var count: usize = 0;
        errdefer {
            for (cells[0..count]) |cell| self.allocator.free(cell);
            self.allocator.free(cells);
        }
        for (row.values, 0..) |value, i| {
            cells[i] = try switch (value) {
                .null_type => self.allocator.dupe(u8, "NULL"),
                .bool_type => |v| self.allocator.dupe(u8, if (v) "true" else "false"),
                .int_type => |v| std.fmt.allocPrint(self.allocator, "{d}", .{v}),
                .float_type => |v| std.fmt.allocPrint(self.allocator, "{d:.6}", .{v}),
                .string_type => |v| self.allocator.dupe(u8, v),
                .mask_type => |v| std.fmt.allocPrint(self.allocator, "{d:.6}", .{v.score}),
                .detections_type => |v| std.fmt.allocPrint(self.allocator, "{d} detections", .{v.len}),
                .frame_type => |v| std.fmt.allocPrint(self.allocator, "Frame #{d}", .{v.index}),
            };
            count += 1;
        }
        try self.table_rows.append(self.allocator, cells);
    }

    pub fn clearRows(self: *QueryTab) void {
        for (self.table_rows.items) |cells| {
            for (cells) |cell| self.allocator.free(cell);
            self.allocator.free(cells);
        }
        self.table_rows.clearRetainingCapacity();
        self.table_scroll = 0;
    }

    pub fn resultCount(self: *const QueryTab) usize {
        return if (self.table_mode) self.table_rows.items.len else self.query_matches.items.len;
    }

    pub fn appendMatch(self: *QueryTab, frame: u32, pts: f64) !void {
        try self.query_matches.ensureUnusedCapacity(self.allocator, 1);
        try self.query_match_pts.ensureUnusedCapacity(self.allocator, 1);
        self.query_matches.appendAssumeCapacity(frame);
        self.query_match_pts.appendAssumeCapacity(pts);
    }

    pub fn matchTime(self: *const QueryTab, index: usize) f64 {
        return self.query_match_pts.items[index];
    }

    pub fn deinit(self: *QueryTab) void {
        self.query_cancel.store(true, .release);
        if (self.query_thread) |thread| thread.join();
        self.allocator.free(self.sql);
        self.allocator.free(self.draft);
        self.allocator.free(self.query_path.?);
        self.clearRows();
        self.table_rows.deinit(self.allocator);
        for (self.table_columns.items) |name| self.allocator.free(name);
        self.table_columns.deinit(self.allocator);
        self.query_matches.deinit(self.allocator);
        self.query_match_pts.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn matchesVideo(self: *const QueryTab, path: ?[]const u8) bool {
        return std.mem.eql(u8, self.query_path.?, path orelse return false);
    }
};

/// Preserve the selected query when an earlier tab moves, or select a neighbor.
pub fn selectionAfterClose(selected: ?usize, closed: usize, remaining: usize) ?usize {
    if (remaining == 0) return null;
    const index = selected orelse return null;
    if (index > closed) return index - 1;
    if (index == closed) return @min(closed, remaining - 1);
    return index;
}

/// Called by the UI thread. Finished workers no longer access the app or tab.
pub fn reapClosed(tabs: *std.ArrayList(*QueryTab)) void {
    var i: usize = 0;
    while (i < tabs.items.len) {
        const tab = tabs.items[i];
        if (tab.query_thread == null or tab.worker_finished.load(.acquire)) {
            _ = tabs.orderedRemove(i);
            tab.deinit();
        } else {
            i += 1;
        }
    }
}

test "closing query tabs preserves selection or selects the nearest remaining tab" {
    try std.testing.expectEqual(@as(?usize, null), selectionAfterClose(0, 0, 0));
    try std.testing.expectEqual(@as(?usize, 1), selectionAfterClose(2, 0, 2));
    try std.testing.expectEqual(@as(?usize, 0), selectionAfterClose(0, 2, 2));
    try std.testing.expectEqual(@as(?usize, 1), selectionAfterClose(1, 1, 2));
    try std.testing.expectEqual(@as(?usize, 1), selectionAfterClose(2, 2, 2));
}

test "closed tabs retain active workers until they finish" {
    const allocator = std.testing.allocator;
    var tabs: std.ArrayList(*QueryTab) = .empty;
    defer tabs.deinit(allocator);
    const tab = try QueryTab.create(allocator, "SELECT frame", "video.mp4");
    tab.query_cancel.store(true, .release);
    tab.query_thread = try std.Thread.spawn(.{}, struct {
        fn run(query: *QueryTab) void {
            while (!query.worker_finished.load(.acquire)) std.atomic.spinLoopHint();
        }
    }.run, .{tab});
    try tabs.append(allocator, tab);
    reapClosed(&tabs);
    try std.testing.expectEqual(@as(usize, 1), tabs.items.len);
    tab.worker_finished.store(true, .release);
    reapClosed(&tabs);
    try std.testing.expectEqual(@as(usize, 0), tabs.items.len);
}

test "query tabs keep owned sources and independent results and cancellation" {
    const allocator = std.testing.allocator;
    const source = try allocator.dupe(u8, "first.mp4");
    const sql = try allocator.dupe(u8, "SELECT frame FROM 'first.mp4'");
    const first = try QueryTab.create(allocator, sql, source);
    defer first.deinit();
    allocator.free(source);
    allocator.free(sql);
    const second = try QueryTab.create(allocator, "SELECT frame FROM 'second.mp4'", "second.mp4");
    defer second.deinit();

    try first.query_matches.append(allocator, 42);
    first.query_cancel.store(true, .release);
    try std.testing.expect(first.matchesVideo("first.mp4"));
    try std.testing.expect(!first.matchesVideo("second.mp4"));
    try std.testing.expect(!first.matchesVideo(null));
    try std.testing.expectEqualStrings("SELECT frame FROM 'first.mp4'", first.sql);
    try std.testing.expectEqual(@as(usize, 0), second.query_matches.items.len);
    try std.testing.expect(!second.query_cancel.load(.acquire));
}

test "query editors retain independent drafts without changing a running SQL snapshot" {
    const allocator = std.testing.allocator;
    const first = try QueryTab.create(allocator, "SELECT frame FROM 'first.mp4'", "first.mp4");
    defer first.deinit();
    const second = try QueryTab.createDraft(allocator, "");
    defer second.deinit();
    try first.setDraft("SELECT frame FROM 'first.mp4' WHERE frame_id BETWEEN 10 AND 20");
    try second.setDraft("SELECT frame FROM 'second.mp4' LIMIT 5");
    try std.testing.expectEqualStrings("SELECT frame FROM 'first.mp4'", first.sql);
    try std.testing.expectEqualStrings("SELECT frame FROM 'second.mp4' LIMIT 5", second.draft);
    try std.testing.expect(!second.has_run);
    try std.testing.expect(!second.query_active.load(.acquire));
    try std.testing.expect(first.query_active.load(.acquire));
}

test "query results retain their decoded timestamps for tab navigation" {
    const tab = try QueryTab.create(std.testing.allocator, "SELECT frame", "video.mp4");
    defer tab.deinit();
    try tab.appendMatch(100, 4.0);
    try tab.appendMatch(125, 5.0);
    tab.query_match_idx = 1;
    try std.testing.expectEqual(@as(u32, 125), tab.query_matches.items[tab.query_match_idx]);
    try std.testing.expectEqual(@as(f64, 5.0), tab.matchTime(tab.query_match_idx));
}

test "scalar projections choose a table and retain owned cells and aliases" {
    const tab = try QueryTab.create(std.testing.allocator, "", "foo.mp4");
    defer tab.deinit();
    try tab.configureResults("select timestamp AS time, 'hat' AS label from 'foo.mp4' where sam3(frame, \"hat\") > 0.9;");
    try std.testing.expect(tab.table_mode);
    try std.testing.expectEqualStrings("time", tab.table_columns.items[0]);
    var text = [_]u8{ 'h', 'a', 't' };
    const values = [_]types.Value{ .{ .float_type = 1.25 }, .{ .string_type = &text } };
    try tab.appendRow(&.{ .values = &values });
    text[0] = 'c';
    try std.testing.expectEqualStrings("1.250000", tab.table_rows.items[0][0]);
    try std.testing.expectEqualStrings("hat", tab.table_rows.items[0][1]);
    try std.testing.expectEqual(@as(usize, 1), tab.resultCount());
    tab.clearRows();
    try std.testing.expectEqual(@as(usize, 0), tab.resultCount());
}

test "frame projections choose video regardless of alias and predicate" {
    for ([_][]const u8{
        "SELECT FRAME AS picture, timestamp FROM 'foo.mp4'",
        "SELECT frame(1) AS picture FROM 'foo.mp4'",
    }) |sql| {
        const tab = try QueryTab.create(std.testing.allocator, sql, "foo.mp4");
        defer tab.deinit();
        try tab.configureResults(sql);
        try std.testing.expect(!tab.table_mode);
    }
}
