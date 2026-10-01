const std = @import("std");
const overlay = @import("overlay.zig");

// A query owns its input, source, cancellation token, and results independently
// of the selected tab and the video currently displayed by the app.
pub const QueryTab = struct {
    allocator: std.mem.Allocator,
    sql: []u8,
    query_path: ?[:0]u8,
    query_thread: ?std.Thread = null,
    query_active: std.atomic.Value(bool) = .init(true),
    query_cancel: std.atomic.Value(bool) = .init(false),
    query_matches: std.ArrayList(u32) = .empty,
    query_match_idx: usize = 0,
    query_prompts: overlay.Prompts = .{},
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
        const path = try allocator.dupeZ(u8, source);
        tab.* = .{ .allocator = allocator, .sql = owned_sql, .query_path = path };
        return tab;
    }

    pub fn deinit(self: *QueryTab) void {
        self.query_cancel.store(true, .release);
        if (self.query_thread) |thread| thread.join();
        self.allocator.free(self.sql);
        self.allocator.free(self.query_path.?);
        self.query_matches.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn matchesVideo(self: *const QueryTab, path: ?[]const u8) bool {
        return std.mem.eql(u8, self.query_path.?, path orelse return false);
    }
};

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
