const std = @import("std");

pub const types = @import("types.zig");
pub const ast = @import("ast.zig");
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const index = @import("index.zig");
pub const planner = @import("planner.zig");
pub const engine = @import("engine.zig");

pub const Database = struct {
    allocator: std.mem.Allocator,
    indexes: std.StringHashMapUnmanaged(index.InvertedIndex) = .{},
    engine_inst: engine.Engine,

    pub fn init(allocator: std.mem.Allocator) Database {
        return .{
            .allocator = allocator,
            .engine_inst = engine.Engine.init(allocator),
        };
    }

    pub fn deinit(self: *Database) void {
        var it = self.indexes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.indexes.deinit(self.allocator);
    }

    pub fn registerIndex(self: *Database, video_path: []const u8, idx: index.InvertedIndex) !void {
        const key = try self.allocator.dupe(u8, video_path);
        try self.indexes.put(self.allocator, key, idx);
    }

    pub fn getIndex(self: *const Database, video_path: []const u8) ?*const index.InvertedIndex {
        return self.indexes.getPtr(video_path);
    }

    pub fn executeQuery(
        self: *Database,
        sql: []const u8,
        reader: engine.VideoReader,
        cancel_token: ?*const std.atomic.Value(bool),
        row_callback: ?engine.RowCallback,
    ) !engine.QueryResult {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        var p = parser.Parser.init(a, sql);
        const stmt = try p.parse();

        switch (stmt) {
            .select_stmt => |sel| {
                const source_path = switch (sel.source) {
                    .file_path => |f| f,
                    .call => |c| c.path,
                };
                const idx = self.getIndex(source_path);

                var query_planner = planner.Planner.init(a);
                var plan = try query_planner.createPlan(sel, idx);
                defer plan.deinit();

                return try self.engine_inst.execute(&plan, reader, idx, cancel_token, row_callback);
            },
            .create_index => {
                return error.UseIndexBuilderDirectly;
            },
        }
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "vdb: lexer and parser test" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const ally = arena.allocator();

    const query =
        \\SELECT frame, sam3(frame, "cat") AS cat_mask
        \\FROM "video.mp4"
        \\WHERE yolov8(frame) CONTAINS "cat"
        \\LIMIT 5;
    ;

    var p = parser.Parser.init(ally, query);
    const stmt = try p.parse();

    try std.testing.expectEqual(@as(usize, 2), stmt.select_stmt.projections.len);
    try std.testing.expectEqualStrings("video.mp4", stmt.select_stmt.source.file_path);
    try std.testing.expect(stmt.select_stmt.where_clause != null);
    try std.testing.expectEqual(@as(?usize, 5), stmt.select_stmt.limit);
}

test "vdb: inverted index serialization and lookup" {
    const ally = std.testing.allocator;

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    try builder.addDetection(10, 333, "dog", 0.92, .{ .x = 0.1, .y = 0.2, .w = 0.3, .h = 0.4 });
    try builder.addDetection(25, 833, "dog", 0.85, .{ .x = 0.12, .y = 0.22, .w = 0.3, .h = 0.4 });
    try builder.addDetection(15, 500, "cat", 0.98, .{ .x = 0.5, .y = 0.5, .w = 0.2, .h = 0.2 });

    var idx = try builder.build();
    defer idx.deinit();

    // Serialize to byte slice
    const bytes = try idx.serialize(ally);
    defer ally.free(bytes);

    // Deserialize from byte slice
    var loaded_idx = try index.InvertedIndex.deserialize(ally, bytes);
    defer loaded_idx.deinit();

    // Verify postings
    const dog_postings = loaded_idx.lookup("dog").?;
    try std.testing.expectEqual(@as(usize, 2), dog_postings.len);
    try std.testing.expectEqual(@as(u32, 10), dog_postings[0].frame_idx);
    try std.testing.expectEqual(@as(u32, 25), dog_postings[1].frame_idx);

    const cat_postings = loaded_idx.lookup("cat").?;
    try std.testing.expectEqual(@as(usize, 1), cat_postings.len);
    try std.testing.expectEqual(@as(u32, 15), cat_postings[0].frame_idx);
}

test "vdb: query planner pushdown" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const ally = arena.allocator();

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    try builder.addDetection(42, 1400, "dog", 0.89, .{ .x = 0, .y = 0, .w = 1, .h = 1 });
    try builder.addDetection(99, 3300, "dog", 0.91, .{ .x = 0, .y = 0, .w = 1, .h = 1 });

    var idx = try builder.build();
    defer idx.deinit();

    var p = parser.Parser.init(ally, "SELECT frame FROM \"video.mp4\" WHERE yolov8(frame) CONTAINS 'dog'");
    const stmt = try p.parse();

    var q_planner = planner.Planner.init(ally);
    var plan = try q_planner.createPlan(stmt.select_stmt, &idx);
    defer plan.deinit();

    // Verify index pushdown succeeded!
    switch (plan.strategy) {
        .indexed_seek => |seek| {
            try std.testing.expectEqual(@as(usize, 2), seek.candidate_frames.len);
            try std.testing.expectEqual(@as(u32, 42), seek.candidate_frames[0]);
            try std.testing.expectEqual(@as(u32, 99), seek.candidate_frames[1]);
        },
        .full_scan => return error.TestUnexpectedResult,
    }
}

// Mock Video Reader for testing
const MockReader = struct {
    frames_total: usize = 100,
    current_frame: usize = 0,

    pub fn reader(self: *MockReader) engine.VideoReader {
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
        const self: *MockReader = @ptrCast(@alignCast(ctx));
        return self.frames_total;
    }

    fn seekToFrameImpl(ctx: *anyopaque, frame_idx: usize) anyerror!void {
        const self: *MockReader = @ptrCast(@alignCast(ctx));
        self.current_frame = frame_idx;
    }

    fn nextFrameImpl(ctx: *anyopaque) anyerror!?types.FrameRef {
        const self: *MockReader = @ptrCast(@alignCast(ctx));
        if (self.current_frame >= self.frames_total) return null;
        const idx = self.current_frame;
        self.current_frame += 1;
        return types.FrameRef{
            .index = idx,
            .pts_seconds = @as(f64, @floatFromInt(idx)) * 0.033,
            .width = 1920,
            .height = 1080,
        };
    }
};

test "vdb: end-to-end database query execution" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    // Build index with dogs on frames 7 and 12
    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    try builder.addDetection(7, 231, "dog", 0.94, .{ .x = 0.2, .y = 0.2, .w = 0.3, .h = 0.3 });
    try builder.addDetection(12, 396, "dog", 0.88, .{ .x = 0.25, .y = 0.2, .w = 0.3, .h = 0.3 });

    try db.registerIndex("test.mp4", try builder.build());

    var mock = MockReader{ .frames_total = 20 };
    var result = try db.executeQuery(
        "SELECT frame, sam3(frame, 'dog') AS mask FROM 'test.mp4' WHERE yolov8(frame) CONTAINS 'dog'",
        mock.reader(),
        null,
        null,
    );
    defer result.deinit();

    // Verify 2 matched rows
    try std.testing.expectEqual(@as(usize, 2), result.rows.len);
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqualStrings("frame", result.columns[0].name);
    try std.testing.expectEqualStrings("mask", result.columns[1].name);

    // Frame 7 was returned
    try std.testing.expectEqual(@as(usize, 7), result.rows[0].values[0].frame_type.index);
    // Frame 12 was returned
    try std.testing.expectEqual(@as(usize, 12), result.rows[1].values[0].frame_type.index);
    // Mask was evaluated
    try std.testing.expect(result.rows[0].values[1] == .mask_type);
}

test "vdb: specific frame_id pushdown" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var mock = MockReader{ .frames_total = 20000 };
    var result = try db.executeQuery(
        "SELECT frame, sam3(frame, 'cat') AS mask FROM 'video.mp4' WHERE frame_id = 10000",
        mock.reader(),
        null,
        null,
    );
    defer result.deinit();

    // Exactly 1 frame evaluated (at index 10000)
    try std.testing.expectEqual(@as(usize, 1), result.rows.len);
    try std.testing.expectEqual(@as(usize, 10000), result.rows[0].values[0].frame_type.index);
}

test "vdb: frame_id pushdown intersected with yolov8 index" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    // Cat is only at frame 500 and 10000
    try builder.addDetection(500, 16500, "cat", 0.90, .{ .x = 0, .y = 0, .w = 1, .h = 1 });
    try builder.addDetection(10000, 330000, "cat", 0.95, .{ .x = 0, .y = 0, .w = 1, .h = 1 });

    try db.registerIndex("video.mp4", try builder.build());

    var mock = MockReader{ .frames_total = 20000 };

    // Query for frame_id = 10000 AND yolov8(frame) CONTAINS 'cat'
    var result1 = try db.executeQuery(
        "SELECT frame FROM 'video.mp4' WHERE frame_id = 10000 AND yolov8(frame) CONTAINS 'cat'",
        mock.reader(),
        null,
        null,
    );
    defer result1.deinit();
    try std.testing.expectEqual(@as(usize, 1), result1.rows.len);
    try std.testing.expectEqual(@as(usize, 10000), result1.rows[0].values[0].frame_type.index);

    // Query for frame_id = 1000 (which has no cat)
    var result2 = try db.executeQuery(
        "SELECT frame FROM 'video.mp4' WHERE frame_id = 1000 AND yolov8(frame) CONTAINS 'cat'",
        mock.reader(),
        null,
        null,
    );
    defer result2.deinit();
    try std.testing.expectEqual(@as(usize, 0), result2.rows.len);
}

test "vdb: query cancellation" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var mock = MockReader{ .frames_total = 1000 };
    var cancel_tok = std.atomic.Value(bool).init(true); // cancelled flag set to true

    const res = db.executeQuery(
        "SELECT frame FROM 'video.mp4'",
        mock.reader(),
        &cancel_tok,
        null,
    );
    try std.testing.expectError(error.QueryCancelled, res);
}

test "vdb: streaming row callback" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var mock = MockReader{ .frames_total = 10 };

    const Counter = struct {
        count: usize = 0,
        first_frame: ?usize = null,

        pub fn onRow(ctx: *anyopaque, row: *const types.Row) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.count += 1;
            if (self.first_frame == null) {
                self.first_frame = row.values[0].frame_type.index;
            }
        }
    };

    var counter = Counter{};
    const callback: engine.RowCallback = .{
        .ctx = &counter,
        .onRow = Counter.onRow,
    };

    var result = try db.executeQuery(
        "SELECT frame FROM 'video.mp4' LIMIT 3",
        mock.reader(),
        null,
        callback,
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), counter.count);
    try std.testing.expectEqual(@as(?usize, 0), counter.first_frame);
}
