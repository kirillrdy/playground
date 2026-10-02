const std = @import("std");

pub const types = @import("types.zig");
pub const ast = @import("ast.zig");
pub const lexer = @import("lexer.zig");
pub const parser = @import("parser.zig");
pub const index = @import("index.zig");
pub const planner = @import("planner.zig");
pub const engine = @import("engine.zig");
pub const overlay = @import("overlay.zig");
pub const query_tab = @import("query_tab.zig");

pub const query_input = @import("query_input.zig");
pub const completion = @import("completion.zig");

test {
    _ = overlay;
    _ = query_input;
    _ = completion;
    _ = query_tab;
}

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
        if (self.indexes.getPtr(video_path)) |existing| {
            existing.deinit();
            existing.* = idx;
            return;
        }
        const key = try self.allocator.dupe(u8, video_path);
        errdefer self.allocator.free(key);
        try self.indexes.put(self.allocator, key, idx);
    }

    pub fn getIndex(self: *const Database, video_path: []const u8) ?*const index.InvertedIndex {
        if (self.indexes.getPtr(video_path)) |idx| return idx;

        var it = self.indexes.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.endsWith(u8, key, video_path) or std.mem.endsWith(u8, video_path, key)) {
                return entry.value_ptr;
            }
            const key_base = std.fs.path.basename(key);
            const path_base = std.fs.path.basename(video_path);
            if (std.ascii.eqlIgnoreCase(key_base, path_base)) {
                return entry.value_ptr;
            }
        }

        if (self.indexes.count() == 1) {
            var it1 = self.indexes.iterator();
            if (it1.next()) |entry| return entry.value_ptr;
        }

        return null;
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

test "vdb: create index parser test" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const ally = arena.allocator();

    const q1 = "CREATE INDEX ON \"video.mp4\" USING sam3(\"person\") WITH (conf = 0.5);";
    var p1 = parser.Parser.init(ally, q1);
    const stmt1 = try p1.parse();
    try std.testing.expectEqualStrings("video.mp4", stmt1.create_index.source_file);
    try std.testing.expectEqualStrings("sam3", stmt1.create_index.model_name);
    try std.testing.expectEqualStrings("person", stmt1.create_index.prompt.?);
    try std.testing.expectEqual(@as(f32, 0.5), stmt1.create_index.min_conf);

    const q2 = "CREATE INDEX person_idx ON \"video.mp4\" (sam3(frame, \"person\")) WITH (conf = 0.6, step = 2);";
    var p2 = parser.Parser.init(ally, q2);
    const stmt2 = try p2.parse();
    try std.testing.expectEqualStrings("person_idx", stmt2.create_index.name);
    try std.testing.expectEqualStrings("video.mp4", stmt2.create_index.source_file);
    try std.testing.expectEqualStrings("person", stmt2.create_index.prompt.?);
    try std.testing.expectEqual(@as(f32, 0.6), stmt2.create_index.min_conf);
    try std.testing.expectEqual(@as(usize, 2), stmt2.create_index.sample_step);
}

test "vdb: sam3 index pushdown with confidence threshold" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    // Add person detections at frames 5, 10, 15 with varying confidence
    try builder.addDetection(5, 165, "person", 0.45, .{ .x = 0.1, .y = 0.1, .w = 0.2, .h = 0.2 });
    try builder.addDetection(10, 330, "person", 0.85, .{ .x = 0.1, .y = 0.1, .w = 0.2, .h = 0.2 });
    try builder.addDetection(15, 495, "person", 0.92, .{ .x = 0.1, .y = 0.1, .w = 0.2, .h = 0.2 });

    try db.registerIndex("test_sam3.mp4", try builder.build());

    var mock = MockReader{ .frames_total = 1000 };

    // Query with WHERE sam3(frame, "person") > 0.5 should only evaluate frames 10 and 15
    var result = try db.executeQuery(
        "SELECT frame FROM \"test_sam3.mp4\" WHERE sam3(frame, \"person\") > 0.5",
        mock.reader(),
        null,
        null,
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.rows.len);
    try std.testing.expectEqual(@as(usize, 10), result.rows[0].values[0].frame_type.index);
    try std.testing.expectEqual(@as(usize, 15), result.rows[1].values[0].frame_type.index);
}

test "vdb: sam3 index pushdown in projections without WHERE clause" {
    const ally = std.testing.allocator;

    var db = Database.init(ally);
    defer db.deinit();

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    try builder.addDetection(42, 1386, "person", 0.88, .{ .x = 0.1, .y = 0.1, .w = 0.2, .h = 0.2 });
    try db.registerIndex("test_sam3.mp4", try builder.build());

    var mock = MockReader{ .frames_total = 10000 };

    // Query without WHERE clause should use index pushdown from projection sam3(frame, "person")
    var result = try db.executeQuery(
        "SELECT frame, sam3(frame, \"person\") FROM \"test_sam3.mp4\"",
        mock.reader(),
        null,
        null,
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.rows.len);
    try std.testing.expectEqual(@as(usize, 42), result.rows[0].values[0].frame_type.index);
}

test "vdb: sam3 index serialization and file load/save roundtrip" {
    const ally = std.testing.allocator;

    var builder = index.IndexBuilder.init(ally);
    defer builder.deinit();

    try builder.addDetection(10, 330, "person", 0.95, .{ .x = 0.1, .y = 0.2, .w = 0.3, .h = 0.4 });
    try builder.addDetection(20, 660, "car", 0.85, .{ .x = 0.5, .y = 0.5, .w = 0.2, .h = 0.2 });

    const idx = try builder.build();
    defer {
        var mutable_idx = idx;
        mutable_idx.deinit();
    }

    const tmp_path = "test_index_roundtrip.vdb";
    try idx.saveToFile(tmp_path);
    defer index.InvertedIndex.deleteFile(ally, tmp_path) catch {};

    var loaded = try index.InvertedIndex.loadFromFile(ally, tmp_path);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(u32, 21), loaded.frame_count);
    const person_postings = loaded.lookup("person");
    try std.testing.expect(person_postings != null);
    try std.testing.expectEqual(@as(usize, 1), person_postings.?.len);
    try std.testing.expectEqual(@as(u32, 10), person_postings.?[0].frame_idx);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), person_postings.?[0].conf, 0.001);

    const car_postings = loaded.lookup("car");
    try std.testing.expect(car_postings != null);
    try std.testing.expectEqual(@as(usize, 1), car_postings.?.len);
    try std.testing.expectEqual(@as(u32, 20), car_postings.?[0].frame_idx);
}

test "vdb: replacing a video index releases the previous index" {
    const allocator = std.testing.allocator;
    var db = Database.init(allocator);
    defer db.deinit();
    var builder = index.IndexBuilder.init(allocator);
    defer builder.deinit();
    try builder.addDetection(1, 33, "person", 0.9, .{ .x = 0, .y = 0, .w = 1, .h = 1 });
    try db.registerIndex("video.mp4", try builder.build());
    try builder.addDetection(2, 66, "person", 0.9, .{ .x = 0, .y = 0, .w = 1, .h = 1 });
    try db.registerIndex("video.mp4", try builder.build());
    try std.testing.expectEqual(@as(u32, 1), db.indexes.count());
    try std.testing.expectEqual(@as(usize, 2), db.getIndex("video.mp4").?.lookup("person").?.len);
}

test "vdb: cancelling a background query does not interrupt a foreground query" {
    const Worker = struct {
        database: Database,
        reader: MockReader = .{ .frames_total = 10 },
        cancel: std.atomic.Value(bool) = .init(false),
        started: std.atomic.Value(bool) = .init(false),
        resume_query: std.atomic.Value(bool) = .init(false),
        cancelled: bool = false,

        fn onRow(ctx: *anyopaque, _: *const types.Row) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.started.store(true, .release);
            while (!self.resume_query.load(.acquire)) std.atomic.spinLoopHint();
        }

        fn run(self: *@This()) void {
            defer self.started.store(true, .release);
            var result = self.database.executeQuery(
                "SELECT frame FROM 'background.mp4'",
                self.reader.reader(),
                &self.cancel,
                .{ .ctx = self, .onRow = onRow },
            ) catch |err| {
                self.cancelled = err == error.QueryCancelled;
                return;
            };
            result.deinit();
        }
    };

    var worker = Worker{ .database = Database.init(std.testing.allocator) };
    defer worker.database.deinit();
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    var joined = false;
    defer if (!joined) {
        worker.resume_query.store(true, .release);
        thread.join();
    };
    while (!worker.started.load(.acquire)) std.atomic.spinLoopHint();

    // Change the foreground source while the background query is still streaming.
    var foreground = Database.init(std.testing.allocator);
    defer foreground.deinit();
    var reader = MockReader{ .frames_total = 20 };
    var result = try foreground.executeQuery(
        "SELECT frame FROM 'foreground.mp4' WHERE frame_id = 8",
        reader.reader(),
        null,
        null,
    );
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.rows.len);
    try std.testing.expectEqual(@as(usize, 8), result.rows[0].values[0].frame_type.index);

    worker.cancel.store(true, .release);
    worker.resume_query.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(worker.cancelled);
}

test "vdb: merged indexing jobs produce sorted unique candidate frames" {
    const allocator = std.testing.allocator;
    var earlier_job = index.IndexBuilder.init(allocator);
    defer earlier_job.deinit();
    const box: types.BBox = .{ .x = 0, .y = 0, .w = 1, .h = 1 };
    try earlier_job.addDetection(100, 3300, "person", 0.9, box);
    try earlier_job.addDetection(5, 165, "person", 0.8, box);
    var existing = try earlier_job.build();
    defer existing.deinit();

    var later_job = index.IndexBuilder.init(allocator);
    defer later_job.deinit();
    try later_job.addDetection(1, 33, "person", 0.9, box);
    try later_job.addDetection(5, 165, "person", 0.95, box);
    try later_job.addDetection(2, 66, "dog", 0.9, box);
    try later_job.importIndex(&existing);
    var merged = try later_job.build();
    defer merged.deinit();
    const frames = try merged.getMatchingFrames(allocator, "person", 0.4);
    defer allocator.free(frames);
    try std.testing.expectEqualSlices(u32, &.{ 1, 5, 100 }, frames);
    try std.testing.expectEqual(@as(usize, 1), merged.lookup("dog").?.len);
}

test "vdb: frame-only shorthand filters by SAM 3 without selecting an overlay" {
    const allocator = std.testing.allocator;
    const sql = try query_input.normalize(allocator, "select frame where sam3(frame, \"hat\" ) > 0.9", "video.mp4");
    defer allocator.free(sql);
    const prompts = try overlay.Prompts.fromSql(allocator, sql);
    try std.testing.expectEqual(@as(usize, 0), prompts.count);

    const Segmenter = struct {
        fn segment(_: *anyopaque, _: std.mem.Allocator, frame: types.FrameRef, prompt: []const u8) anyerror!types.MaskRef {
            try std.testing.expectEqualStrings("hat", prompt);
            return .{
                .score = if (frame.index == 1) 0.95 else 0.85,
                .coverage = 0.2,
                .width = frame.width,
                .height = frame.height,
            };
        }
    };
    var reader = MockReader{ .frames_total = 3 };
    var db = Database.init(allocator);
    defer db.deinit();
    db.engine_inst.sam3 = .{ .ptr = &reader, .segmentFn = Segmenter.segment };
    var result = try db.executeQuery(sql, reader.reader(), null, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.rows.len);
    try std.testing.expectEqual(@as(usize, 1), result.rows[0].values.len);
    try std.testing.expectEqual(@as(usize, 1), result.rows[0].values[0].frame_type.index);
}
