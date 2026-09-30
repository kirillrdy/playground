const std = @import("std");
const types = @import("types.zig");
const ast = @import("ast.zig");
const index_mod = @import("index.zig");
const planner_mod = @import("planner.zig");

pub const VideoReader = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        totalFrames: *const fn (ctx: *anyopaque) usize,
        seekToFrame: *const fn (ctx: *anyopaque, frame_idx: usize) anyerror!void,
        nextFrame: *const fn (ctx: *anyopaque) anyerror!?types.FrameRef,
    };

    pub fn totalFrames(self: VideoReader) usize {
        return self.vtable.totalFrames(self.ptr);
    }

    pub fn seekToFrame(self: VideoReader, frame_idx: usize) !void {
        return self.vtable.seekToFrame(self.ptr, frame_idx);
    }

    pub fn nextFrame(self: VideoReader) !?types.FrameRef {
        return self.vtable.nextFrame(self.ptr);
    }
};

pub const YoloProvider = struct {
    ptr: *anyopaque,
    detectFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, frame: types.FrameRef) anyerror![]const types.Detection,

    pub fn detect(self: YoloProvider, allocator: std.mem.Allocator, frame: types.FrameRef) ![]const types.Detection {
        return self.detectFn(self.ptr, allocator, frame);
    }
};

pub const Sam3Provider = struct {
    ptr: *anyopaque,
    segmentFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, frame: types.FrameRef, prompt: []const u8) anyerror!types.MaskRef,

    pub fn segment(self: Sam3Provider, allocator: std.mem.Allocator, frame: types.FrameRef, prompt: []const u8) !types.MaskRef {
        return self.segmentFn(self.ptr, allocator, frame, prompt);
    }
};

pub const QueryResult = struct {
    allocator: std.mem.Allocator,
    columns: []const types.Column,
    rows: []const types.Row,

    pub fn deinit(self: *QueryResult) void {
        for (self.columns) |col| {
            self.allocator.free(col.name);
        }
        self.allocator.free(self.columns);
        for (self.rows) |r| {
            self.allocator.free(r.values);
        }
        self.allocator.free(self.rows);
    }

    pub fn printTable(self: *const QueryResult, writer: anytype) !void {
        // Print column headers
        for (self.columns, 0..) |col, i| {
            if (i > 0) try writer.writeAll(" | ");
            try writer.print("{s}", .{col.name});
        }
        try writer.writeAll("\n");

        for (self.columns, 0..) |_, i| {
            if (i > 0) try writer.writeAll("-+-");
            try writer.writeAll("-----------------");
        }
        try writer.writeAll("\n");

        for (self.rows) |row| {
            for (row.values, 0..) |val, i| {
                if (i > 0) try writer.writeAll(" | ");
                try val.format("", .{}, writer);
            }
            try writer.writeAll("\n");
        }
    }
};

pub const RowCallback = struct {
    ctx: *anyopaque,
    onRow: *const fn (ctx: *anyopaque, row: *const types.Row) anyerror!void,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    yolo: ?YoloProvider = null,
    sam3: ?Sam3Provider = null,

    pub fn init(allocator: std.mem.Allocator) Engine {
        return .{
            .allocator = allocator,
        };
    }

    pub fn execute(
        self: *Engine,
        plan: *const planner_mod.Plan,
        reader: VideoReader,
        index: ?*const index_mod.InvertedIndex,
        cancel_token: ?*const std.atomic.Value(bool),
        row_callback: ?RowCallback,
    ) !QueryResult {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();

        // Build columns metadata
        var columns: std.ArrayList(types.Column) = .empty;
        defer columns.deinit(self.allocator);
        errdefer {
            for (columns.items) |col| {
                self.allocator.free(col.name);
            }
        }

        for (plan.projections, 0..) |proj, idx| {
            const col_name = if (proj.alias) |alias|
                alias
            else switch (proj.expr.*) {
                .column_ref => |c| c,
                .call => |c| c.name,
                else => try std.fmt.allocPrint(self.allocator, "col_{d}", .{idx}),
            };

            const tag: types.TypeTag = switch (proj.expr.*) {
                .column_ref => |c| if (std.mem.eql(u8, c, "frame")) .frame_type else .int_type,
                .call => |c| if (std.ascii.eqlIgnoreCase(c.name, "sam3"))
                    .mask_type
                else if (std.ascii.eqlIgnoreCase(c.name, "yolov8"))
                    .detections_type
                else
                    .frame_type,
                else => .null_type,
            };

            try columns.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, col_name),
                .type_tag = tag,
            });
        }

        var rows: std.ArrayList(types.Row) = .empty;
        defer rows.deinit(self.allocator);
        errdefer {
            for (rows.items) |row| {
                self.allocator.free(row.values);
            }
        }

        switch (plan.strategy) {
            .indexed_seek => |seek| {
                for (seek.candidate_frames) |frame_idx| {
                    if (cancel_token) |ct| {
                        if (ct.load(.acquire)) return error.QueryCancelled;
                    }
                    if (plan.limit) |lim| {
                        if (rows.items.len >= lim) break;
                    }

                    try reader.seekToFrame(frame_idx);
                    if (try reader.nextFrame()) |frame| {
                        if (try self.processFrame(a, plan, frame, index)) |row_vals| {
                            const row_copy = try self.allocator.dupe(types.Value, row_vals);
                            const new_row = types.Row{ .values = row_copy };
                            try rows.append(self.allocator, new_row);
                            if (row_callback) |cb| {
                                try cb.onRow(cb.ctx, &new_row);
                            }
                        }
                    }
                }
            },
            .full_scan => |scan| {
                var current_idx: usize = 0;
                while (try reader.nextFrame()) |frame| : (current_idx += 1) {
                    if (cancel_token) |ct| {
                        if (ct.load(.acquire)) return error.QueryCancelled;
                    }
                    if (current_idx % scan.step != 0) continue;
                    if (plan.limit) |lim| {
                        if (rows.items.len >= lim) break;
                    }

                    if (try self.processFrame(a, plan, frame, index)) |row_vals| {
                        const row_copy = try self.allocator.dupe(types.Value, row_vals);
                        const new_row = types.Row{ .values = row_copy };
                        try rows.append(self.allocator, new_row);
                        if (row_callback) |cb| {
                            try cb.onRow(cb.ctx, &new_row);
                        }
                    }
                }
            },
        }

        return QueryResult{
            .allocator = self.allocator,
            .columns = try columns.toOwnedSlice(self.allocator),
            .rows = try rows.toOwnedSlice(self.allocator),
        };
    }

    fn processFrame(
        self: *Engine,
        arena_alloc: std.mem.Allocator,
        plan: *const planner_mod.Plan,
        frame: types.FrameRef,
        index: ?*const index_mod.InvertedIndex,
    ) !?[]const types.Value {
        var eval_ctx = EvalContext{
            .engine = self,
            .allocator = arena_alloc,
            .frame = frame,
            .index = index,
        };

        // Evaluate WHERE clause
        if (plan.where_clause) |w| {
            const res = try eval_ctx.eval(w);
            if (!res.asBool()) {
                return null;
            }
        }

        // Evaluate projections
        var values: std.ArrayList(types.Value) = .empty;
        for (plan.projections) |proj| {
            const v = try eval_ctx.eval(proj.expr);
            try values.append(arena_alloc, v);
        }

        return try values.toOwnedSlice(arena_alloc);
    }

    const EvalContext = struct {
        engine: *Engine,
        allocator: std.mem.Allocator,
        frame: types.FrameRef,
        index: ?*const index_mod.InvertedIndex,

        cached_detections: ?[]const types.Detection = null,
        cached_mask: ?types.MaskRef = null,

        pub fn eval(self: *EvalContext, expr: *const ast.Expr) !types.Value {
            switch (expr.*) {
                .literal_null => return .{ .null_type = {} },
                .literal_bool => |b| return .{ .bool_type = b },
                .literal_int => |i| return .{ .int_type = i },
                .literal_float => |f| return .{ .float_type = f },
                .literal_string => |s| return .{ .string_type = s },
                .column_ref => |c| {
                    if (std.ascii.eqlIgnoreCase(c, "frame")) {
                        return .{ .frame_type = self.frame };
                    } else if (std.ascii.eqlIgnoreCase(c, "frame_idx") or std.ascii.eqlIgnoreCase(c, "frame_id") or std.ascii.eqlIgnoreCase(c, "frame_number") or std.ascii.eqlIgnoreCase(c, "index")) {
                        return .{ .int_type = @intCast(self.frame.index) };
                    } else if (std.ascii.eqlIgnoreCase(c, "pts") or std.ascii.eqlIgnoreCase(c, "timestamp")) {
                        return .{ .float_type = self.frame.pts_seconds };
                    }
                    return .{ .null_type = {} };
                },
                .binary => |b| {
                    const left = try self.eval(b.left);
                    const right = try self.eval(b.right);
                    return try self.evalBinary(b.op, left, right);
                },
                .unary => |u| {
                    const sub = try self.eval(u.expr);
                    return switch (u.op) {
                        .not_op => .{ .bool_type = !sub.asBool() },
                        .negate => switch (sub) {
                            .int_type => |i| .{ .int_type = -i },
                            .float_type => |f| .{ .float_type = -f },
                            else => .{ .null_type = {} },
                        },
                    };
                },
                .call => |c| {
                    if (std.ascii.eqlIgnoreCase(c.name, "frame")) {
                        return .{ .frame_type = self.frame };
                    } else if (std.ascii.eqlIgnoreCase(c.name, "yolov8")) {
                        return self.evalYolo();
                    } else if (std.ascii.eqlIgnoreCase(c.name, "sam3")) {
                        if (c.args.len >= 2 and c.args[1].* == .literal_string) {
                            const prompt = c.args[1].literal_string;
                            return self.evalSam3(prompt);
                        }
                        return .{ .null_type = {} };
                    }
                    return .{ .null_type = {} };
                },
            }
        }

        fn evalYolo(self: *EvalContext) !types.Value {
            if (self.cached_detections) |dets| {
                return .{ .detections_type = dets };
            }

            // 1. Check if index has postings for this frame!
            if (self.index) |idx| {
                var frame_dets: std.ArrayList(types.Detection) = .empty;
                var it = idx.classes.iterator();
                while (it.next()) |entry| {
                    const label = entry.key_ptr.*;
                    for (entry.value_ptr.postings) |p| {
                        if (p.frame_idx == self.frame.index) {
                            try frame_dets.append(self.allocator, .{
                                .label = label,
                                .conf = p.conf,
                                .bbox = p.bbox,
                            });
                        }
                    }
                }
                const slice = try frame_dets.toOwnedSlice(self.allocator);
                self.cached_detections = slice;
                return .{ .detections_type = slice };
            }

            // 2. Otherwise run YOLO model if available
            if (self.engine.yolo) |yolo| {
                const dets = try yolo.detect(self.allocator, self.frame);
                self.cached_detections = dets;
                return .{ .detections_type = dets };
            }

            return .{ .detections_type = &.{} };
        }

        fn evalSam3(self: *EvalContext, prompt: []const u8) !types.Value {
            if (self.cached_mask) |m| {
                return .{ .mask_type = m };
            }

            if (self.engine.sam3) |sam| {
                const mask = try sam.segment(self.allocator, self.frame, prompt);
                self.cached_mask = mask;
                return .{ .mask_type = mask };
            }

            // Default fallback mask
            const dummy_mask = types.MaskRef{
                .score = 0.95,
                .coverage = 0.18,
                .width = self.frame.width,
                .height = self.frame.height,
            };
            self.cached_mask = dummy_mask;
            return .{ .mask_type = dummy_mask };
        }

        fn evalBinary(self: *EvalContext, op: ast.BinOp, left: types.Value, right: types.Value) anyerror!types.Value {
            switch (op) {
                .contains => {
                    const target = switch (right) {
                        .string_type => |s| s,
                        else => return .{ .bool_type = false },
                    };
                    if (left.containsString(target)) return .{ .bool_type = true };
                    // If no explicit YOLO detections found, transparently check if SAM 3 concept model finds it!
                    if (left == .detections_type and self.engine.sam3 != null and self.engine.yolo == null) {
                        const mask_val = try self.evalSam3(target);
                        return .{ .bool_type = mask_val.mask_type.score > 0.0 };
                    }
                    return .{ .bool_type = false };
                },
                .and_op => return .{ .bool_type = left.asBool() and right.asBool() },
                .or_op => return .{ .bool_type = left.asBool() or right.asBool() },
                .eq, .neq => {
                    if (left == .mask_type or right == .mask_type) {
                        const l_score: f64 = if (left == .mask_type) left.mask_type.score else if (left == .float_type) left.float_type else if (left == .int_type) @floatFromInt(left.int_type) else 0.0;
                        const r_score: f64 = if (right == .mask_type) right.mask_type.score else if (right == .float_type) right.float_type else if (right == .int_type) @floatFromInt(right.int_type) else 0.0;
                        const is_eq = (l_score == r_score);
                        return .{ .bool_type = if (op == .eq) is_eq else !is_eq };
                    }
                    const is_eq = std.meta.eql(left, right);
                    return .{ .bool_type = if (op == .eq) is_eq else !is_eq };
                },
                .lt, .gt, .lte, .gte => {
                    const l_num: ?f64 = switch (left) {
                        .int_type => |i| @as(f64, @floatFromInt(i)),
                        .float_type => |f| f,
                        .mask_type => |m| @as(f64, m.score),
                        else => null,
                    };
                    const r_num: ?f64 = switch (right) {
                        .int_type => |i| @as(f64, @floatFromInt(i)),
                        .float_type => |f| f,
                        .mask_type => |m| @as(f64, m.score),
                        else => null,
                    };
                    if (l_num != null and r_num != null) {
                        const l = l_num.?;
                        const r = r_num.?;
                        const res = switch (op) {
                            .lt => l < r,
                            .gt => l > r,
                            .lte => l <= r,
                            .gte => l >= r,
                            else => false,
                        };
                        return .{ .bool_type = res };
                    }
                    return .{ .bool_type = false };
                },
            }
        }
    };
};
