const std = @import("std");
const ast = @import("ast.zig");
const index_mod = @import("index.zig");

pub const ScanStrategy = union(enum) {
    full_scan: struct {
        step: usize,
    },
    indexed_seek: struct {
        candidate_frames: []const u32,
    },
};

pub const Plan = struct {
    allocator: std.mem.Allocator,
    source_path: []const u8,
    projections: []const ast.Projection,
    where_clause: ?*ast.Expr,
    limit: ?usize,
    strategy: ScanStrategy,
    requires_sam3: bool,
    requires_yolo: bool,
    sam3_prompt: ?[]const u8,

    pub fn deinit(self: *Plan) void {
        switch (self.strategy) {
            .indexed_seek => |seek| self.allocator.free(seek.candidate_frames),
            .full_scan => {},
        }
    }
};

pub const Planner = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Planner {
        return .{ .allocator = allocator };
    }

    pub fn createPlan(
        self: *Planner,
        stmt: ast.SelectStmt,
        available_index: ?*const index_mod.InvertedIndex,
    ) !Plan {
        const source_path = switch (stmt.source) {
            .file_path => |p| p,
            .call => |c| c.path,
        };

        const default_step: usize = switch (stmt.source) {
            .file_path => 1,
            .call => |c| c.step,
        };

        var requires_sam3 = false;
        var requires_yolo = false;
        var sam3_prompt: ?[]const u8 = null;

        // Inspect projections
        for (stmt.projections) |proj| {
            inspectExpr(proj.expr, &requires_sam3, &requires_yolo, &sam3_prompt);
        }

        if (stmt.where_clause) |w| {
            inspectExpr(w, &requires_sam3, &requires_yolo, &sam3_prompt);
        }

        // Try index or frame pushdown
        var candidate_frames: ?[]u32 = null;
        if (stmt.where_clause) |w| {
            candidate_frames = try self.tryPushdown(w, available_index);
        } else {
            // Check projections for frame(N)
            for (stmt.projections) |proj| {
                if (proj.expr.* == .call) {
                    const c = proj.expr.call;
                    if (std.ascii.eqlIgnoreCase(c.name, "frame") and c.args.len == 1) {
                        if (c.args[0].* == .literal_int) {
                            const f_idx: u32 = @intCast(@max(c.args[0].literal_int, 0));
                            const single = try self.allocator.alloc(u32, 1);
                            single[0] = f_idx;
                            candidate_frames = single;
                            break;
                        }
                    }
                }
            }
            // If no explicit frame(N), check if sam3_prompt is present and indexed
            if (candidate_frames == null and available_index != null and sam3_prompt != null) {
                const matches = try available_index.?.getMatchingFrames(self.allocator, sam3_prompt.?, 0.4);
                if (matches.len > 0) {
                    candidate_frames = matches;
                } else {
                    self.allocator.free(matches);
                }
            }
        }

        const strategy: ScanStrategy = if (candidate_frames) |frames|
            .{ .indexed_seek = .{ .candidate_frames = frames } }
        else
            .{ .full_scan = .{ .step = default_step } };

        return Plan{
            .allocator = self.allocator,
            .source_path = source_path,
            .projections = stmt.projections,
            .where_clause = stmt.where_clause,
            .limit = stmt.limit,
            .strategy = strategy,
            .requires_sam3 = requires_sam3,
            .requires_yolo = requires_yolo,
            .sam3_prompt = sam3_prompt,
        };
    }

    fn inspectExpr(
        expr: *ast.Expr,
        requires_sam3: *bool,
        requires_yolo: *bool,
        sam3_prompt: *?[]const u8,
    ) void {
        switch (expr.*) {
            .call => |c| {
                if (std.ascii.eqlIgnoreCase(c.name, "sam3")) {
                    requires_sam3.* = true;
                    if (c.args.len >= 2 and c.args[1].* == .literal_string) {
                        sam3_prompt.* = c.args[1].literal_string;
                    }
                } else if (std.ascii.eqlIgnoreCase(c.name, "yolov8")) {
                    requires_yolo.* = true;
                }
                for (c.args) |arg| {
                    inspectExpr(arg, requires_sam3, requires_yolo, sam3_prompt);
                }
            },
            .binary => |b| {
                inspectExpr(b.left, requires_sam3, requires_yolo, sam3_prompt);
                inspectExpr(b.right, requires_sam3, requires_yolo, sam3_prompt);
            },
            .unary => |u| {
                inspectExpr(u.expr, requires_sam3, requires_yolo, sam3_prompt);
            },
            else => {},
        }
    }

    fn tryPushdown(
        self: *Planner,
        expr: *ast.Expr,
        idx: ?*const index_mod.InvertedIndex,
    ) !?[]u32 {
        switch (expr.*) {
            .call => {
                if (isSam3Call(expr)) |prompt| {
                    if (idx) |sam_idx| {
                        return try sam_idx.getMatchingFrames(self.allocator, prompt, 0.4);
                    }
                }
            },
            .binary => |b| {
                // Check frame_id = 10000 or 10000 = frame_id
                if (b.op == .eq) {
                    if (isFrameIdent(b.left) and b.right.* == .literal_int) {
                        const target: u32 = @intCast(@max(b.right.literal_int, 0));
                        const single = try self.allocator.alloc(u32, 1);
                        single[0] = target;
                        return single;
                    } else if (isFrameIdent(b.right) and b.left.* == .literal_int) {
                        const target: u32 = @intCast(@max(b.left.literal_int, 0));
                        const single = try self.allocator.alloc(u32, 1);
                        single[0] = target;
                        return single;
                    }
                }

                if (b.op == .gt or b.op == .gte) {
                    if (isSam3Call(b.left)) |prompt| {
                        if (idx) |sam_idx| {
                            var min_conf: f32 = 0.4;
                            if (b.right.* == .literal_float) min_conf = @floatCast(b.right.literal_float);
                            if (b.right.* == .literal_int) min_conf = @floatFromInt(b.right.literal_int);
                            return try sam_idx.getMatchingFrames(self.allocator, prompt, min_conf);
                        }
                    } else if (isSam3Call(b.right)) |prompt| {
                        if (idx) |sam_idx| {
                            var min_conf: f32 = 0.4;
                            if (b.left.* == .literal_float) min_conf = @floatCast(b.left.literal_float);
                            if (b.left.* == .literal_int) min_conf = @floatFromInt(b.left.literal_int);
                            return try sam_idx.getMatchingFrames(self.allocator, prompt, min_conf);
                        }
                    }
                }

                if (b.op == .lt or b.op == .lte) {
                    if (isSam3Call(b.right)) |prompt| {
                        if (idx) |sam_idx| {
                            var min_conf: f32 = 0.4;
                            if (b.left.* == .literal_float) min_conf = @floatCast(b.left.literal_float);
                            if (b.left.* == .literal_int) min_conf = @floatFromInt(b.left.literal_int);
                            return try sam_idx.getMatchingFrames(self.allocator, prompt, min_conf);
                        }
                    }
                }

                if (b.op == .contains) {
                    if (isSam3Call(b.left)) |prompt| {
                        if (idx) |sam_idx| {
                            const target_label = if (b.right.* == .literal_string) b.right.literal_string else prompt;
                            return try sam_idx.getMatchingFrames(self.allocator, target_label, 0.25);
                        }
                    } else if (isYoloCall(b.left) and b.right.* == .literal_string) {
                        if (idx) |yolo_idx| {
                            const target_label = b.right.literal_string;
                            return try yolo_idx.getMatchingFrames(self.allocator, target_label, 0.25);
                        }
                    }
                } else if (b.op == .and_op) {
                    const left_opt = try self.tryPushdown(b.left, idx);
                    const right_opt = try self.tryPushdown(b.right, idx);

                    if (left_opt != null and right_opt != null) {
                        defer self.allocator.free(left_opt.?);
                        defer self.allocator.free(right_opt.?);
                        return try intersectSlices(self.allocator, left_opt.?, right_opt.?);
                    } else if (left_opt != null) {
                        return left_opt;
                    } else if (right_opt != null) {
                        return right_opt;
                    }
                } else if (b.op == .or_op) {
                    const left_opt = try self.tryPushdown(b.left, idx);
                    const right_opt = try self.tryPushdown(b.right, idx);

                    if (left_opt != null and right_opt != null) {
                        defer self.allocator.free(left_opt.?);
                        defer self.allocator.free(right_opt.?);
                        return try unionSlices(self.allocator, left_opt.?, right_opt.?);
                    }
                }
            },
            else => {},
        }
        return null;
    }

    fn isFrameIdent(expr: *ast.Expr) bool {
        return switch (expr.*) {
            .column_ref => |c| std.ascii.eqlIgnoreCase(c, "frame_id") or
                std.ascii.eqlIgnoreCase(c, "frame_idx") or
                std.ascii.eqlIgnoreCase(c, "frame_number") or
                std.ascii.eqlIgnoreCase(c, "index"),
            else => false,
        };
    }

    fn isYoloCall(expr: *ast.Expr) bool {
        return switch (expr.*) {
            .call => |c| std.ascii.eqlIgnoreCase(c.name, "yolov8"),
            else => false,
        };
    }

    fn isSam3Call(expr: *ast.Expr) ?[]const u8 {
        return switch (expr.*) {
            .call => |c| {
                if (std.ascii.eqlIgnoreCase(c.name, "sam3")) {
                    if (c.args.len >= 2 and c.args[1].* == .literal_string) {
                        return c.args[1].literal_string;
                    } else if (c.args.len >= 1 and c.args[0].* == .literal_string) {
                        return c.args[0].literal_string;
                    }
                }
                return null;
            },
            else => null,
        };
    }

    fn intersectSlices(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        var result: std.ArrayList(u32) = .empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        var j: usize = 0;
        while (i < a.len and j < b.len) {
            if (a[i] == b[j]) {
                try result.append(allocator, a[i]);
                i += 1;
                j += 1;
            } else if (a[i] < b[j]) {
                i += 1;
            } else {
                j += 1;
            }
        }
        return result.toOwnedSlice(allocator);
    }

    fn unionSlices(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
        var result: std.ArrayList(u32) = .empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        var j: usize = 0;
        var last: ?u32 = null;

        while (i < a.len or j < b.len) {
            var val: u32 = undefined;
            if (i < a.len and j < b.len) {
                if (a[i] < b[j]) {
                    val = a[i];
                    i += 1;
                } else if (b[j] < a[i]) {
                    val = b[j];
                    j += 1;
                } else {
                    val = a[i];
                    i += 1;
                    j += 1;
                }
            } else if (i < a.len) {
                val = a[i];
                i += 1;
            } else {
                val = b[j];
                j += 1;
            }

            if (last == null or last.? != val) {
                try result.append(allocator, val);
                last = val;
            }
        }
        return result.toOwnedSlice(allocator);
    }
};
