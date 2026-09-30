const std = @import("std");

pub const BBox = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    pub fn format(self: BBox, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("[{d:.2}, {d:.2}, {d:.2}, {d:.2}]", .{ self.x, self.y, self.w, self.h });
    }
};

pub const Detection = struct {
    class_id: u16 = 0,
    label: []const u8,
    conf: f32,
    bbox: BBox,

    pub fn format(self: Detection, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{s}({d:.2})@{s}", .{ self.label, self.conf, self.bbox });
    }
};

pub const FrameRef = struct {
    index: usize,
    pts_seconds: f64,
    width: u32,
    height: u32,
    rgb: ?[*]const u8 = null,

    pub fn format(self: FrameRef, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Frame#{d}({d:.3}s, {d}x{d})", .{ self.index, self.pts_seconds, self.width, self.height });
    }
};

pub const MaskRef = struct {
    score: f32,
    coverage: f32,
    width: u32,
    height: u32,
    bytes: ?[]const u8 = null,

    pub fn format(self: MaskRef, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Mask(score={d:.2}, cov={d:.2}%)", .{ self.score, self.coverage * 100.0 });
    }
};

pub const TypeTag = enum {
    null_type,
    bool_type,
    int_type,
    float_type,
    string_type,
    frame_type,
    detections_type,
    mask_type,
};

pub const Value = union(TypeTag) {
    null_type: void,
    bool_type: bool,
    int_type: i64,
    float_type: f64,
    string_type: []const u8,
    frame_type: FrameRef,
    detections_type: []const Detection,
    mask_type: MaskRef,

    pub fn asBool(self: Value) bool {
        return switch (self) {
            .null_type => false,
            .bool_type => |b| b,
            .int_type => |i| i != 0,
            .float_type => |f| f != 0.0,
            .string_type => |s| s.len > 0,
            .frame_type => true,
            .detections_type => |dets| dets.len > 0,
            .mask_type => |m| m.score > 0.0,
        };
    }

    pub fn containsString(self: Value, target: []const u8) bool {
        switch (self) {
            .string_type => |s| return std.mem.indexOf(u8, s, target) != null,
            .detections_type => |dets| {
                for (dets) |d| {
                    if (std.ascii.eqlIgnoreCase(d.label, target)) {
                        return true;
                    }
                }
                return false;
            },
            else => return false,
        }
    }

    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        switch (self) {
            .null_type => try writer.writeAll("NULL"),
            .bool_type => |b| try writer.print("{}", .{b}),
            .int_type => |i| try writer.print("{d}", .{i}),
            .float_type => |f| try writer.print("{d:.4}", .{f}),
            .string_type => |s| try writer.print("'{s}'", .{s}),
            .frame_type => |f| try f.format(fmt, options, writer),
            .detections_type => |dets| {
                try writer.writeAll("[");
                for (dets, 0..) |d, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try d.format(fmt, options, writer);
                }
                try writer.writeAll("]");
            },
            .mask_type => |m| try m.format(fmt, options, writer),
        }
    }
};

pub const Column = struct {
    name: []const u8,
    type_tag: TypeTag,
};

pub const Row = struct {
    values: []const Value,
};
