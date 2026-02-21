const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ImageSize = struct {
    width: usize,
    height: usize,
};

pub fn rgbU8ToNchwF32(
    allocator: Allocator,
    src_rgb: []const u8,
    src_size: ImageSize,
    dst_size: ImageSize,
) ![]f32 {
    if (src_rgb.len != src_size.width * src_size.height * 3) return error.InvalidSourceSize;
    const dst_len = 3 * dst_size.width * dst_size.height;
    var out = try allocator.alloc(f32, dst_len);
    errdefer allocator.free(out);

    const plane = dst_size.width * dst_size.height;
    for (0..dst_size.height) |dy| {
        const sy = mapNearest(dy, dst_size.height, src_size.height);
        for (0..dst_size.width) |dx| {
            const sx = mapNearest(dx, dst_size.width, src_size.width);
            const src_idx = ((sy * src_size.width) + sx) * 3;
            const dst_idx = (dy * dst_size.width) + dx;

            // CHW order and [0,1] normalization.
            out[dst_idx] = @as(f32, @floatFromInt(src_rgb[src_idx])) / 255.0;
            out[plane + dst_idx] = @as(f32, @floatFromInt(src_rgb[src_idx + 1])) / 255.0;
            out[(2 * plane) + dst_idx] = @as(f32, @floatFromInt(src_rgb[src_idx + 2])) / 255.0;
        }
    }
    return out;
}

fn mapNearest(dst_idx: usize, dst_len: usize, src_len: usize) usize {
    // Center-point nearest-neighbor mapping.
    const num = (2 * dst_idx + 1) * src_len;
    const den = 2 * dst_len;
    var mapped = num / den;
    if (mapped >= src_len) mapped = src_len - 1;
    return mapped;
}

test "rgbU8ToNchwF32 preserves single pixel" {
    const allocator = std.testing.allocator;
    const src = [_]u8{ 255, 127, 0 };
    const out = try rgbU8ToNchwF32(
        allocator,
        &src,
        .{ .width = 1, .height = 1 },
        .{ .width = 1, .height = 1 },
    );
    defer allocator.free(out);

    try std.testing.expectEqual(@as(usize, 3), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 127.0 / 255.0), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 0.0001);
}

test "rgbU8ToNchwF32 2x2 to 1x1 picks nearest center" {
    const allocator = std.testing.allocator;
    // Row-major RGB:
    // (0,0) red      (1,0) green
    // (0,1) blue     (1,1) white
    const src = [_]u8{
        255, 0, 0, 0, 255, 0,
        0,   0, 255, 255, 255, 255,
    };
    const out = try rgbU8ToNchwF32(
        allocator,
        &src,
        .{ .width = 2, .height = 2 },
        .{ .width = 1, .height = 1 },
    );
    defer allocator.free(out);

    // center maps to bottom-right in this mapping => white
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 0.0001);
}
