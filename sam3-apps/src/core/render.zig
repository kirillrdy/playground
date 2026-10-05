const std = @import("std");
const sam3 = @import("sam3");
const zigimg = @import("zigimg");

pub const MaskColor = struct { red: u8, green: u8, blue: u8 };
pub const mask_colors = [_]MaskColor{
    .{ .red = 0, .green = 220, .blue = 100 },
    .{ .red = 0, .green = 180, .blue = 255 },
    .{ .red = 255, .green = 175, .blue = 0 },
    .{ .red = 220, .green = 90, .blue = 255 },
    .{ .red = 255, .green = 80, .blue = 110 },
    .{ .red = 255, .green = 225, .blue = 80 },
    .{ .red = 70, .green = 220, .blue = 210 },
    .{ .red = 160, .green = 155, .blue = 255 },
};

pub fn maskColor(index: usize) MaskColor {
    return mask_colors[index % mask_colors.len];
}

pub fn clampMaskIndex(coordinate: f32, limit: usize) usize {
    if (coordinate <= 0) return 0;
    return @min(@as(usize, @intFromFloat(@floor(coordinate))), limit - 1);
}

pub fn blendChannel(original: u8, tint: u8, alpha: f32) u8 {
    return @intFromFloat(@as(f32, @floatFromInt(original)) * (1 - alpha) + @as(f32, @floatFromInt(tint)) * alpha);
}

pub fn overlayMask(frame: []u8, img: zigimg.Image, masks: sam3.Masks, index: usize, alpha: f32) void {
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
            frame[offset] = blendChannel(frame[offset], color.red, alpha);
            frame[offset + 1] = blendChannel(frame[offset + 1], color.green, alpha);
            frame[offset + 2] = blendChannel(frame[offset + 2], color.blue, alpha);
        }
    }
}

pub fn drawPointMarkers(frame: []u8, img: zigimg.Image, points: []const sam3.Point) void {
    for (points) |point| {
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
                frame[offset] = color.red;
                frame[offset + 1] = color.green;
                frame[offset + 2] = color.blue;
            }
        }
    }
}
