const std = @import("std");
const Allocator = std.mem.Allocator;
const preprocess = @import("image_preprocess.zig");
const zigimg = @import("zigimg");

pub const DecodedImage = struct {
    size: preprocess.ImageSize,
    rgb: []u8,
};

pub fn decodeRgb(allocator: Allocator, bytes: []const u8) !DecodedImage {
    if (zigimgDecodeRgb(allocator, bytes)) |decoded| {
        return decoded;
    } else |_| {
        return decodePpmP6(allocator, bytes);
    }
}

fn zigimgDecodeRgb(allocator: Allocator, bytes: []const u8) !DecodedImage {
    var image = try zigimg.Image.fromMemory(allocator, bytes);
    defer image.deinit(allocator);

    if (image.pixelFormat() != zigimg.PixelFormat.rgb24) {
        try image.convert(allocator, .rgb24);
    }

    const pixels = image.pixels.rgb24;
    const rgb = try allocator.alloc(u8, pixels.len * 3);
    for (pixels, 0..) |pixel, idx| {
        const base = idx * 3;
        rgb[base] = pixel.r;
        rgb[base + 1] = pixel.g;
        rgb[base + 2] = pixel.b;
    }

    return .{
        .size = .{
            .width = image.width,
            .height = image.height,
        },
        .rgb = rgb,
    };
}

// Minimal binary PPM (P6) decoder.
fn decodePpmP6(allocator: Allocator, bytes: []const u8) !DecodedImage {
    var idx: usize = 0;
    const magic = try readToken(bytes, &idx);
    if (!std.mem.eql(u8, magic, "P6")) return error.UnsupportedImageFormat;

    const width_token = try readToken(bytes, &idx);
    const height_token = try readToken(bytes, &idx);
    const max_token = try readToken(bytes, &idx);

    const width = try std.fmt.parseInt(usize, width_token, 10);
    const height = try std.fmt.parseInt(usize, height_token, 10);
    const max_value = try std.fmt.parseInt(usize, max_token, 10);
    if (max_value != 255) return error.UnsupportedImageFormat;

    if (idx >= bytes.len or !isWhitespace(bytes[idx])) return error.InvalidImageData;
    idx += 1;

    const pixel_len = width * height * 3;
    if (bytes.len < idx + pixel_len) return error.InvalidImageData;
    const rgb = try allocator.dupe(u8, bytes[idx .. idx + pixel_len]);

    return .{
        .size = .{ .width = width, .height = height },
        .rgb = rgb,
    };
}

fn readToken(bytes: []const u8, idx: *usize) ![]const u8 {
    skipWhitespaceAndComments(bytes, idx);
    if (idx.* >= bytes.len) return error.InvalidImageData;
    const start = idx.*;
    while (idx.* < bytes.len and !isWhitespace(bytes[idx.*])) : (idx.* += 1) {}
    if (start == idx.*) return error.InvalidImageData;
    return bytes[start..idx.*];
}

fn skipWhitespaceAndComments(bytes: []const u8, idx: *usize) void {
    while (idx.* < bytes.len) {
        if (isWhitespace(bytes[idx.*])) {
            idx.* += 1;
            continue;
        }
        if (bytes[idx.*] == '#') {
            while (idx.* < bytes.len and bytes[idx.*] != '\n') : (idx.* += 1) {}
            continue;
        }
        break;
    }
}

fn isWhitespace(ch: u8) bool {
    return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r';
}

test "decodePpmP6 simple image" {
    const allocator = std.testing.allocator;
    const ppm = "P6\n2 1\n255\n" ++ "\xFF\x00\x00\x00\xFF\x00";

    const image = try decodeRgb(allocator, ppm);
    defer allocator.free(image.rgb);

    try std.testing.expectEqual(@as(usize, 2), image.size.width);
    try std.testing.expectEqual(@as(usize, 1), image.size.height);
    try std.testing.expectEqual(@as(usize, 6), image.rgb.len);
    try std.testing.expectEqual(@as(u8, 255), image.rgb[0]);
    try std.testing.expectEqual(@as(u8, 255), image.rgb[4]);
}
