const std = @import("std");
const sam3 = @import("sam3");
const zimo = @import("zimo");
const vdb = @import("vdb");

const here = zimo.bind(@This(), @embedFile("inference.zig"));

var active_model: ?*sam3.Model = null;

pub fn setActiveModel(model: ?*sam3.Model) void {
    active_model = model;
}

pub fn getActiveModel() ?*sam3.Model {
    return active_model;
}

pub fn unpackMasks(allocator: std.mem.Allocator, values: []const f32) !sam3.Masks {
    if (values.len < 4) return error.InvalidCachedQuery;
    const count: usize = @as(u32, @bitCast(values[0]));
    const width: usize = @as(u32, @bitCast(values[1]));
    const height: usize = @as(u32, @bitCast(values[2]));
    if (width == 0 or height == 0) return error.InvalidCachedQuery;
    const pixels = std.math.mul(usize, width, height) catch return error.InvalidCachedQuery;
    const logits_len = std.math.mul(usize, count, pixels) catch return error.InvalidCachedQuery;
    const header_and_scores = std.math.add(usize, 4, count) catch return error.InvalidCachedQuery;
    const expected = std.math.add(usize, header_and_scores, logits_len) catch return error.InvalidCachedQuery;
    if (values.len != expected) return error.InvalidCachedQuery;

    const scores = try allocator.dupe(f32, values[4..header_and_scores]);
    errdefer allocator.free(scores);
    const logits = try allocator.dupe(f32, values[header_and_scores..]);
    return .{
        .allocator = allocator,
        .scores = scores,
        .logits = logits,
        .count = count,
        .width = width,
        .height = height,
        .object_score = values[3],
    };
}

pub fn computeQuery(
    allocator: std.mem.Allocator,
    model_ids: [6][]const u8,
    image: sam3.RgbImage,
    phrase: []const u8,
    min_score: f32,
) ![]f32 {
    const model = active_model orelse return error.NoActiveApp;
    const text_features = try here.call(.computeTextFeatures, .{
        allocator,
        [_][]const u8{
            model_ids[1],
            model_ids[5],
            model_ids[3],
        },
        phrase,
    });
    defer allocator.free(text_features);
    var embedding = try model.encodeForText(image);
    defer embedding.deinit();
    var masks = try model.findWithTextFeatures(&embedding, phrase, text_features, .{ .min_score = min_score });
    defer masks.deinit();

    const header_and_scores = try std.math.add(usize, 4, masks.scores.len);
    const len = try std.math.add(usize, header_and_scores, masks.logits.len);
    const values = try allocator.alloc(f32, len);
    values[0] = @bitCast(@as(u32, @intCast(masks.count)));
    values[1] = @bitCast(@as(u32, @intCast(masks.width)));
    values[2] = @bitCast(@as(u32, @intCast(masks.height)));
    values[3] = masks.object_score;
    @memcpy(values[4..][0..masks.scores.len], masks.scores);
    @memcpy(values[header_and_scores..], masks.logits);
    return values;
}

pub fn queryArgs(allocator: std.mem.Allocator, image: sam3.RgbImage, phrase: []const u8) std.meta.ArgsTuple(@TypeOf(computeQuery)) {
    return .{
        allocator,
        [_][]const u8{
            sam3.assets.concept_vision_encoder.sha256,
            sam3.assets.concept_text_encoder.sha256,
            sam3.assets.concept_decoder.sha256,
            sam3.assets.concept_tokenizer_json.sha256,
            sam3.assets.concept_vision_encoder_data.sha256,
            sam3.assets.concept_text_encoder_data.sha256,
        },
        image,
        phrase,
        @as(f32, 0.5),
    };
}

pub fn cachedQuery(allocator: std.mem.Allocator, image: sam3.RgbImage, phrase: []const u8) ![]f32 {
    return here.call(.computeQuery, queryArgs(allocator, image, phrase));
}

pub fn computeTextFeatures(
    allocator: std.mem.Allocator,
    model_ids: [3][]const u8,
    phrase: []const u8,
) ![]f32 {
    _ = allocator;
    _ = model_ids;
    const model = active_model orelse return error.NoActiveApp;
    return model.encodeTextFeatures(phrase);
}

pub fn evaluateSam3Frame(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_mutex: *std.Io.Mutex,
    frame: vdb.types.FrameRef,
    prompt: []const u8,
) !vdb.types.MaskRef {
    if (frame.rgb == null or frame.width == 0 or frame.height == 0) {
        return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
    }
    const pixel_count = std.math.mul(usize, frame.width, frame.height) catch return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
    const bytes_len = std.math.mul(usize, pixel_count, 3) catch return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
    const rgb_img = sam3.RgbImage{
        .pixels = frame.rgb.?[0..bytes_len],
        .width = frame.width,
        .height = frame.height,
    };
    try model_mutex.lock(io);
    defer model_mutex.unlock(io);
    const maybe_vals = cachedQuery(allocator, rgb_img, prompt) catch |err| return err;
    defer allocator.free(maybe_vals);
    if (maybe_vals.len >= 4) {
        const count: usize = @as(u32, @bitCast(maybe_vals[0]));
        if (count > 0) {
            const obj_score = maybe_vals[3];
            const best_score = if (maybe_vals.len >= 5) @max(obj_score, maybe_vals[4]) else obj_score;
            return .{
                .score = best_score,
                .coverage = 0.2,
                .width = frame.width,
                .height = frame.height,
            };
        }
    }
    return .{ .score = 0, .coverage = 0, .width = frame.width, .height = frame.height };
}
