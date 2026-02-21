const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Detection = struct {
    class_id: usize,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
};

pub const OutputLayout = enum {
    boxes_first, // [1, boxes, attributes]
    attributes_first, // [1, attributes, boxes]
};

pub fn decodeV8(
    allocator: Allocator,
    tensor: []const f32,
    boxes: usize,
    classes: usize,
    layout: OutputLayout,
    conf_threshold: f32,
) ![]Detection {
    var out = try std.ArrayList(Detection).initCapacity(allocator, 0);
    errdefer out.deinit(allocator);

    const attrs = classes + 4;
    const expected = boxes * attrs;
    if (tensor.len < expected) return error.InvalidTensorSize;

    for (0..boxes) |box_idx| {
        const cx = getValue(tensor, boxes, attrs, box_idx, 0, layout);
        const cy = getValue(tensor, boxes, attrs, box_idx, 1, layout);
        const w = getValue(tensor, boxes, attrs, box_idx, 2, layout);
        const h = getValue(tensor, boxes, attrs, box_idx, 3, layout);

        var best_class: usize = 0;
        var best_score: f32 = 0.0;
        for (0..classes) |class_idx| {
            const score = getValue(tensor, boxes, attrs, box_idx, class_idx + 4, layout);
            if (score > best_score) {
                best_score = score;
                best_class = class_idx;
            }
        }

        if (best_score < conf_threshold) continue;

        try out.append(allocator, .{
            .class_id = best_class,
            .score = best_score,
            .x1 = cx - (w / 2.0),
            .y1 = cy - (h / 2.0),
            .x2 = cx + (w / 2.0),
            .y2 = cy + (h / 2.0),
        });
    }

    return out.toOwnedSlice(allocator);
}

pub fn nms(
    allocator: Allocator,
    detections: []const Detection,
    iou_threshold: f32,
) ![]Detection {
    if (detections.len == 0) return allocator.alloc(Detection, 0);

    const sorted = try allocator.dupe(Detection, detections);
    defer allocator.free(sorted);

    std.mem.sort(Detection, sorted, {}, struct {
        fn lessThan(_: void, a: Detection, b: Detection) bool {
            return a.score > b.score;
        }
    }.lessThan);

    var keep = try std.ArrayList(Detection).initCapacity(allocator, 0);
    errdefer keep.deinit(allocator);

    for (sorted) |candidate| {
        var suppressed = false;
        for (keep.items) |selected| {
            if (candidate.class_id != selected.class_id) continue;
            if (iou(candidate, selected) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) try keep.append(allocator, candidate);
    }

    return keep.toOwnedSlice(allocator);
}

fn getValue(
    tensor: []const f32,
    boxes: usize,
    attrs: usize,
    box_idx: usize,
    attr_idx: usize,
    layout: OutputLayout,
) f32 {
    return switch (layout) {
        .boxes_first => tensor[(box_idx * attrs) + attr_idx],
        .attributes_first => tensor[(attr_idx * boxes) + box_idx],
    };
}

fn area(d: Detection) f32 {
    const w = @max(0.0, d.x2 - d.x1);
    const h = @max(0.0, d.y2 - d.y1);
    return w * h;
}

fn iou(a: Detection, b: Detection) f32 {
    const ix1 = @max(a.x1, b.x1);
    const iy1 = @max(a.y1, b.y1);
    const ix2 = @min(a.x2, b.x2);
    const iy2 = @min(a.y2, b.y2);
    const inter_w = @max(0.0, ix2 - ix1);
    const inter_h = @max(0.0, iy2 - iy1);
    const inter_area = inter_w * inter_h;
    const union_area = area(a) + area(b) - inter_area;
    if (union_area <= 0.0) return 0.0;
    return inter_area / union_area;
}

test "decodeV8 boxes_first" {
    const allocator = std.testing.allocator;
    // two boxes, two classes
    // box0: cx=10 cy=10 w=4 h=4 class1=0.8
    // box1: cx=20 cy=20 w=2 h=2 class0=0.9
    const tensor = [_]f32{
        10, 10, 4, 4, 0.2, 0.8,
        20, 20, 2, 2, 0.9, 0.1,
    };
    const decoded = try decodeV8(allocator, &tensor, 2, 2, .boxes_first, 0.5);
    defer allocator.free(decoded);

    try std.testing.expectEqual(@as(usize, 2), decoded.len);
    try std.testing.expectEqual(@as(usize, 1), decoded[0].class_id);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), decoded[0].x1, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), decoded[0].x2, 0.001);
}

test "nms per class" {
    const allocator = std.testing.allocator;
    const detections = [_]Detection{
        .{ .class_id = 0, .score = 0.9, .x1 = 0, .y1 = 0, .x2 = 10, .y2 = 10 },
        .{ .class_id = 0, .score = 0.8, .x1 = 1, .y1 = 1, .x2 = 11, .y2 = 11 },
        .{ .class_id = 1, .score = 0.7, .x1 = 1, .y1 = 1, .x2 = 11, .y2 = 11 },
    };
    const filtered = try nms(allocator, &detections, 0.5);
    defer allocator.free(filtered);

    try std.testing.expectEqual(@as(usize, 2), filtered.len);
    try std.testing.expectEqual(@as(usize, 0), filtered[0].class_id);
    try std.testing.expectEqual(@as(usize, 1), filtered[1].class_id);
}
