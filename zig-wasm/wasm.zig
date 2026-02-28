const std = @import("std");

const allocator = std.heap.wasm_allocator;

extern "env" fn jsCanvasClear() void;
extern "env" fn jsCanvasDraw(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    label_ptr: [*]const u8,
    label_len: usize,
    score: f64,
) void;

const Detection = struct {
    class_id: i32,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
};

const Frame = struct {
    frame: usize,
    time: f64,
    detections: []Detection,
};

const ParsedFrame = struct {
    frame: usize,
    time: f64,
    detections: []Detection,
};

var frames: std.ArrayList(Frame) = .empty;

const coco_class_names = [_][]const u8{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

fn className(class_id: i32) []const u8 {
    if (class_id < 0) return "unknown";
    const idx: usize = @intCast(class_id);
    if (idx >= coco_class_names.len) return "unknown";
    return coco_class_names[idx];
}

fn freeFrames() void {
    for (frames.items) |f| allocator.free(f.detections);
    frames.clearAndFree(allocator);
}

export fn alloc(len: usize) ?[*]u8 {
    const slice = allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

export fn free(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

export fn clearDetections() void {
    freeFrames();
}

export fn loadDetections(ptr: [*]const u8, len: usize) bool {
    const raw = ptr[0..len];
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;

        var parsed = std.json.parseFromSlice(ParsedFrame, allocator, line, .{}) catch return false;
        defer parsed.deinit();

        // Check if we already have this time to avoid duplicates
        var exists = false;
        for (frames.items) |f| {
            if (@abs(f.time - parsed.value.time) < 0.001) {
                exists = true;
                break;
            }
        }
        if (exists) continue;

        const dets = allocator.alloc(Detection, parsed.value.detections.len) catch return false;
        @memcpy(dets, parsed.value.detections);

        frames.append(allocator, .{
            .frame = parsed.value.frame,
            .time = parsed.value.time,
            .detections = dets,
        }) catch return false;
    }

    // Keep frames sorted by time
    std.sort.pdq(Frame, frames.items, {}, struct {
        fn lessThan(_: void, a: Frame, b: Frame) bool {
            return a.time < b.time;
        }
    }.lessThan);

    return true;
}

export fn renderAt(current_time_ms: f64, duration_ms: f64, width: f64, height: f64) void {
    jsCanvasClear();
    if (frames.items.len == 0 or duration_ms <= 0 or width <= 0 or height <= 0) return;

    const current_s = current_time_ms / 1000.0;
    
    // Find the frame closest to current_s
    var best_frame: ?Frame = null;
    var min_diff: f64 = 1e18;

    for (frames.items) |f| {
        const diff = @abs(f.time - current_s);
        if (diff < min_diff) {
            min_diff = diff;
            best_frame = f;
        }
        // Since it's sorted, we can stop if it starts getting further away
        if (f.time > current_s and diff > min_diff) break;
    }

    if (best_frame) |frame| {
        // Only render if we are reasonably close (e.g. within 166ms)
        if (min_diff > 0.166) return;

        for (frame.detections) |det| {
            const x = @as(f64, det.x1) * width;
            const y = @as(f64, det.y1) * height;
            const w = (@as(f64, det.x2) - @as(f64, det.x1)) * width;
            const h = (@as(f64, det.y2) - @as(f64, det.y1)) * height;
            if (w <= 0 or h <= 0) continue;
            const label = className(det.class_id);
            jsCanvasDraw(x, y, w, h, label.ptr, label.len, det.score);
        }
    }
}

export fn main() void {}
