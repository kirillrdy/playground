const std = @import("std");
const onnxruntime = @import("onnxruntime.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2 or args.len > 3) {
        std.debug.print("usage: {s} <video.mp4> [model.onnx]\n", .{args[0]});
        return error.InvalidArguments;
    }

    const video_path = args[1];
    const model_path = if (args.len == 3) args[2] else blk: {
        const exe_dir = try std.fs.selfExeDirPathAlloc(allocator);
        defer allocator.free(exe_dir);
        break :blk try std.fmt.allocPrint(allocator, "{s}/model.onnx", .{exe_dir});
    };
    defer if (args.len != 3) allocator.free(model_path);

    var runtime = try onnxruntime.Runtime.init(allocator, model_path);
    defer runtime.deinit();

    const frame_dir = try createFrameDir(allocator);
    defer allocator.free(frame_dir);
    defer std.fs.cwd().deleteTree(frame_dir) catch {};

    try extractFrames(allocator, video_path, frame_dir);
    try processFrames(allocator, &runtime, frame_dir);
}

fn createFrameDir(allocator: std.mem.Allocator) ![]u8 {
    const ts = std.time.microTimestamp();
    const path = try std.fmt.allocPrint(allocator, ".zig-cache/video_frames_{d}", .{ts});
    try std.fs.cwd().makePath(path);
    return path;
}

fn extractFrames(allocator: std.mem.Allocator, video_path: []const u8, frame_dir: []const u8) !void {
    const frame_pattern = try std.fmt.allocPrint(allocator, "{s}/frame_%06d.ppm", .{frame_dir});
    defer allocator.free(frame_pattern);

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vsync",
            "0",
            "-f",
            "image2",
            frame_pattern,
        },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term != .Exited or result.term.Exited != 0) {
        if (result.stderr.len > 0) {
            std.log.err("ffmpeg failed: {s}", .{std.mem.trim(u8, result.stderr, "\r\n")});
        }
        return error.FrameExtractionFailed;
    }
}

fn processFrames(
    allocator: std.mem.Allocator,
    runtime: *onnxruntime.Runtime,
    frame_dir: []const u8,
) !void {
    var dir = try std.fs.cwd().openDir(frame_dir, .{ .iterate = true });
    defer dir.close();

    var frame_names = std.ArrayList([]u8).empty;
    defer {
        for (frame_names.items) |name| allocator.free(name);
        frame_names.deinit(allocator);
    }

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".ppm")) continue;
        try frame_names.append(allocator, try allocator.dupe(u8, entry.name));
    }

    std.mem.sort([]u8, frame_names.items, {}, struct {
        fn lessThan(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    if (frame_names.items.len == 0) {
        std.log.warn("no frames extracted from video", .{});
        return;
    }

    for (frame_names.items, 0..) |frame_name, index| {
        const frame_bytes = try dir.readFileAlloc(allocator, frame_name, 64 * 1024 * 1024);
        defer allocator.free(frame_bytes);

        const detections = runtime.detectFromImageBytes(allocator, frame_bytes) catch |err| {
            std.log.err("frame {d} ({s}) detect error: {}", .{ index + 1, frame_name, err });
            continue;
        };
        defer runtime.freeDetections(allocator, detections);

        std.debug.print("frame={d} detections={d}\n", .{ index + 1, detections.len });
        for (detections) |d| {
            const label = classLabel(d.class_id);
            std.debug.print(
                "  class={s}({d}) score={d:.3} box=[{d:.1},{d:.1},{d:.1},{d:.1}]\n",
                .{ label, d.class_id, d.score, d.x1, d.y1, d.x2, d.y2 },
            );
        }
    }
}

fn classLabel(class_id: usize) []const u8 {
    if (class_id < coco80_labels.len) return coco80_labels[class_id];
    return "unknown";
}

const coco80_labels = [_][]const u8{
    "person",       "bicycle",      "car",           "motorcycle",   "airplane",
    "bus",          "train",        "truck",         "boat",         "traffic light",
    "fire hydrant", "stop sign",    "parking meter", "bench",        "bird",
    "cat",          "dog",          "horse",         "sheep",        "cow",
    "elephant",     "bear",         "zebra",         "giraffe",      "backpack",
    "umbrella",     "handbag",      "tie",           "suitcase",     "frisbee",
    "skis",         "snowboard",    "sports ball",   "kite",         "baseball bat",
    "baseball glove","skateboard",  "surfboard",     "tennis racket","bottle",
    "wine glass",   "cup",          "fork",          "knife",        "spoon",
    "bowl",         "banana",       "apple",         "sandwich",     "orange",
    "broccoli",     "carrot",       "hot dog",       "pizza",        "donut",
    "cake",         "chair",        "couch",         "potted plant", "bed",
    "dining table", "toilet",       "tv",            "laptop",       "mouse",
    "remote",       "keyboard",     "cell phone",    "microwave",    "oven",
    "toaster",      "sink",         "refrigerator",  "book",         "clock",
    "vase",         "scissors",     "teddy bear",    "hair drier",   "toothbrush",
};
