const std = @import("std");
const onnxruntime = @import("onnxruntime.zig");

const c = @cImport({
    @cInclude("libavformat/avformat.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavutil/avutil.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libswscale/swscale.h");
});

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

    try processVideo(allocator, &runtime, video_path);
}

fn processVideo(allocator: std.mem.Allocator, runtime: *onnxruntime.Runtime, video_path: []const u8) !void {
    const video_path_z = try allocator.dupeZ(u8, video_path);
    defer allocator.free(video_path_z);

    var fmt_ctx: ?*c.AVFormatContext = null;
    _ = try checkAv(c.avformat_open_input(&fmt_ctx, video_path_z.ptr, null, null));
    defer c.avformat_close_input(&fmt_ctx);

    _ = try checkAv(c.avformat_find_stream_info(fmt_ctx, null));

    const stream_idx = c.av_find_best_stream(
        fmt_ctx,
        c.AVMEDIA_TYPE_VIDEO,
        -1,
        -1,
        null,
        0,
    );
    if (stream_idx < 0) return error.NoVideoStream;

    const stream = fmt_ctx.?.*.streams[@intCast(stream_idx)];
    const codecpar = stream.?.*.codecpar;
    const decoder = c.avcodec_find_decoder(codecpar.?.*.codec_id);
    if (decoder == null) return error.UnsupportedCodec;

    var codec_ctx = c.avcodec_alloc_context3(decoder);
    if (codec_ctx == null) return error.OutOfMemory;
    defer c.avcodec_free_context(&codec_ctx);

    _ = try checkAv(c.avcodec_parameters_to_context(codec_ctx, codecpar));
    _ = try checkAv(c.avcodec_open2(codec_ctx, decoder, null));

    var frame = c.av_frame_alloc();
    if (frame == null) return error.OutOfMemory;
    defer c.av_frame_free(&frame);

    var packet = c.av_packet_alloc();
    if (packet == null) return error.OutOfMemory;
    defer c.av_packet_free(&packet);

    const width: usize = @intCast(codec_ctx.?.*.width);
    const height: usize = @intCast(codec_ctx.?.*.height);
    const width_i32: c_int = @intCast(width);
    const height_i32: c_int = @intCast(height);
    const rgb_len = width * height * 3;
    const rgb = try allocator.alloc(u8, rgb_len);
    defer allocator.free(rgb);

    var dst_data: [4][*c]u8 = .{ null, null, null, null };
    var dst_linesize: [4]c_int = .{ 0, 0, 0, 0 };
    _ = try checkAv(c.av_image_fill_arrays(
        &dst_data,
        &dst_linesize,
        rgb.ptr,
        c.AV_PIX_FMT_RGB24,
        width_i32,
        height_i32,
        1,
    ));

    const sws_ctx = c.sws_getContext(
        width_i32,
        height_i32,
        codec_ctx.?.*.pix_fmt,
        width_i32,
        height_i32,
        c.AV_PIX_FMT_RGB24,
        c.SWS_BILINEAR,
        null,
        null,
        null,
    );
    if (sws_ctx == null) return error.FfmpegScaleInitFailed;
    defer c.sws_freeContext(sws_ctx);

    var frame_index: usize = 0;
    while (c.av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet.?.*.stream_index == stream_idx) {
            try decodePacket(allocator, runtime, codec_ctx.?, frame.?, packet.?, sws_ctx, &dst_data, &dst_linesize, rgb, width, height, &frame_index);
        }
        c.av_packet_unref(packet);
    }

    _ = try checkAv(c.avcodec_send_packet(codec_ctx, null));
    try decodePacket(allocator, runtime, codec_ctx.?, frame.?, null, sws_ctx, &dst_data, &dst_linesize, rgb, width, height, &frame_index);
}

fn decodePacket(
    allocator: std.mem.Allocator,
    runtime: *onnxruntime.Runtime,
    codec_ctx: *c.AVCodecContext,
    frame: *c.AVFrame,
    packet: ?*c.AVPacket,
    sws_ctx: *c.SwsContext,
    dst_data: *[4][*c]u8,
    dst_linesize: *[4]c_int,
    rgb: []u8,
    width: usize,
    height: usize,
    frame_index: *usize,
) !void {
    _ = try checkAv(c.avcodec_send_packet(codec_ctx, packet));
    while (true) {
        const ret = c.avcodec_receive_frame(codec_ctx, frame);
        if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) return;
        _ = try checkAv(ret);

        _ = c.sws_scale(
            sws_ctx,
            @ptrCast(&frame.data),
            @ptrCast(&frame.linesize),
            0,
            @intCast(codec_ctx.height),
            dst_data,
            dst_linesize,
        );

        frame_index.* += 1;
        const detections = runtime.detectFromRgb(allocator, rgb, .{
            .width = width,
            .height = height,
        }) catch |err| {
            std.log.err("frame {d} detect error: {}", .{ frame_index.*, err });
            continue;
        };
        defer runtime.freeDetections(allocator, detections);

        std.debug.print("frame={d} detections={d}\n", .{ frame_index.*, detections.len });
        for (detections) |d| {
            const label = classLabel(d.class_id);
            std.debug.print(
                "  class={s}({d}) score={d:.3} box=[{d:.1},{d:.1},{d:.1},{d:.1}]\n",
                .{ label, d.class_id, d.score, d.x1, d.y1, d.x2, d.y2 },
            );
        }
    }
}

fn checkAv(ret: c_int) !c_int {
    if (ret >= 0) return ret;
    var err_buf: [256]u8 = undefined;
    _ = c.av_strerror(ret, &err_buf, err_buf.len);
    std.log.err("ffmpeg error ({d}): {s}", .{ ret, std.mem.sliceTo(&err_buf, 0) });
    return error.FfmpegError;
}

fn classLabel(class_id: usize) []const u8 {
    if (class_id < coco80_labels.len) return coco80_labels[class_id];
    return "unknown";
}

const coco80_labels = [_][]const u8{
    "person",        "bicycle",      "car",          "motorcycle",    "airplane",
    "bus",           "train",        "truck",        "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter","bench",         "bird",
    "cat",           "dog",          "horse",        "sheep",         "cow",
    "elephant",      "bear",         "zebra",        "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",          "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",  "kite",          "baseball bat",
    "baseball glove","skateboard",   "surfboard",    "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",         "knife",         "spoon",
    "bowl",          "banana",       "apple",        "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",      "pizza",         "donut",
    "cake",          "chair",        "couch",        "potted plant",  "bed",
    "dining table",  "toilet",       "tv",           "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",   "microwave",     "oven",
    "toaster",       "sink",         "refrigerator", "book",          "clock",
    "vase",          "scissors",     "teddy bear",   "hair drier",    "toothbrush",
};
