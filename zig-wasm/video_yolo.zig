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

    var runtime = try onnxruntime.Runtime.initWithOptions(allocator, model_path, .{
        .use_cuda = true,
        .cuda_device_id = 0,
        .require_cuda = true,
    });
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

    const src_width_i32: c_int = codec_ctx.?.*.width;
    const src_height_i32: c_int = codec_ctx.?.*.height;
    const infer_width: usize = 640;
    const infer_height: usize = 640;
    const infer_width_i32: c_int = @intCast(infer_width);
    const infer_height_i32: c_int = @intCast(infer_height);
    const rgb_len = infer_width * infer_height * 3;
    const rgb = try allocator.alloc(u8, rgb_len);
    defer allocator.free(rgb);

    var dst_data: [4][*c]u8 = .{ null, null, null, null };
    var dst_linesize: [4]c_int = .{ 0, 0, 0, 0 };
    _ = try checkAv(c.av_image_fill_arrays(
        &dst_data,
        &dst_linesize,
        rgb.ptr,
        c.AV_PIX_FMT_RGB24,
        infer_width_i32,
        infer_height_i32,
        1,
    ));

    const sws_ctx = c.sws_getContext(
        src_width_i32,
        src_height_i32,
        codec_ctx.?.*.pix_fmt,
        infer_width_i32,
        infer_height_i32,
        c.AV_PIX_FMT_RGB24,
        c.SWS_FAST_BILINEAR,
        null,
        null,
        null,
    );
    if (sws_ctx == null) return error.FfmpegScaleInitFailed;
    defer c.sws_freeContext(sws_ctx);

    var frame_index: usize = 0;
    while (c.av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet.?.*.stream_index == stream_idx) {
            try decodePacket(allocator, runtime, codec_ctx.?, frame.?, packet.?, sws_ctx, &dst_data, &dst_linesize, rgb, infer_width, infer_height, &frame_index);
        }
        c.av_packet_unref(packet);
    }

    try decodePacket(allocator, runtime, codec_ctx.?, frame.?, null, sws_ctx, &dst_data, &dst_linesize, rgb, infer_width, infer_height, &frame_index);
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
    const send_ret = c.avcodec_send_packet(codec_ctx, packet);
    if (send_ret == c.AVERROR_EOF) return;
    _ = try checkAv(send_ret);
    while (true) {
        const decode_start_ns = std.time.nanoTimestamp();
        const ret = c.avcodec_receive_frame(codec_ctx, frame);
        const decode_end_ns = std.time.nanoTimestamp();
        if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) return;
        _ = try checkAv(ret);

        const scale_start_ns = std.time.nanoTimestamp();
        _ = c.sws_scale(
            sws_ctx,
            @ptrCast(&frame.data),
            @ptrCast(&frame.linesize),
            0,
            @intCast(codec_ctx.height),
            dst_data,
            dst_linesize,
        );
        const scale_end_ns = std.time.nanoTimestamp();

        frame_index.* += 1;
        var detect_profile: onnxruntime.DetectProfile = .{};
        const infer_total_start_ns = std.time.nanoTimestamp();
        const detections = runtime.detectFromRgbWithProfile(allocator, rgb, .{
            .width = width,
            .height = height,
        }, &detect_profile) catch |err| {
            std.log.err("frame {d} detect error: {}", .{ frame_index.*, err });
            continue;
        };
        const infer_total_end_ns = std.time.nanoTimestamp();
        defer runtime.freeDetections(allocator, detections);

        const decode_ms = @as(f64, @floatFromInt(decode_end_ns - decode_start_ns)) / @as(f64, std.time.ns_per_ms);
        const scale_ms = @as(f64, @floatFromInt(scale_end_ns - scale_start_ns)) / @as(f64, std.time.ns_per_ms);
        const preprocess_ms = @as(f64, @floatFromInt(detect_profile.preprocess_ns)) / @as(f64, std.time.ns_per_ms);
        const run_ms = @as(f64, @floatFromInt(detect_profile.run_ns)) / @as(f64, std.time.ns_per_ms);
        const post_ms = @as(f64, @floatFromInt(detect_profile.decode_ns)) / @as(f64, std.time.ns_per_ms);
        const infer_total_ms = @as(f64, @floatFromInt(infer_total_end_ns - infer_total_start_ns)) / @as(f64, std.time.ns_per_ms);

        std.debug.print(
            "frame={d} decode_ms={d:.3} scale_ms={d:.3} preprocess_ms={d:.3} ort_run_ms={d:.3} post_ms={d:.3} infer_total_ms={d:.3} dets={d}\n",
            .{
                frame_index.*,
                decode_ms,
                scale_ms,
                preprocess_ms,
                run_ms,
                post_ms,
                infer_total_ms,
                detections.len,
            },
        );
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
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",
    "bus",            "train",      "truck",         "boat",          "traffic light",
    "fire hydrant",   "stop sign",  "parking meter", "bench",         "bird",
    "cat",            "dog",        "horse",         "sheep",         "cow",
    "elephant",       "bear",       "zebra",         "giraffe",       "backpack",
    "umbrella",       "handbag",    "tie",           "suitcase",      "frisbee",
    "skis",           "snowboard",  "sports ball",   "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",
    "wine glass",     "cup",        "fork",          "knife",         "spoon",
    "bowl",           "banana",     "apple",         "sandwich",      "orange",
    "broccoli",       "carrot",     "hot dog",       "pizza",         "donut",
    "cake",           "chair",      "couch",         "potted plant",  "bed",
    "dining table",   "toilet",     "tv",            "laptop",        "mouse",
    "remote",         "keyboard",   "cell phone",    "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",
    "vase",           "scissors",   "teddy bear",    "hair drier",    "toothbrush",
};
