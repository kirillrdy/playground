const std = @import("std");
const onnxruntime = @import("onnxruntime.zig");

const c = @cImport({
    @cInclude("libavformat/avformat.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavutil/avutil.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libswscale/swscale.h");
});

const DecodeStats = struct {
    read_packet_count: usize = 0,
    read_packet_ns_sum: i128 = 0,
    read_packet_ns_min: i128 = std.math.maxInt(i128),
    read_packet_ns_max: i128 = 0,
    send_packet_count: usize = 0,
    send_packet_ns_sum: i128 = 0,
    send_packet_ns_min: i128 = std.math.maxInt(i128),
    send_packet_ns_max: i128 = 0,
    frame_count: usize = 0,
    decode_ns_sum: i128 = 0,
    decode_ns_min: i128 = std.math.maxInt(i128),
    decode_ns_max: i128 = 0,
    scale_ns_sum: i128 = 0,
    scale_ns_min: i128 = std.math.maxInt(i128),
    scale_ns_max: i128 = 0,
};

const InferStats = struct {
    frame_count: usize = 0,
    detection_count: usize = 0,
    preprocess_ns_sum: i128 = 0,
    preprocess_ns_min: i128 = std.math.maxInt(i128),
    preprocess_ns_max: i128 = 0,
    run_ns_sum: i128 = 0,
    run_ns_min: i128 = std.math.maxInt(i128),
    run_ns_max: i128 = 0,
    post_ns_sum: i128 = 0,
    post_ns_min: i128 = std.math.maxInt(i128),
    post_ns_max: i128 = 0,
    infer_total_ns_sum: i128 = 0,
    infer_total_ns_min: i128 = std.math.maxInt(i128),
    infer_total_ns_max: i128 = 0,
};

const FrameQueue = struct {
    allocator: std.mem.Allocator,
    rgb_len: usize,
    capacity: usize,
    storage: []u8,
    free_ring: []usize,
    ready_ring: []usize,
    free_head: usize = 0,
    free_len: usize = 0,
    ready_head: usize = 0,
    ready_len: usize = 0,
    closed: bool = false,
    mutex: std.Thread.Mutex = .{},
    not_empty: std.Thread.Condition = .{},
    not_full: std.Thread.Condition = .{},

    fn init(allocator: std.mem.Allocator, capacity: usize, rgb_len: usize) !FrameQueue {
        const storage = try allocator.alloc(u8, capacity * rgb_len);
        errdefer allocator.free(storage);
        const free_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(free_ring);
        const ready_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(ready_ring);

        for (0..capacity) |i| free_ring[i] = i;
        return .{
            .allocator = allocator,
            .rgb_len = rgb_len,
            .capacity = capacity,
            .storage = storage,
            .free_ring = free_ring,
            .ready_ring = ready_ring,
            .free_len = capacity,
        };
    }

    fn deinit(self: *FrameQueue) void {
        self.allocator.free(self.ready_ring);
        self.allocator.free(self.free_ring);
        self.allocator.free(self.storage);
    }

    fn slot(self: *FrameQueue, idx: usize) []u8 {
        const start = idx * self.rgb_len;
        return self.storage[start .. start + self.rgb_len];
    }

    fn acquireWriteSlot(self: *FrameQueue) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.free_len == 0) self.not_full.wait(&self.mutex);
        const idx = self.free_ring[self.free_head];
        self.free_head = (self.free_head + 1) % self.capacity;
        self.free_len -= 1;
        return idx;
    }

    fn publishReady(self: *FrameQueue, idx: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const tail = (self.ready_head + self.ready_len) % self.capacity;
        self.ready_ring[tail] = idx;
        self.ready_len += 1;
        self.not_empty.signal();
    }

    fn acquireReadSlot(self: *FrameQueue) ?usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.ready_len == 0 and !self.closed) self.not_empty.wait(&self.mutex);
        if (self.ready_len == 0 and self.closed) return null;

        const idx = self.ready_ring[self.ready_head];
        self.ready_head = (self.ready_head + 1) % self.capacity;
        self.ready_len -= 1;
        return idx;
    }

    fn releaseWriteSlot(self: *FrameQueue, idx: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const tail = (self.free_head + self.free_len) % self.capacity;
        self.free_ring[tail] = idx;
        self.free_len += 1;
        self.not_full.signal();
    }

    fn close(self: *FrameQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.closed = true;
        self.not_empty.broadcast();
    }
};

const InferWorkerCtx = struct {
    queue: *FrameQueue,
    runtime: *onnxruntime.Runtime,
    infer_stats: *InferStats,
    width: usize,
    height: usize,
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("usage: {s} <video.mp4> [model.onnx] [--skip-infer]\n", .{args[0]});
        return error.InvalidArguments;
    }

    const video_path = args[1];
    var skip_infer = false;
    var explicit_model_path: ?[]const u8 = null;
    for (args[2..]) |arg| {
        if (std.mem.eql(u8, arg, "--skip-infer")) {
            skip_infer = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown option: {s}\n", .{arg});
            return error.InvalidArguments;
        }
        if (explicit_model_path != null) {
            std.debug.print("multiple model paths provided\n", .{});
            return error.InvalidArguments;
        }
        explicit_model_path = arg;
    }

    const model_path = if (explicit_model_path) |path| path else blk: {
        const exe_dir = try std.fs.selfExeDirPathAlloc(allocator);
        defer allocator.free(exe_dir);
        break :blk try std.fmt.allocPrint(allocator, "{s}/model.onnx", .{exe_dir});
    };
    defer if (explicit_model_path == null) allocator.free(model_path);

    var runtime: ?onnxruntime.Runtime = null;
    if (!skip_infer) {
        runtime = try onnxruntime.Runtime.initWithOptions(allocator, model_path, .{
            .use_cuda = true,
            .cuda_device_id = 0,
            .require_cuda = true,
        });
    }
    defer if (runtime) |*r| r.deinit();

    try processVideo(allocator, if (runtime) |*r| r else null, video_path, skip_infer);
}

fn processVideo(allocator: std.mem.Allocator, runtime: ?*onnxruntime.Runtime, video_path: []const u8, skip_infer: bool) !void {
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
    const cpu_count = std.Thread.getCpuCount() catch 0;
    if (cpu_count > 0) codec_ctx.?.*.thread_count = @intCast(@min(cpu_count, 16));
    codec_ctx.?.*.thread_type = c.FF_THREAD_FRAME | c.FF_THREAD_SLICE;
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

    const wall_start_ns = std.time.nanoTimestamp();
    var decode_stats: DecodeStats = .{};
    var infer_stats: InferStats = .{};
    var queue = try FrameQueue.init(allocator, 8, rgb_len);
    defer queue.deinit();
    var infer_thread: ?std.Thread = null;
    var infer_ctx: InferWorkerCtx = undefined;
    if (!skip_infer) {
        const runtime_ptr = runtime orelse return error.InvalidArguments;
        infer_ctx = .{
            .queue = &queue,
            .runtime = runtime_ptr,
            .infer_stats = &infer_stats,
            .width = infer_width,
            .height = infer_height,
        };
        infer_thread = try std.Thread.spawn(.{}, inferWorkerMain, .{&infer_ctx});
    }

    while (true) {
        const read_start_ns = std.time.nanoTimestamp();
        const read_ret = c.av_read_frame(fmt_ctx, packet);
        const read_end_ns = std.time.nanoTimestamp();
        if (read_ret < 0) break;
        recordReadPacketSample(&decode_stats, read_end_ns - read_start_ns);
        if (packet.?.*.stream_index == stream_idx) {
            try decodePacket(skip_infer, codec_ctx.?, frame.?, packet.?, sws_ctx, &dst_data, &dst_linesize, rgb, &queue, &decode_stats);
        }
        c.av_packet_unref(packet);
    }

    try decodePacket(skip_infer, codec_ctx.?, frame.?, null, sws_ctx, &dst_data, &dst_linesize, rgb, &queue, &decode_stats);
    if (!skip_infer) {
        queue.close();
        if (infer_thread) |thread| thread.join();
    }

    const wall_end_ns = std.time.nanoTimestamp();
    printSummary(decode_stats, infer_stats, "final", wall_end_ns - wall_start_ns);
}

fn decodePacket(
    skip_infer: bool,
    codec_ctx: *c.AVCodecContext,
    frame: *c.AVFrame,
    packet: ?*c.AVPacket,
    sws_ctx: *c.SwsContext,
    dst_data: *[4][*c]u8,
    dst_linesize: *[4]c_int,
    rgb: []u8,
    queue: *FrameQueue,
    decode_stats: *DecodeStats,
) !void {
    const send_start_ns = std.time.nanoTimestamp();
    const send_ret = c.avcodec_send_packet(codec_ctx, packet);
    const send_end_ns = std.time.nanoTimestamp();
    recordSendPacketSample(decode_stats, send_end_ns - send_start_ns);
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

        const decode_ns = decode_end_ns - decode_start_ns;
        const scale_ns = scale_end_ns - scale_start_ns;
        recordDecodeFrameSample(decode_stats, decode_ns, scale_ns);
        if (!skip_infer) {
            const idx = queue.acquireWriteSlot();
            @memcpy(queue.slot(idx), rgb);
            queue.publishReady(idx);
        }
    }
}

fn inferWorkerMain(ctx: *InferWorkerCtx) void {
    const allocator = std.heap.c_allocator;
    while (true) {
        const idx = ctx.queue.acquireReadSlot() orelse break;
        const rgb = ctx.queue.slot(idx);
        var detect_profile: onnxruntime.DetectProfile = .{};
        const infer_start_ns = std.time.nanoTimestamp();
        const detections = ctx.runtime.detectFromRgbWithProfile(allocator, rgb, .{
            .width = ctx.width,
            .height = ctx.height,
        }, &detect_profile) catch |err| {
            std.log.err("infer worker detect error: {}", .{err});
            ctx.queue.releaseWriteSlot(idx);
            continue;
        };
        const infer_end_ns = std.time.nanoTimestamp();
        defer ctx.runtime.freeDetections(allocator, detections);

        recordInferSample(
            ctx.infer_stats,
            detect_profile.preprocess_ns,
            detect_profile.run_ns,
            detect_profile.decode_ns,
            infer_end_ns - infer_start_ns,
            detections.len,
        );
        ctx.queue.releaseWriteSlot(idx);
    }
}

fn recordReadPacketSample(stats: *DecodeStats, read_ns: i128) void {
    stats.read_packet_count += 1;
    stats.read_packet_ns_sum += read_ns;
    stats.read_packet_ns_min = @min(stats.read_packet_ns_min, read_ns);
    stats.read_packet_ns_max = @max(stats.read_packet_ns_max, read_ns);
}

fn recordSendPacketSample(stats: *DecodeStats, send_ns: i128) void {
    stats.send_packet_count += 1;
    stats.send_packet_ns_sum += send_ns;
    stats.send_packet_ns_min = @min(stats.send_packet_ns_min, send_ns);
    stats.send_packet_ns_max = @max(stats.send_packet_ns_max, send_ns);
}

fn recordDecodeFrameSample(stats: *DecodeStats, decode_ns: i128, scale_ns: i128) void {
    stats.frame_count += 1;
    stats.decode_ns_sum += decode_ns;
    stats.decode_ns_min = @min(stats.decode_ns_min, decode_ns);
    stats.decode_ns_max = @max(stats.decode_ns_max, decode_ns);
    stats.scale_ns_sum += scale_ns;
    stats.scale_ns_min = @min(stats.scale_ns_min, scale_ns);
    stats.scale_ns_max = @max(stats.scale_ns_max, scale_ns);
}

fn recordInferSample(
    stats: *InferStats,
    preprocess_ns: i128,
    run_ns: i128,
    post_ns: i128,
    infer_total_ns: i128,
    dets: usize,
) void {
    stats.frame_count += 1;
    stats.detection_count += dets;
    stats.preprocess_ns_sum += preprocess_ns;
    stats.preprocess_ns_min = @min(stats.preprocess_ns_min, preprocess_ns);
    stats.preprocess_ns_max = @max(stats.preprocess_ns_max, preprocess_ns);
    stats.run_ns_sum += run_ns;
    stats.run_ns_min = @min(stats.run_ns_min, run_ns);
    stats.run_ns_max = @max(stats.run_ns_max, run_ns);
    stats.post_ns_sum += post_ns;
    stats.post_ns_min = @min(stats.post_ns_min, post_ns);
    stats.post_ns_max = @max(stats.post_ns_max, post_ns);
    stats.infer_total_ns_sum += infer_total_ns;
    stats.infer_total_ns_min = @min(stats.infer_total_ns_min, infer_total_ns);
    stats.infer_total_ns_max = @max(stats.infer_total_ns_max, infer_total_ns);
}

fn printSummary(decode: DecodeStats, infer: InferStats, label: []const u8, elapsed_ns: i128) void {
    if (decode.frame_count == 0) return;
    const frames_f: f64 = @floatFromInt(decode.frame_count);
    const ns_to_ms = @as(f64, std.time.ns_per_ms);
    const elapsed_ms = if (elapsed_ns > 0) @as(f64, @floatFromInt(elapsed_ns)) / ns_to_ms else 0.0;
    const fps = if (elapsed_ms > 0.0) frames_f / (elapsed_ms / 1000.0) else 0.0;
    const read_count_f = @as(f64, @floatFromInt(decode.read_packet_count));
    const send_count_f = @as(f64, @floatFromInt(decode.send_packet_count));
    const infer_frames_f = if (infer.frame_count > 0) @as(f64, @floatFromInt(infer.frame_count)) else 1.0;
    const read_avg_ms = if (decode.read_packet_count > 0) @as(f64, @floatFromInt(decode.read_packet_ns_sum)) / read_count_f / ns_to_ms else 0.0;
    const send_avg_ms = if (decode.send_packet_count > 0) @as(f64, @floatFromInt(decode.send_packet_ns_sum)) / send_count_f / ns_to_ms else 0.0;
    const read_min_ms = if (decode.read_packet_count > 0) @as(f64, @floatFromInt(decode.read_packet_ns_min)) / ns_to_ms else 0.0;
    const read_max_ms = if (decode.read_packet_count > 0) @as(f64, @floatFromInt(decode.read_packet_ns_max)) / ns_to_ms else 0.0;
    const send_min_ms = if (decode.send_packet_count > 0) @as(f64, @floatFromInt(decode.send_packet_ns_min)) / ns_to_ms else 0.0;
    const send_max_ms = if (decode.send_packet_count > 0) @as(f64, @floatFromInt(decode.send_packet_ns_max)) / ns_to_ms else 0.0;
    const avg_dets = if (infer.frame_count > 0) @as(f64, @floatFromInt(infer.detection_count)) / infer_frames_f else 0.0;

    std.debug.print(
        "[{s}] frames={d} fps={d:.2} avg_dets={d:.3} read_packet_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} send_packet_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} decode_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} scale_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} preprocess_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} ort_run_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} post_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3} infer_total_ms(avg/min/max)={d:.3}/{d:.3}/{d:.3}\n",
        .{
            label,
            decode.frame_count,
            fps,
            avg_dets,
            read_avg_ms,
            read_min_ms,
            read_max_ms,
            send_avg_ms,
            send_min_ms,
            send_max_ms,
            @as(f64, @floatFromInt(decode.decode_ns_sum)) / frames_f / ns_to_ms,
            @as(f64, @floatFromInt(decode.decode_ns_min)) / ns_to_ms,
            @as(f64, @floatFromInt(decode.decode_ns_max)) / ns_to_ms,
            @as(f64, @floatFromInt(decode.scale_ns_sum)) / frames_f / ns_to_ms,
            @as(f64, @floatFromInt(decode.scale_ns_min)) / ns_to_ms,
            @as(f64, @floatFromInt(decode.scale_ns_max)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.preprocess_ns_sum)) / infer_frames_f / ns_to_ms,
            @as(f64, @floatFromInt(if (infer.frame_count > 0) infer.preprocess_ns_min else 0)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.preprocess_ns_max)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.run_ns_sum)) / infer_frames_f / ns_to_ms,
            @as(f64, @floatFromInt(if (infer.frame_count > 0) infer.run_ns_min else 0)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.run_ns_max)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.post_ns_sum)) / infer_frames_f / ns_to_ms,
            @as(f64, @floatFromInt(if (infer.frame_count > 0) infer.post_ns_min else 0)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.post_ns_max)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.infer_total_ns_sum)) / infer_frames_f / ns_to_ms,
            @as(f64, @floatFromInt(if (infer.frame_count > 0) infer.infer_total_ns_min else 0)) / ns_to_ms,
            @as(f64, @floatFromInt(infer.infer_total_ns_max)) / ns_to_ms,
        },
    );
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
