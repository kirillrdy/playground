const std = @import("std");
const config = @import("config");
const yolo = @import("yolo.zig");

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavutil/avutil.h");
    @cInclude("libswscale/swscale.h");
});

const input_w: usize = 640;
const input_h: usize = 640;
const input_len: usize = 1 * 3 * input_w * input_h;
const queue_capacity: usize = 32;

const OutputSpec = struct {
    boxes: usize,
    classes: usize,
    layout: yolo.OutputLayout,
};

const TensorQueue = struct {
    allocator: std.mem.Allocator,
    tensor_len: usize,
    capacity: usize,
    storage: []f32,
    free_ring: []usize,
    ready_ring: []usize,
    free_head: usize = 0,
    free_len: usize = 0,
    ready_head: usize = 0,
    ready_len: usize = 0,
    wg: *std.Thread.WaitGroup,
    mutex: std.Thread.Mutex = .{},
    not_empty: std.Thread.Condition = .{},
    not_full: std.Thread.Condition = .{},

    fn init(allocator: std.mem.Allocator, capacity: usize, tensor_len: usize, wg: *std.Thread.WaitGroup) !TensorQueue {
        const storage = try allocator.alloc(f32, capacity * tensor_len);
        errdefer allocator.free(storage);
        const free_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(free_ring);
        const ready_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(ready_ring);
        for (0..capacity) |i| free_ring[i] = i;
        return .{
            .allocator = allocator,
            .tensor_len = tensor_len,
            .capacity = capacity,
            .storage = storage,
            .free_ring = free_ring,
            .ready_ring = ready_ring,
            .free_len = capacity,
            .wg = wg,
        };
    }

    fn deinit(self: *TensorQueue) void {
        self.allocator.free(self.ready_ring);
        self.allocator.free(self.free_ring);
        self.allocator.free(self.storage);
    }

    fn slot(self: *TensorQueue, idx: usize) []f32 {
        const start = idx * self.tensor_len;
        return self.storage[start .. start + self.tensor_len];
    }

    fn acquireWriteSlot(self: *TensorQueue) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.free_len == 0) self.not_full.wait(&self.mutex);
        const idx = self.free_ring[self.free_head];
        self.free_head = (self.free_head + 1) % self.capacity;
        self.free_len -= 1;
        return idx;
    }

    fn publishReady(self: *TensorQueue, idx: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const tail = (self.ready_head + self.ready_len) % self.capacity;
        self.ready_ring[tail] = idx;
        self.ready_len += 1;
        self.not_empty.signal();
    }

    fn acquireReadSlot(self: *TensorQueue) ?usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.ready_len == 0 and !self.wg.isDone()) self.not_empty.wait(&self.mutex);
        if (self.ready_len == 0 and self.wg.isDone()) return null;
        const idx = self.ready_ring[self.ready_head];
        self.ready_head = (self.ready_head + 1) % self.capacity;
        self.ready_len -= 1;
        return idx;
    }

    fn releaseReadSlot(self: *TensorQueue, idx: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const tail = (self.free_head + self.free_len) % self.capacity;
        self.free_ring[tail] = idx;
        self.free_len += 1;
        self.not_full.signal();
    }

    fn producerDoneSignal(self: *TensorQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.not_empty.broadcast();
    }
};

const DecoderCtx = struct {
    queue: *TensorQueue,
    wg: *std.Thread.WaitGroup,
    video_path_z: [:0]const u8,
};

fn ortCheck(api: *const c.OrtApi, status: ?*c.OrtStatus) !void {
    if (status == null) return;
    defer api.ReleaseStatus.?(status);
    const msg = api.GetErrorMessage.?(status);
    if (msg != null) std.log.err("onnxruntime: {s}", .{std.mem.span(msg)});
    return error.OnnxRuntimeError;
}

fn avCheck(code: c_int) !void {
    if (code >= 0) return;
    return error.FfmpegError;
}

fn inferOutputSpec(dim_count: usize, dims: []const i64, total_values: usize) OutputSpec {
    if (dim_count >= 3) {
        const d1 = @as(usize, @intCast(dims[dim_count - 2]));
        const d2 = @as(usize, @intCast(dims[dim_count - 1]));
        if (d1 > d2) {
            const attrs = d2;
            return .{
                .boxes = d1,
                .classes = if (attrs > 4) attrs - 4 else 80,
                .layout = .boxes_first,
            };
        } else {
            const attrs = d1;
            return .{
                .boxes = d2,
                .classes = if (attrs > 4) attrs - 4 else 80,
                .layout = .attributes_first,
            };
        }
    }
    return .{
        .boxes = total_values / (80 + 4),
        .classes = 80,
        .layout = .boxes_first,
    };
}

fn rgbU8ToNchwF32(dst: []f32, rgb: []const u8, width: usize, height: usize) void {
    const plane = width * height;
    for (0..height) |y| {
        const row_base = y * width;
        for (0..width) |x| {
            const src_idx = (row_base + x) * 3;
            const dst_idx = row_base + x;
            dst[dst_idx] = @as(f32, @floatFromInt(rgb[src_idx])) * (1.0 / 255.0);
            dst[plane + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 1])) * (1.0 / 255.0);
            dst[(2 * plane) + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 2])) * (1.0 / 255.0);
        }
    }
}

fn decodeVideoIntoQueue(ctx: *DecoderCtx) !void {
    var fmt_ctx: ?*c.AVFormatContext = null;
    try avCheck(c.avformat_open_input(&fmt_ctx, ctx.video_path_z.ptr, null, null));
    defer c.avformat_close_input(&fmt_ctx);
    try avCheck(c.avformat_find_stream_info(fmt_ctx, null));

    const stream_idx = c.av_find_best_stream(fmt_ctx, c.AVMEDIA_TYPE_VIDEO, -1, -1, null, 0);
    if (stream_idx < 0) return error.NoVideoStream;
    const stream = fmt_ctx.?.*.streams[@intCast(stream_idx)];
    const codecpar = stream.?.*.codecpar;
    const decoder = c.avcodec_find_decoder(codecpar.?.*.codec_id) orelse return error.UnsupportedCodec;

    var codec_ctx = c.avcodec_alloc_context3(decoder);
    if (codec_ctx == null) return error.OutOfMemory;
    defer c.avcodec_free_context(&codec_ctx);
    try avCheck(c.avcodec_parameters_to_context(codec_ctx, codecpar));
    try avCheck(c.avcodec_open2(codec_ctx, decoder, null));

    var frame = c.av_frame_alloc();
    if (frame == null) return error.OutOfMemory;
    defer c.av_frame_free(&frame);

    var packet = c.av_packet_alloc();
    if (packet == null) return error.OutOfMemory;
    defer c.av_packet_free(&packet);

    const sws_ctx = c.sws_getContext(
        codec_ctx.?.*.width,
        codec_ctx.?.*.height,
        codec_ctx.?.*.pix_fmt,
        input_w,
        input_h,
        c.AV_PIX_FMT_RGB24,
        c.SWS_BILINEAR,
        null,
        null,
        null,
    ) orelse return error.OutOfMemory;
    defer c.sws_freeContext(sws_ctx);

    const rgb_len = input_w * input_h * 3;
    const rgb = try std.heap.c_allocator.alloc(u8, rgb_len);
    defer std.heap.c_allocator.free(rgb);
    var dst_data = [_][*c]u8{
        @ptrCast(rgb.ptr),
        null,
        null,
        null,
    };
    var dst_linesize = [_]c_int{
        @intCast(input_w * 3),
        0,
        0,
        0,
    };

    var done = false;
    while (!done) {
        const read_ret = c.av_read_frame(fmt_ctx, packet);
        if (read_ret < 0) {
            try avCheck(c.avcodec_send_packet(codec_ctx, null));
            done = true;
        } else {
            defer c.av_packet_unref(packet);
            if (packet.?.*.stream_index != stream_idx) continue;
            try avCheck(c.avcodec_send_packet(codec_ctx, packet));
        }

        while (true) {
            const recv_ret = c.avcodec_receive_frame(codec_ctx, frame);
            if (recv_ret == c.AVERROR(c.EAGAIN)) break;
            if (recv_ret == c.AVERROR_EOF) {
                done = true;
                break;
            }
            try avCheck(recv_ret);

            _ = c.sws_scale(
                sws_ctx,
                &frame.?.*.data,
                &frame.?.*.linesize,
                0,
                codec_ctx.?.*.height,
                &dst_data,
                &dst_linesize,
            );

            const slot_idx = ctx.queue.acquireWriteSlot();
            const slot = ctx.queue.slot(slot_idx);
            rgbU8ToNchwF32(slot, rgb, input_w, input_h);
            ctx.queue.publishReady(slot_idx);

            c.av_frame_unref(frame);
        }
    }
}

fn decoderThreadMain(ctx: *DecoderCtx) void {
    defer ctx.wg.finish();
    decodeVideoIntoQueue(ctx) catch |err| {
        std.log.err("decoder error ({s}): {}", .{ ctx.video_path_z, err });
    };
    ctx.queue.producerDoneSignal();
}

fn runBenchmark(allocator: std.mem.Allocator, video_path: []const u8, producer_count: usize) !void {
    var wg: std.Thread.WaitGroup = .{};
    wg.startMany(producer_count);
    var queue = try TensorQueue.init(allocator, queue_capacity, input_len, &wg);
    defer queue.deinit();

    const video_path_z = try allocator.dupeZ(u8, video_path);
    defer allocator.free(video_path_z);

    const decoder_ctxs = try allocator.alloc(DecoderCtx, producer_count);
    defer allocator.free(decoder_ctxs);
    const decoder_threads = try allocator.alloc(?std.Thread, producer_count);
    defer allocator.free(decoder_threads);
    @memset(decoder_threads, null);
    defer for (decoder_threads) |t| if (t) |thr| thr.join();

    for (0..producer_count) |i| {
        decoder_ctxs[i] = .{
            .queue = &queue,
            .wg = &wg,
            .video_path_z = video_path_z,
        };
        decoder_threads[i] = try std.Thread.spawn(.{}, decoderThreadMain, .{&decoder_ctxs[i]});
    }

    const base = c.OrtGetApiBase() orelse return error.OnnxRuntimeUnavailable;
    const get_api = base.*.GetApi orelse return error.OnnxRuntimeUnavailable;
    const api: *const c.OrtApi = @ptrCast(get_api(c.ORT_API_VERSION) orelse return error.OnnxRuntimeUnavailable);

    var env_raw: ?*c.OrtEnv = null;
    try ortCheck(api, api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "video-yolo-zig", &env_raw));
    const env = env_raw orelse return error.OnnxRuntimeError;
    defer api.ReleaseEnv.?(env);

    var session_options_raw: ?*c.OrtSessionOptions = null;
    try ortCheck(api, api.CreateSessionOptions.?(&session_options_raw));
    const session_options = session_options_raw orelse return error.OnnxRuntimeError;
    defer api.ReleaseSessionOptions.?(session_options);
    try ortCheck(api, api.SetInterOpNumThreads.?(session_options, 1));
    try ortCheck(api, api.SetIntraOpNumThreads.?(session_options, 1));

    const create_cuda = api.CreateCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
    const append_cuda = api.SessionOptionsAppendExecutionProvider_CUDA_V2 orelse return error.CudaExecutionProviderUnavailable;
    const release_cuda = api.ReleaseCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
    var cuda_options_raw: ?*c.OrtCUDAProviderOptionsV2 = null;
    try ortCheck(api, create_cuda(&cuda_options_raw));
    const cuda_options = cuda_options_raw orelse return error.OnnxRuntimeError;
    defer release_cuda(cuda_options);
    try ortCheck(api, append_cuda(session_options, cuda_options));

    const model_path_z = try allocator.dupeZ(u8, config.model_path);
    defer allocator.free(model_path_z);
    var session: ?*c.OrtSession = null;
    try ortCheck(api, api.CreateSession.?(env, model_path_z.ptr, session_options, &session));
    defer if (session) |s| api.ReleaseSession.?(s);

    var memory_info: ?*c.OrtMemoryInfo = null;
    try ortCheck(api, api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info));
    defer if (memory_info) |mi| api.ReleaseMemoryInfo.?(mi);

    var ort_allocator: ?*c.OrtAllocator = null;
    try ortCheck(api, api.GetAllocatorWithDefaultOptions.?(&ort_allocator));
    var input_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetInputName.?(session, 0, ort_allocator, &input_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, input_name_alloc);
    var output_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetOutputName.?(session, 0, ort_allocator, &output_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, output_name_alloc);

    const input_shape = [_]i64{ 1, 3, @intCast(input_h), @intCast(input_w) };
    const input_names = [_][*c]const u8{input_name_alloc};
    const output_names = [_][*c]const u8{output_name_alloc};

    var decoded_frames: usize = 0;
    var total_ns: i128 = 0;

    while (queue.acquireReadSlot()) |slot_idx| {
        defer queue.releaseReadSlot(slot_idx);
        const slot = queue.slot(slot_idx);

        var input_value: ?*c.OrtValue = null;
        try ortCheck(api, api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            slot.ptr,
            slot.len * @sizeOf(f32),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value,
        ));
        defer if (input_value) |v| api.ReleaseValue.?(v);

        var output_value: ?*c.OrtValue = null;
        defer if (output_value) |v| api.ReleaseValue.?(v);
        const input_values = [_]?*c.OrtValue{input_value};

        const start_ns = std.time.nanoTimestamp();
        try ortCheck(api, api.Run.?(
            session,
            null,
            &input_names,
            &input_values,
            1,
            &output_names,
            1,
            &output_value,
        ));
        const end_ns = std.time.nanoTimestamp();

        var tensor_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try ortCheck(api, api.GetTensorTypeAndShape.?(output_value, &tensor_info));
        defer api.ReleaseTensorTypeAndShapeInfo.?(tensor_info);
        var dim_count: usize = 0;
        try ortCheck(api, api.GetDimensionsCount.?(tensor_info, &dim_count));
        var dims_buf: [8]i64 = undefined;
        if (dim_count > dims_buf.len) return error.UnsupportedOutputRank;
        try ortCheck(api, api.GetDimensions.?(tensor_info, &dims_buf, dim_count));
        const dims = dims_buf[0..dim_count];
        var out_ptr_raw: ?*anyopaque = null;
        try ortCheck(api, api.GetTensorMutableData.?(output_value, &out_ptr_raw));
        const out_ptr: [*]f32 = @ptrCast(@alignCast(out_ptr_raw));
        var total: usize = 1;
        for (dims) |d| total *= @as(usize, @intCast(d));
        const spec = inferOutputSpec(dim_count, dims, total);
        const raw = try yolo.decodeV8(allocator, out_ptr[0..total], spec.boxes, spec.classes, spec.layout, 0.25);
        defer allocator.free(raw);
        const dets = try yolo.nms(allocator, raw, 0.45);
        defer allocator.free(dets);

        decoded_frames += 1;
        total_ns += end_ns - start_ns;
    }

    if (decoded_frames == 0 or total_ns <= 0) {
        std.debug.print("duration_s=0.000 frames=0 fps=0.00\n", .{});
        return;
    }
    const duration_s = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const fps =
        (@as(f64, @floatFromInt(decoded_frames)) * @as(f64, @floatFromInt(std.time.ns_per_s))) /
        @as(f64, @floatFromInt(total_ns));
    std.debug.print("duration_s={d:.3} frames={d} fps={d:.2}\n", .{ duration_s, decoded_frames, fps });
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 3) {
        std.debug.print("usage: {s} <video> <producer_count>\n", .{args[0]});
        return error.InvalidArguments;
    }
    const producer_count = try std.fmt.parseInt(usize, args[2], 10);
    if (producer_count == 0) return error.InvalidArguments;

    try runBenchmark(allocator, args[1], producer_count);
}
