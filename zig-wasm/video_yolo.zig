const std = @import("std");
const config = @import("config");

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavutil/avutil.h");
    @cInclude("libavutil/hwcontext.h");
    @cInclude("libavutil/hwcontext_cuda.h");
    @cInclude("libavutil/pixdesc.h");
    @cInclude("libavfilter/avfilter.h");
    @cInclude("libavfilter/buffersink.h");
    @cInclude("libavfilter/buffersrc.h");
    @cInclude("cuda_runtime.h");
});

extern fn launch_nv12_resize_to_rgb_nchw(
    src_y: [*]const u8,
    src_uv: [*]const u8,
    src_pitch_y: c_int,
    src_pitch_uv: c_int,
    dst: [*]f32,
    src_width: c_int,
    src_height: c_int,
    dst_width: c_int,
    dst_height: c_int,
    stream: ?*anyopaque,
) void;

const input_w: usize = 640;
const input_h: usize = 640;
const input_len: usize = 1 * 3 * input_w * input_h;
const queue_capacity: usize = 32;
const inference_workers: usize = 1;
const conf_threshold: f32 = 0.25;
const max_detections_per_frame: usize = 32;

const FrameQueue = struct {
    allocator: std.mem.Allocator,
    capacity: usize,
    frames: []?*c.AVFrame,
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

    fn init(allocator: std.mem.Allocator, capacity: usize, wg: *std.Thread.WaitGroup) !FrameQueue {
        const frames = try allocator.alloc(?*c.AVFrame, capacity);
        errdefer allocator.free(frames);
        @memset(frames, null);
        const free_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(free_ring);
        const ready_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(ready_ring);
        for (0..capacity) |i| free_ring[i] = i;
        return .{
            .allocator = allocator,
            .capacity = capacity,
            .frames = frames,
            .free_ring = free_ring,
            .ready_ring = ready_ring,
            .free_len = capacity,
            .wg = wg,
        };
    }

    fn deinit(self: *FrameQueue) void {
        for (self.frames) |frame_opt| {
            if (frame_opt) |frame| {
                var f: ?*c.AVFrame = frame;
                c.av_frame_free(&f);
            }
        }
        self.allocator.free(self.frames);
        self.allocator.free(self.ready_ring);
        self.allocator.free(self.free_ring);
    }

    fn setFrame(self: *FrameQueue, idx: usize, frame: *c.AVFrame) void {
        std.debug.assert(self.frames[idx] == null);
        self.frames[idx] = frame;
    }

    fn takeFrame(self: *FrameQueue, idx: usize) ?*c.AVFrame {
        const frame = self.frames[idx];
        self.frames[idx] = null;
        return frame;
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
        while (self.ready_len == 0 and !self.wg.isDone()) self.not_empty.wait(&self.mutex);
        if (self.ready_len == 0 and self.wg.isDone()) return null;
        const idx = self.ready_ring[self.ready_head];
        self.ready_head = (self.ready_head + 1) % self.capacity;
        self.ready_len -= 1;
        return idx;
    }

    fn releaseSlot(self: *FrameQueue, idx: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const tail = (self.free_head + self.free_len) % self.capacity;
        self.free_ring[tail] = idx;
        self.free_len += 1;
        self.not_full.signal();
    }

    fn producerDoneSignal(self: *FrameQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.not_empty.broadcast();
    }
};

const DecoderCtx = struct {
    queue: *FrameQueue,
    wg: *std.Thread.WaitGroup,
    video_path_z: [:0]const u8,
};

const Metrics = struct {
    mutex: std.Thread.Mutex = .{},
    frames: usize = 0,
    submit_ns: i128 = 0,
    queue_wait_ns: i128 = 0,

    fn add(self: *Metrics, frame_count: usize, submit_delta_ns: i128, queue_wait_delta_ns: i128) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.frames += frame_count;
        self.submit_ns += submit_delta_ns;
        self.queue_wait_ns += queue_wait_delta_ns;
    }
};

const InferenceWorkerCtx = struct {
    queue: *FrameQueue,
    api: *const c.OrtApi,
    model_path_z: [:0]const u8,
    metrics: *Metrics,
    detections_file: ?*std.fs.File,
    detections_mutex: ?*std.Thread.Mutex,
};

fn clamp01(v: f32) f32 {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

fn normalizeBox(x1_in: f32, y1_in: f32, x2_in: f32, y2_in: f32) struct { x1: f32, y1: f32, x2: f32, y2: f32 } {
    var x1 = x1_in;
    var y1 = y1_in;
    var x2 = x2_in;
    var y2 = y2_in;

    if (x2 <= x1 or y2 <= y1) {
        const cx = x1;
        const cy = y1;
        const w = @abs(x2);
        const h = @abs(y2);
        x1 = cx - (w / 2.0);
        y1 = cy - (h / 2.0);
        x2 = cx + (w / 2.0);
        y2 = cy + (h / 2.0);
    }

    if (x2 > 1.5 or y2 > 1.5 or x1 > 1.5 or y1 > 1.5) {
        x1 /= @as(f32, @floatFromInt(input_w));
        x2 /= @as(f32, @floatFromInt(input_w));
        y1 /= @as(f32, @floatFromInt(input_h));
        y2 /= @as(f32, @floatFromInt(input_h));
    }

    return .{
        .x1 = clamp01(x1),
        .y1 = clamp01(y1),
        .x2 = clamp01(x2),
        .y2 = clamp01(y2),
    };
}

fn writeFrameDetections(ctx: *InferenceWorkerCtx, frame_index: usize, boxes: []const f32, logits: []const f32) !void {
    if (ctx.detections_file == null or ctx.detections_mutex == null) return;
    const file = ctx.detections_file.?;
    const mutex = ctx.detections_mutex.?;
    var line = std.ArrayList(u8).empty;
    defer line.deinit(std.heap.c_allocator);
    const writer = &line.writer(std.heap.c_allocator);

    mutex.lock();
    defer mutex.unlock();

    try writer.print("{{\"frame\":{d},\"detections\":[", .{frame_index});
    var written: usize = 0;
    for (0..300) |box_idx| {
        var best_class: usize = 0;
        var best_score: f32 = 0.0;
        for (0..80) |class_idx| {
            const score = logits[box_idx * 80 + class_idx];
            if (score > best_score) {
                best_score = score;
                best_class = class_idx;
            }
        }
        if (best_score < conf_threshold) continue;
        if (written >= max_detections_per_frame) break;

        const box = normalizeBox(
            boxes[box_idx * 4 + 0],
            boxes[box_idx * 4 + 1],
            boxes[box_idx * 4 + 2],
            boxes[box_idx * 4 + 3],
        );
        if (box.x2 <= box.x1 or box.y2 <= box.y1) continue;

        if (written > 0) try writer.writeAll(",");
        try writer.print(
            "{{\"class_id\":{d},\"score\":{d:.4},\"x1\":{d:.6},\"y1\":{d:.6},\"x2\":{d:.6},\"y2\":{d:.6}}}",
            .{ best_class, best_score, box.x1, box.y1, box.x2, box.y2 },
        );
        written += 1;
    }
    try writer.writeAll("]}\n");
    try file.writeAll(line.items);
}

fn get_format(ctx: ?*c.AVCodecContext, pix_fmts: ?[*]const c.enum_AVPixelFormat) callconv(.c) c.enum_AVPixelFormat {
    _ = ctx;
    var p = pix_fmts.?;
    var first: c.enum_AVPixelFormat = c.AV_PIX_FMT_NONE;
    var first_sw: c.enum_AVPixelFormat = c.AV_PIX_FMT_NONE;
    while (p[0] != c.AV_PIX_FMT_NONE) : (p += 1) {
        const fmt = p[0];
        if (first == c.AV_PIX_FMT_NONE) first = fmt;
        const desc = c.av_pix_fmt_desc_get(fmt);
        if (desc != null and (desc.?.*.flags & c.AV_PIX_FMT_FLAG_HWACCEL) == 0 and first_sw == c.AV_PIX_FMT_NONE) {
            first_sw = fmt;
        }
    }
    if (first_sw != c.AV_PIX_FMT_NONE) return first_sw;
    return first;
}

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

    // HW Context
    var hw_device_ctx: ?*c.AVBufferRef = null;
    try avCheck(c.av_hwdevice_ctx_create(&hw_device_ctx, c.AV_HWDEVICE_TYPE_CUDA, null, null, 0));
    defer c.av_buffer_unref(&hw_device_ctx);

    var codec_ctx = c.avcodec_alloc_context3(decoder);
    if (codec_ctx == null) return error.OutOfMemory;
    defer c.avcodec_free_context(&codec_ctx);

    try avCheck(c.avcodec_parameters_to_context(codec_ctx, codecpar));
    codec_ctx.?.*.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);
    codec_ctx.?.*.get_format = get_format;
    try avCheck(c.avcodec_open2(codec_ctx, decoder, null));

    // Filter graph is initialized lazily from the first decoded frame so
    // pix_fmt / hw_frames_ctx match the actual decoder output.
    var filter_graph: ?*c.AVFilterGraph = null;
    defer if (filter_graph != null) c.avfilter_graph_free(&filter_graph);
    var buffer_ctx: ?*c.AVFilterContext = null;
    var buffersink_ctx: ?*c.AVFilterContext = null;

    var frame = c.av_frame_alloc();
    defer c.av_frame_free(&frame);
    var filt_frame = c.av_frame_alloc();
    defer c.av_frame_free(&filt_frame);
    var packet = c.av_packet_alloc();
    defer c.av_packet_free(&packet);
    var done = false;
    while (!done) {
        const read_ret = c.av_read_frame(fmt_ctx, packet);
        if (read_ret < 0) {
            try avCheck(c.avcodec_send_packet(codec_ctx, null));
            done = true;
        } else {
            defer c.av_packet_unref(packet);
            if (packet.?.*.stream_index == stream_idx) {
                try avCheck(c.avcodec_send_packet(codec_ctx, packet));
            }
        }

        while (true) {
            const recv_ret = c.avcodec_receive_frame(codec_ctx, frame);
            if (recv_ret == c.AVERROR(c.EAGAIN)) break;
            if (recv_ret == c.AVERROR_EOF) {
                // Flush filter
                if (buffer_ctx != null and c.av_buffersrc_add_frame_flags(buffer_ctx, null, 0) >= 0) {
                    // Could pull remaining frames here if loop continued
                }
                done = true;
                break;
            }
            try avCheck(recv_ret);

            if (filter_graph == null) {
                filter_graph = c.avfilter_graph_alloc();
                if (filter_graph == null) return error.OutOfMemory;

                var args_buf: [512]u8 = undefined;
                const args = try std.fmt.bufPrintZ(
                    &args_buf,
                    "video_size={d}x{d}:pix_fmt={d}:time_base={d}/{d}:pixel_aspect={d}/{d}",
                    .{
                        frame.?.*.width,                       frame.?.*.height,
                        frame.?.*.format,                      stream.?.*.time_base.num,
                        stream.?.*.time_base.den,              codec_ctx.?.*.sample_aspect_ratio.num,
                        codec_ctx.?.*.sample_aspect_ratio.den,
                    },
                );

                const buffer_filter = c.avfilter_get_by_name("buffer");
                const buffersink_filter = c.avfilter_get_by_name("buffersink");

                try avCheck(c.avfilter_graph_create_filter(&buffer_ctx, buffer_filter, "in", args.ptr, null, filter_graph));
                try avCheck(c.avfilter_graph_create_filter(&buffersink_ctx, buffersink_filter, "out", null, null, filter_graph));
                buffer_ctx.?.*.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);
                if (frame.?.*.hw_frames_ctx != null) {
                    const src_params = c.av_buffersrc_parameters_alloc() orelse return error.OutOfMemory;
                    defer c.av_free(src_params);
                    src_params.?.*.hw_frames_ctx = c.av_buffer_ref(frame.?.*.hw_frames_ctx);
                    try avCheck(c.av_buffersrc_parameters_set(buffer_ctx, src_params));
                }

                var inputs = c.avfilter_inout_alloc();
                var outputs = c.avfilter_inout_alloc();
                defer c.avfilter_inout_free(&inputs);
                defer c.avfilter_inout_free(&outputs);

                outputs.?.*.name = c.av_strdup("in");
                outputs.?.*.filter_ctx = buffer_ctx;
                outputs.?.*.pad_idx = 0;
                outputs.?.*.next = null;

                inputs.?.*.name = c.av_strdup("out");
                inputs.?.*.filter_ctx = buffersink_ctx;
                inputs.?.*.pad_idx = 0;
                inputs.?.*.next = null;

                const hw_filter_spec = "scale_cuda=format=nv12";
                const sw_filter_spec = "format=nv12,hwupload_cuda,scale_cuda=format=nv12";
                const filter_spec = if (frame.?.*.hw_frames_ctx != null) hw_filter_spec else sw_filter_spec;
                try avCheck(c.avfilter_graph_parse_ptr(filter_graph, filter_spec, &inputs, &outputs, null));
                try avCheck(c.avfilter_graph_config(filter_graph, null));
            }

            try avCheck(c.av_buffersrc_add_frame_flags(buffer_ctx, frame, c.AV_BUFFERSRC_FLAG_KEEP_REF));
            c.av_frame_unref(frame);

            while (true) {
                const filt_ret = c.av_buffersink_get_frame(buffersink_ctx, filt_frame);
                if (filt_ret == c.AVERROR(c.EAGAIN) or filt_ret == c.AVERROR_EOF) break;
                try avCheck(filt_ret);

                const slot_idx = ctx.queue.acquireWriteSlot();
                const owned_frame = c.av_frame_clone(filt_frame) orelse {
                    ctx.queue.releaseSlot(slot_idx);
                    return error.OutOfMemory;
                };
                ctx.queue.setFrame(slot_idx, owned_frame);
                ctx.queue.publishReady(slot_idx);
                c.av_frame_unref(filt_frame);
            }
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

fn runInferenceWorker(ctx: *InferenceWorkerCtx) !void {
    var preprocess_stream_raw: c.cudaStream_t = null;
    if (c.cudaStreamCreate(&preprocess_stream_raw) != c.cudaSuccess) return error.CudaStreamCreateFailed;
    const preprocess_stream = preprocess_stream_raw.?;
    defer _ = c.cudaStreamDestroy(preprocess_stream);

    var inference_stream_raw: c.cudaStream_t = null;
    if (c.cudaStreamCreate(&inference_stream_raw) != c.cudaSuccess) return error.CudaStreamCreateFailed;
    const inference_stream = inference_stream_raw.?;
    defer _ = c.cudaStreamDestroy(inference_stream);

    var env_raw: ?*c.OrtEnv = null;
    try ortCheck(ctx.api, ctx.api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "video-yolo-zig-worker", &env_raw));
    const env = env_raw orelse return error.OnnxRuntimeError;
    defer ctx.api.ReleaseEnv.?(env);

    var session_options_raw: ?*c.OrtSessionOptions = null;
    try ortCheck(ctx.api, ctx.api.CreateSessionOptions.?(&session_options_raw));
    const session_options = session_options_raw orelse return error.OnnxRuntimeError;
    defer ctx.api.ReleaseSessionOptions.?(session_options);
    try ortCheck(ctx.api, ctx.api.SetInterOpNumThreads.?(session_options, 1));
    try ortCheck(ctx.api, ctx.api.SetIntraOpNumThreads.?(session_options, 1));

    const create_cuda = ctx.api.CreateCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
    const update_cuda = ctx.api.UpdateCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
    const append_cuda = ctx.api.SessionOptionsAppendExecutionProvider_CUDA_V2 orelse return error.CudaExecutionProviderUnavailable;
    const release_cuda = ctx.api.ReleaseCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
    var cuda_options_raw: ?*c.OrtCUDAProviderOptionsV2 = null;
    try ortCheck(ctx.api, create_cuda(&cuda_options_raw));
    const cuda_options = cuda_options_raw orelse return error.OnnxRuntimeError;
    defer release_cuda(cuda_options);

    var stream_buf: [32]u8 = undefined;
    const stream_str = try std.fmt.bufPrintZ(&stream_buf, "{d}", .{@intFromPtr(inference_stream)});
    const cuda_keys = [_][*c]const u8{"user_compute_stream"};
    const cuda_values = [_][*c]const u8{stream_str.ptr};
    try ortCheck(ctx.api, update_cuda(cuda_options, &cuda_keys, &cuda_values, cuda_keys.len));
    try ortCheck(ctx.api, append_cuda(session_options, cuda_options));

    var session: ?*c.OrtSession = null;
    try ortCheck(ctx.api, ctx.api.CreateSession.?(env, ctx.model_path_z.ptr, session_options, &session));
    defer if (session) |s| ctx.api.ReleaseSession.?(s);

    var memory_info: ?*c.OrtMemoryInfo = null;
    try ortCheck(ctx.api, ctx.api.CreateMemoryInfo.?("Cuda", c.OrtArenaAllocator, 0, c.OrtMemTypeDefault, &memory_info));
    defer if (memory_info) |mi| ctx.api.ReleaseMemoryInfo.?(mi);

    var ort_allocator: ?*c.OrtAllocator = null;
    try ortCheck(ctx.api, ctx.api.GetAllocatorWithDefaultOptions.?(&ort_allocator));
    var input_name_alloc: [*c]u8 = null;
    try ortCheck(ctx.api, ctx.api.SessionGetInputName.?(session, 0, ort_allocator, &input_name_alloc));
    defer _ = ctx.api.AllocatorFree.?(ort_allocator, input_name_alloc);
    var output0_name_alloc: [*c]u8 = null;
    var output1_name_alloc: [*c]u8 = null;
    try ortCheck(ctx.api, ctx.api.SessionGetOutputName.?(session, 0, ort_allocator, &output0_name_alloc));
    try ortCheck(ctx.api, ctx.api.SessionGetOutputName.?(session, 1, ort_allocator, &output1_name_alloc));
    defer _ = ctx.api.AllocatorFree.?(ort_allocator, output0_name_alloc);
    defer _ = ctx.api.AllocatorFree.?(ort_allocator, output1_name_alloc);

    const input_shape = [_]i64{ 1, 3, @intCast(input_h), @intCast(input_w) };
    const output_boxes_shape = [_]i64{ 1, 300, 4 };
    const output_logits_shape = [_]i64{ 1, 300, 80 };

    var input_tensor_ptr: ?*anyopaque = null;
    if (c.cudaMalloc(&input_tensor_ptr, input_len * @sizeOf(f32)) != c.cudaSuccess) return error.OutOfMemory;
    defer _ = c.cudaFree(input_tensor_ptr);
    const input_tensor_d: [*]f32 = @ptrCast(@alignCast(input_tensor_ptr));

    var output_boxes_ptr: ?*anyopaque = null;
    if (c.cudaMalloc(&output_boxes_ptr, 300 * 4 * @sizeOf(f32)) != c.cudaSuccess) return error.OutOfMemory;
    defer _ = c.cudaFree(output_boxes_ptr);
    const output_boxes_d: [*]f32 = @ptrCast(@alignCast(output_boxes_ptr));

    var output_logits_ptr: ?*anyopaque = null;
    if (c.cudaMalloc(&output_logits_ptr, 300 * 80 * @sizeOf(f32)) != c.cudaSuccess) return error.OutOfMemory;
    defer _ = c.cudaFree(output_logits_ptr);
    const output_logits_d: [*]f32 = @ptrCast(@alignCast(output_logits_ptr));

    var input_value: ?*c.OrtValue = null;
    try ortCheck(ctx.api, ctx.api.CreateTensorWithDataAsOrtValue.?(
        memory_info,
        input_tensor_d,
        input_len * @sizeOf(f32),
        &input_shape,
        input_shape.len,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_value,
    ));
    defer if (input_value) |v| ctx.api.ReleaseValue.?(v);

    var output_boxes_value: ?*c.OrtValue = null;
    try ortCheck(ctx.api, ctx.api.CreateTensorWithDataAsOrtValue.?(
        memory_info,
        output_boxes_d,
        300 * 4 * @sizeOf(f32),
        &output_boxes_shape,
        output_boxes_shape.len,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &output_boxes_value,
    ));
    defer if (output_boxes_value) |v| ctx.api.ReleaseValue.?(v);

    var output_logits_value: ?*c.OrtValue = null;
    try ortCheck(ctx.api, ctx.api.CreateTensorWithDataAsOrtValue.?(
        memory_info,
        output_logits_d,
        300 * 80 * @sizeOf(f32),
        &output_logits_shape,
        output_logits_shape.len,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &output_logits_value,
    ));
    defer if (output_logits_value) |v| ctx.api.ReleaseValue.?(v);

    var io_binding_raw: ?*c.OrtIoBinding = null;
    try ortCheck(ctx.api, ctx.api.CreateIoBinding.?(session, &io_binding_raw));
    const io_binding = io_binding_raw orelse return error.OnnxRuntimeError;
    defer ctx.api.ReleaseIoBinding.?(io_binding);
    ctx.api.ClearBoundInputs.?(io_binding);
    ctx.api.ClearBoundOutputs.?(io_binding);
    try ortCheck(ctx.api, ctx.api.BindInput.?(io_binding, input_name_alloc, input_value));
    try ortCheck(ctx.api, ctx.api.BindOutput.?(io_binding, output0_name_alloc, output_logits_value));
    try ortCheck(ctx.api, ctx.api.BindOutput.?(io_binding, output1_name_alloc, output_boxes_value));

    var preprocess_event: c.cudaEvent_t = null;
    if (c.cudaEventCreateWithFlags(&preprocess_event, c.cudaEventDisableTiming) != c.cudaSuccess) return error.CudaEventCreateFailed;
    defer _ = c.cudaEventDestroy(preprocess_event);

    const host_boxes = try std.heap.c_allocator.alloc(f32, 300 * 4);
    defer std.heap.c_allocator.free(host_boxes);
    const host_logits = try std.heap.c_allocator.alloc(f32, 300 * 80);
    defer std.heap.c_allocator.free(host_logits);
    var frame_index: usize = 0;

    while (true) {
        const wait_start_ns = std.time.nanoTimestamp();
        const maybe_slot_idx = ctx.queue.acquireReadSlot();
        const wait_end_ns = std.time.nanoTimestamp();
        const slot_idx = maybe_slot_idx orelse break;
        const queue_wait_delta_ns = wait_end_ns - wait_start_ns;

        defer ctx.queue.releaseSlot(slot_idx);
        const frame = ctx.queue.takeFrame(slot_idx) orelse continue;
        defer {
            var frame_to_free: ?*c.AVFrame = frame;
            c.av_frame_free(&frame_to_free);
        }
        if (frame.*.data[0] == null or frame.*.data[1] == null) return error.InvalidFrameData;
        if (frame.*.width <= 0 or frame.*.height <= 0) return error.InvalidFrameData;

        launch_nv12_resize_to_rgb_nchw(
            frame.*.data[0],
            frame.*.data[1],
            frame.*.linesize[0],
            frame.*.linesize[1],
            input_tensor_d,
            frame.*.width,
            frame.*.height,
            @intCast(input_w),
            @intCast(input_h),
            preprocess_stream,
        );
        if (c.cudaEventRecord(preprocess_event, preprocess_stream) != c.cudaSuccess) return error.CudaEventRecordFailed;
        if (c.cudaStreamWaitEvent(inference_stream, preprocess_event, 0) != c.cudaSuccess) return error.CudaStreamWaitEventFailed;

        const start_ns = std.time.nanoTimestamp();
        try ortCheck(ctx.api, ctx.api.RunWithBinding.?(session, null, io_binding));
        if (c.cudaStreamSynchronize(inference_stream) != c.cudaSuccess) return error.CudaStreamSyncFailed;
        if (c.cudaMemcpy(
            host_boxes.ptr,
            output_boxes_d,
            300 * 4 * @sizeOf(f32),
            c.cudaMemcpyDeviceToHost,
        ) != c.cudaSuccess) return error.CudaMemcpyFailed;
        if (c.cudaMemcpy(
            host_logits.ptr,
            output_logits_d,
            300 * 80 * @sizeOf(f32),
            c.cudaMemcpyDeviceToHost,
        ) != c.cudaSuccess) return error.CudaMemcpyFailed;
        const end_ns = std.time.nanoTimestamp();
        frame_index += 1;
        try writeFrameDetections(ctx, frame_index, host_boxes, host_logits);
        ctx.metrics.add(1, end_ns - start_ns, queue_wait_delta_ns);
    }
}

fn inferenceWorkerMain(ctx: *InferenceWorkerCtx) void {
    runInferenceWorker(ctx) catch |err| {
        std.log.err("inference worker error: {}", .{err});
    };
}

fn runBenchmark(allocator: std.mem.Allocator, video_path: []const u8, producer_count: usize, detections_path: ?[]const u8) !void {
    var wg: std.Thread.WaitGroup = .{};
    wg.startMany(producer_count);
    var queue = try FrameQueue.init(allocator, queue_capacity, &wg);
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
    const model_path_z = try allocator.dupeZ(u8, config.model_path);
    defer allocator.free(model_path_z);
    var metrics: Metrics = .{};
    var detections_file: ?std.fs.File = null;
    defer if (detections_file) |*f| f.close();
    if (detections_path) |path| {
        detections_file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    }
    var detections_mutex: std.Thread.Mutex = .{};
    var worker_ctxs: [inference_workers]InferenceWorkerCtx = undefined;
    var worker_threads: [inference_workers]?std.Thread = .{null} ** inference_workers;
    const wall_start_ns = std.time.nanoTimestamp();

    for (0..inference_workers) |i| {
        worker_ctxs[i] = .{
            .queue = &queue,
            .api = api,
            .model_path_z = model_path_z,
            .metrics = &metrics,
            .detections_file = if (detections_file) |*f| f else null,
            .detections_mutex = if (detections_file != null) &detections_mutex else null,
        };
        worker_threads[i] = try std.Thread.spawn(.{}, inferenceWorkerMain, .{&worker_ctxs[i]});
    }

    for (worker_threads) |t| if (t) |thr| thr.join();

    const wall_end_ns = std.time.nanoTimestamp();
    const wall_ns = wall_end_ns - wall_start_ns;

    if (metrics.frames == 0 or metrics.submit_ns <= 0 or wall_ns <= 0) {
        std.debug.print("submit_s=0.000 wall_s=0.000 queue_wait_s=0.000 workers={d} frames=0 submit_fps=0.00 wall_fps=0.00\n", .{inference_workers});
        return;
    }
    const submit_s = @as(f64, @floatFromInt(metrics.submit_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const wall_s = @as(f64, @floatFromInt(wall_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const queue_wait_s = @as(f64, @floatFromInt(metrics.queue_wait_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const submit_fps =
        (@as(f64, @floatFromInt(metrics.frames)) * @as(f64, @floatFromInt(std.time.ns_per_s))) /
        @as(f64, @floatFromInt(metrics.submit_ns));
    const wall_fps =
        (@as(f64, @floatFromInt(metrics.frames)) * @as(f64, @floatFromInt(std.time.ns_per_s))) /
        @as(f64, @floatFromInt(wall_ns));
    std.debug.print("submit_s={d:.3} wall_s={d:.3} queue_wait_s={d:.3} workers={d} frames={d} submit_fps={d:.2} wall_fps={d:.2}\n", .{
        submit_s,
        wall_s,
        queue_wait_s,
        inference_workers,
        metrics.frames,
        submit_fps,
        wall_fps,
    });
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: {s} <video> <producer_count> [detections.jsonl]\n", .{args[0]});
        return error.InvalidArguments;
    }
    const producer_count = try std.fmt.parseInt(usize, args[2], 10);
    if (producer_count == 0) return error.InvalidArguments;
    const detections_path: ?[]const u8 = if (args.len == 4) args[3] else null;
    try runBenchmark(allocator, args[1], producer_count, detections_path);
}
