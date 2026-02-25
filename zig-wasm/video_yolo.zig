const std = @import("std");
const config = @import("config");
const yolo = @import("yolo.zig");

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

extern fn launch_nv12_to_rgb_nchw(
    src_y: [*]const u8,
    src_uv: [*]const u8,
    src_pitch_y: c_int,
    src_pitch_uv: c_int,
    dst: [*]f32,
    width: c_int,
    height: c_int,
    stream: ?*anyopaque,
) void;

const input_w: usize = 640;
const input_h: usize = 640;
const input_len: usize = 1 * 3 * input_w * input_h;
const queue_capacity: usize = 32;

const TensorQueue = struct {
    allocator: std.mem.Allocator,
    tensor_len: usize,
    capacity: usize,
    storage_d: [*]f32,
    ready_events: []c.cudaEvent_t,
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
        var storage_ptr: ?*anyopaque = null;
        if (c.cudaMalloc(&storage_ptr, capacity * tensor_len * @sizeOf(f32)) != c.cudaSuccess) return error.OutOfMemory;
        const storage_d: [*]f32 = @ptrCast(@alignCast(storage_ptr));

        const ready_events = try allocator.alloc(c.cudaEvent_t, capacity);
        var events_created: usize = 0;
        errdefer {
            for (0..events_created) |i| _ = c.cudaEventDestroy(ready_events[i]);
            allocator.free(ready_events);
        }
        for (0..capacity) |i| {
            var ev: c.cudaEvent_t = null;
            if (c.cudaEventCreateWithFlags(&ev, c.cudaEventDisableTiming) != c.cudaSuccess) return error.CudaEventCreateFailed;
            ready_events[i] = ev;
            events_created += 1;
        }

        const free_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(free_ring);
        const ready_ring = try allocator.alloc(usize, capacity);
        errdefer allocator.free(ready_ring);
        for (0..capacity) |i| free_ring[i] = i;
        return .{
            .allocator = allocator,
            .tensor_len = tensor_len,
            .capacity = capacity,
            .storage_d = storage_d,
            .ready_events = ready_events,
            .free_ring = free_ring,
            .ready_ring = ready_ring,
            .free_len = capacity,
            .wg = wg,
        };
    }

    fn deinit(self: *TensorQueue) void {
        for (self.ready_events) |ev| _ = c.cudaEventDestroy(ev);
        self.allocator.free(self.ready_events);
        self.allocator.free(self.ready_ring);
        self.allocator.free(self.free_ring);
        _ = c.cudaFree(self.storage_d);
    }

    fn slot(self: *TensorQueue, idx: usize) [*]f32 {
        const start = idx * self.tensor_len;
        return self.storage_d + start;
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

    fn recordAndPublishReady(self: *TensorQueue, idx: usize, stream: c.cudaStream_t) !void {
        if (c.cudaEventRecord(self.ready_events[idx], stream) != c.cudaSuccess) return error.CudaEventRecordFailed;
        self.publishReady(idx);
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

    fn waitReady(self: *TensorQueue, idx: usize) !void {
        if (c.cudaEventSynchronize(self.ready_events[idx]) != c.cudaSuccess) return error.CudaEventSynchronizeFailed;
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
    cuda_stream: c.cudaStream_t,
};

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

fn decodeYolo26Heads(
    allocator: std.mem.Allocator,
    logits: []const f32,
    boxes: []const f32,
    det_count: usize,
    class_count: usize,
    conf_threshold: f32,
) ![]yolo.Detection {
    var out = try std.ArrayList(yolo.Detection).initCapacity(allocator, 0);
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, det_count);

    for (0..det_count) |i| {
        const logit_row = logits[(i * class_count)..][0..class_count];
        var best_class: usize = 0;
        var best_logit: f32 = logit_row[0];
        for (1..class_count) |j| {
            if (logit_row[j] > best_logit) {
                best_logit = logit_row[j];
                best_class = j;
            }
        }

        const score = 1.0 / (1.0 + @exp(-best_logit));
        if (score < conf_threshold) continue;

        const cx = boxes[(i * 4) + 0];
        const cy = boxes[(i * 4) + 1];
        const w = boxes[(i * 4) + 2];
        const h = boxes[(i * 4) + 3];
        const normalized = (@abs(cx) <= 1.5 and @abs(cy) <= 1.5 and @abs(w) <= 1.5 and @abs(h) <= 1.5);
        const scale_x: f32 = if (normalized) @floatFromInt(input_w) else 1.0;
        const scale_y: f32 = if (normalized) @floatFromInt(input_h) else 1.0;

        try out.append(allocator, .{
            .class_id = best_class,
            .score = score,
            .x1 = (cx - (w / 2.0)) * scale_x,
            .y1 = (cy - (h / 2.0)) * scale_y,
            .x2 = (cx + (w / 2.0)) * scale_x,
            .y2 = (cy + (h / 2.0)) * scale_y,
        });
    }

    std.mem.sort(yolo.Detection, out.items, {}, struct {
        fn lessThan(_: void, a: yolo.Detection, b: yolo.Detection) bool {
            return a.score > b.score;
        }
    }.lessThan);

    return out.toOwnedSlice(allocator);
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
                const args = try std.fmt.bufPrintZ(&args_buf,
                    "video_size={d}x{d}:pix_fmt={d}:time_base={d}/{d}:pixel_aspect={d}/{d}",
                    .{
                        frame.?.*.width, frame.?.*.height,
                        frame.?.*.format,
                        stream.?.*.time_base.num, stream.?.*.time_base.den,
                        codec_ctx.?.*.sample_aspect_ratio.num, codec_ctx.?.*.sample_aspect_ratio.den,
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

                const hw_filter_spec = "scale_cuda=640:640:format=nv12";
                const sw_filter_spec = "format=nv12,hwupload_cuda,scale_cuda=640:640:format=nv12";
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
                const slot = ctx.queue.slot(slot_idx);

                launch_nv12_to_rgb_nchw(
                    filt_frame.?.*.data[0],
                    filt_frame.?.*.data[1],
                    filt_frame.?.*.linesize[0],
                    filt_frame.?.*.linesize[1],
                    slot,
                    640,
                    640,
                    ctx.cuda_stream,
                );

                try ctx.queue.recordAndPublishReady(slot_idx, ctx.cuda_stream);
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
        var stream: c.cudaStream_t = null;
        if (c.cudaStreamCreate(&stream) != c.cudaSuccess) return error.CudaStreamCreateFailed;
        decoder_ctxs[i] = .{
            .queue = &queue,
            .wg = &wg,
            .video_path_z = video_path_z,
            .cuda_stream = stream.?,
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
    try ortCheck(api, api.CreateMemoryInfo.?("Cuda", c.OrtArenaAllocator, 0, c.OrtMemTypeDefault, &memory_info));
    defer if (memory_info) |mi| api.ReleaseMemoryInfo.?(mi);

    var ort_allocator: ?*c.OrtAllocator = null;
    try ortCheck(api, api.GetAllocatorWithDefaultOptions.?(&ort_allocator));
    var input_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetInputName.?(session, 0, ort_allocator, &input_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, input_name_alloc);
    var output0_name_alloc: [*c]u8 = null;
    var output1_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetOutputName.?(session, 0, ort_allocator, &output0_name_alloc));
    try ortCheck(api, api.SessionGetOutputName.?(session, 1, ort_allocator, &output1_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, output0_name_alloc);
    defer _ = api.AllocatorFree.?(ort_allocator, output1_name_alloc);

    const input_shape = [_]i64{ 1, 3, @intCast(input_h), @intCast(input_w) };
    const input_names = [_][*c]const u8{input_name_alloc};
    const output_names = [_][*c]const u8{ output0_name_alloc, output1_name_alloc };

    var decoded_frames: usize = 0;
    var total_ns: i128 = 0;

    while (queue.acquireReadSlot()) |slot_idx| {
        defer queue.releaseReadSlot(slot_idx);
        const slot = queue.slot(slot_idx);
        try queue.waitReady(slot_idx);

        var input_value: ?*c.OrtValue = null;
        try ortCheck(api, api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            slot,
            input_len * @sizeOf(f32),
            &input_shape,
            input_shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value,
        ));
        defer if (input_value) |v| api.ReleaseValue.?(v);

        var output_values = [_]?*c.OrtValue{ null, null };
        defer for (output_values) |v| if (v) |x| api.ReleaseValue.?(x);
        const input_values = [_]?*c.OrtValue{input_value};

        const start_ns = std.time.nanoTimestamp();
        try ortCheck(api, api.Run.?(
            session,
            null,
            &input_names,
            &input_values,
            1,
            &output_names,
            output_names.len,
            &output_values,
        ));
        const end_ns = std.time.nanoTimestamp();

        var logits: []const f32 = &.{};
        var boxes: []const f32 = &.{};
        var det_count: usize = 0;
        var class_count: usize = 0;

        for (output_values) |output_value| {
            var tensor_info: ?*c.OrtTensorTypeAndShapeInfo = null;
            try ortCheck(api, api.GetTensorTypeAndShape.?(output_value, &tensor_info));
            defer api.ReleaseTensorTypeAndShapeInfo.?(tensor_info);
            var dim_count: usize = 0;
            try ortCheck(api, api.GetDimensionsCount.?(tensor_info, &dim_count));
            var dims_buf: [8]i64 = undefined;
            if (dim_count > dims_buf.len) return error.UnsupportedOutputRank;
            try ortCheck(api, api.GetDimensions.?(tensor_info, &dims_buf, dim_count));
            if (dim_count != 3) return error.UnsupportedOutputShape;
            const rows: usize = @intCast(dims_buf[1]);
            const cols: usize = @intCast(dims_buf[2]);

            var out_ptr_raw: ?*anyopaque = null;
            try ortCheck(api, api.GetTensorMutableData.?(output_value, &out_ptr_raw));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(out_ptr_raw));
            const total = rows * cols;
            const slice = out_ptr[0..total];

            if (cols == 4) {
                boxes = slice;
                if (det_count == 0) det_count = rows else if (det_count != rows) return error.UnsupportedOutputShape;
            } else if (cols > 4) {
                logits = slice;
                class_count = cols;
                if (det_count == 0) det_count = rows else if (det_count != rows) return error.UnsupportedOutputShape;
            } else {
                return error.UnsupportedOutputShape;
            }
        }
        if (det_count == 0 or class_count == 0 or logits.len == 0 or boxes.len == 0) return error.UnsupportedOutputShape;

        const dets = try decodeYolo26Heads(allocator, logits, boxes, det_count, class_count, 0.25);
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
