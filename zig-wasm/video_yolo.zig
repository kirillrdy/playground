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

const OutputSpec = struct {
    boxes: usize,
    classes: usize,
    layout: yolo.OutputLayout,
};

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

fn runBenchmark(allocator: std.mem.Allocator, video_path: []const u8) !void {
    const input_w: usize = 640;
    const input_h: usize = 640;

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

    var fmt_ctx: ?*c.AVFormatContext = null;
    const video_path_z = try allocator.dupeZ(u8, video_path);
    defer allocator.free(video_path_z);
    try avCheck(c.avformat_open_input(&fmt_ctx, video_path_z.ptr, null, null));
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
    const rgb = try allocator.alloc(u8, rgb_len);
    defer allocator.free(rgb);
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

    const input_data = try allocator.alloc(f32, 1 * 3 * input_w * input_h);
    defer allocator.free(input_data);
    const input_shape = [_]i64{ 1, 3, @intCast(input_h), @intCast(input_w) };
    var input_value: ?*c.OrtValue = null;
    try ortCheck(api, api.CreateTensorWithDataAsOrtValue.?(
        memory_info,
        input_data.ptr,
        input_data.len * @sizeOf(f32),
        &input_shape,
        input_shape.len,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_value,
    ));
    defer if (input_value) |v| api.ReleaseValue.?(v);

    const input_names = [_][*c]const u8{input_name_alloc};
    const output_names = [_][*c]const u8{output_name_alloc};
    const input_values = [_]?*c.OrtValue{input_value};

    var decoded_frames: usize = 0;
    var total_ns: i128 = 0;

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
            rgbU8ToNchwF32(input_data, rgb, input_w, input_h);

            var output_value: ?*c.OrtValue = null;
            defer if (output_value) |v| api.ReleaseValue.?(v);

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

            decoded_frames += 1;
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
            total_ns += end_ns - start_ns;

            c.av_frame_unref(frame);
        }
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
    if (args.len < 2) return error.InvalidArguments;

    try runBenchmark(allocator, args[1]);
}
