const std = @import("std");
const config = @import("config");

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

fn ortCheck(api: *const c.OrtApi, status: ?*c.OrtStatus) !void {
    if (status == null) return;
    defer api.ReleaseStatus.?(status);
    const msg = api.GetErrorMessage.?(status);
    if (msg != null) std.log.err("onnxruntime: {s}", .{std.mem.span(msg)});
    return error.OnnxRuntimeError;
}

fn runBenchmark(allocator: std.mem.Allocator) !void {
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
    defer if (session) |v| api.ReleaseSession.?(v);

    var memory_info: ?*c.OrtMemoryInfo = null;
    try ortCheck(api, api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info));
    defer if (memory_info) |v| api.ReleaseMemoryInfo.?(v);

    var ort_allocator: ?*c.OrtAllocator = null;
    try ortCheck(api, api.GetAllocatorWithDefaultOptions.?(&ort_allocator));

    var input_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetInputName.?(session, 0, ort_allocator, &input_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, input_name_alloc);
    var output_name_alloc: [*c]u8 = null;
    try ortCheck(api, api.SessionGetOutputName.?(session, 0, ort_allocator, &output_name_alloc));
    defer _ = api.AllocatorFree.?(ort_allocator, output_name_alloc);

    const input_data = try allocator.alloc(f32, 1 * 3 * 640 * 640);
    defer allocator.free(input_data);
    @memset(input_data, 0);

    const input_shape = [_]i64{ 1, 3, 640, 640 };
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

    const warmup_iters: usize = 20;
    const measured_iters: usize = 100;
    const total_iters = warmup_iters + measured_iters;

    var total_ms: f64 = 0;
    for (0..total_iters) |i| {
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
        if (i >= warmup_iters) {
            total_ms += @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, std.time.ns_per_ms);
        }
    }

    const fps = (@as(f64, @floatFromInt(measured_iters)) * 1000.0) / total_ms;
    std.debug.print("{d:.2}\n", .{fps});
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    try runBenchmark(allocator);
}
