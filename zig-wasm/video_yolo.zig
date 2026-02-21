const std = @import("std");

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const MappingStat = struct {
    rss_kb: u64 = 0,
    pss_kb: u64 = 0,
    private_clean_kb: u64 = 0,
    private_dirty_kb: u64 = 0,
    name_len: usize = 0,
    name: [256]u8 = undefined,
};

fn ortCheck(api: *const c.OrtApi, status: ?*c.OrtStatus) !void {
    if (status == null) return;
    defer api.ReleaseStatus.?(status);
    const msg = api.GetErrorMessage.?(status);
    if (msg != null) {
        std.log.err("onnxruntime: {s}", .{std.mem.span(msg)});
    }
    return error.OnnxRuntimeError;
}

fn parseKbLine(line: []const u8, prefix: []const u8) ?u64 {
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const tail = std.mem.trimLeft(u8, line[prefix.len..], " \t");
    const end = std.mem.indexOfAny(u8, tail, " \t") orelse tail.len;
    return std.fmt.parseInt(u64, tail[0..end], 10) catch null;
}

fn isSmapsHeaderLine(line: []const u8) bool {
    const first_space = std.mem.indexOfScalar(u8, line, ' ') orelse return false;
    const addr = line[0..first_space];
    const dash = std.mem.indexOfScalar(u8, addr, '-') orelse return false;
    _ = std.fmt.parseInt(u64, addr[0..dash], 16) catch return false;
    _ = std.fmt.parseInt(u64, addr[dash + 1 ..], 16) catch return false;
    return true;
}

fn parseMappingName(line: []const u8) []const u8 {
    var idx: usize = 0;
    var fields: usize = 0;
    while (idx < line.len and fields < 5) {
        while (idx < line.len and (line[idx] == ' ' or line[idx] == '\t')) : (idx += 1) {}
        if (idx >= line.len) break;
        while (idx < line.len and line[idx] != ' ' and line[idx] != '\t') : (idx += 1) {}
        fields += 1;
    }
    const rest = std.mem.trimLeft(u8, line[idx..], " \t");
    if (rest.len == 0) return "[anonymous]";
    return rest;
}

fn copyName(dst: *[256]u8, src: []const u8) usize {
    const n = @min(dst.len - 1, src.len);
    @memcpy(dst[0..n], src[0..n]);
    dst[n] = 0;
    return n;
}

fn mappingLess(_: void, lhs: MappingStat, rhs: MappingStat) bool {
    if (lhs.private_dirty_kb != rhs.private_dirty_kb) return lhs.private_dirty_kb > rhs.private_dirty_kb;
    return lhs.rss_kb > rhs.rss_kb;
}

fn runEmptyFrameInference(allocator: std.mem.Allocator, model_path: []const u8, use_cuda: bool) !void {
    const base = c.OrtGetApiBase() orelse return error.OnnxRuntimeUnavailable;
    const get_api = base.*.GetApi orelse return error.OnnxRuntimeUnavailable;
    const api_c = get_api(c.ORT_API_VERSION) orelse return error.OnnxRuntimeUnavailable;
    const api: *const c.OrtApi = @ptrCast(api_c);

    var env: ?*c.OrtEnv = null;
    try ortCheck(api, api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "video-yolo-zig", &env));
    defer if (env) |v| api.ReleaseEnv.?(v);

    var session_options: ?*c.OrtSessionOptions = null;
    try ortCheck(api, api.CreateSessionOptions.?(&session_options));
    defer if (session_options) |v| api.ReleaseSessionOptions.?(v);

    try ortCheck(api, api.SetInterOpNumThreads.?(session_options, 1));
    try ortCheck(api, api.SetIntraOpNumThreads.?(session_options, 1));
    try ortCheck(api, api.SetSessionGraphOptimizationLevel.?(session_options, c.ORT_DISABLE_ALL));
    try ortCheck(api, api.DisableCpuMemArena.?(session_options));
    try ortCheck(api, api.DisableMemPattern.?(session_options));
    try ortCheck(api, api.AddSessionConfigEntry.?(session_options, "session.enable_mem_reuse", "0"));
    try ortCheck(api, api.AddSessionConfigEntry.?(session_options, "session.enable_mem_pattern", "0"));

    var cuda_options: ?*c.OrtCUDAProviderOptionsV2 = null;
    defer if (cuda_options != null and api.ReleaseCUDAProviderOptions != null) api.ReleaseCUDAProviderOptions.?(cuda_options);
    if (use_cuda) {
        const create_cuda = api.CreateCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
        const update_cuda = api.UpdateCUDAProviderOptions orelse return error.CudaExecutionProviderUnavailable;
        const append_cuda = api.SessionOptionsAppendExecutionProvider_CUDA_V2 orelse return error.CudaExecutionProviderUnavailable;
        try ortCheck(api, create_cuda(&cuda_options));

        const keys = [_][*c]const u8{
            "device_id",
        };
        const values = [_][*c]const u8{
            "0",
        };
        try ortCheck(api, update_cuda(cuda_options, &keys, &values, keys.len));
        try ortCheck(api, append_cuda(session_options, cuda_options));
    }

    const model_path_z = try allocator.dupeZ(u8, model_path);
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

    const input_tensor_len = 1 * 3 * 640 * 640;
    const input_data = try allocator.alloc(f32, input_tensor_len);
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
    var min_ms: f64 = 0;
    var max_ms: f64 = 0;

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
        const run_ms = @as(f64, @floatFromInt(end_ns - start_ns)) / @as(f64, std.time.ns_per_ms);

        if (i >= warmup_iters) {
            if (i == warmup_iters or run_ms < min_ms) min_ms = run_ms;
            if (i == warmup_iters or run_ms > max_ms) max_ms = run_ms;
            total_ms += run_ms;
        }
    }

    std.debug.print("ep: {s}\n", .{if (use_cuda) "cuda" else "cpu"});
    std.debug.print("inference ok (empty frame x {d}, warmup {d})\n", .{ measured_iters, warmup_iters });
    std.debug.print("run total: {d:.3} ms\n", .{total_ms});
    std.debug.print("run avg: {d:.3} ms\n", .{total_ms / @as(f64, @floatFromInt(measured_iters))});
    std.debug.print("run min: {d:.3} ms\n", .{min_ms});
    std.debug.print("run max: {d:.3} ms\n", .{max_ms});
    std.debug.print("throughput: {d:.2} fps\n", .{(@as(f64, @floatFromInt(measured_iters)) * 1000.0) / total_ms});
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var use_cuda = true;
    var explicit_model_path: ?[]const u8 = null;

    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--ep=")) {
            const ep = arg["--ep=".len..];
            if (std.mem.eql(u8, ep, "cpu")) {
                use_cuda = false;
            } else if (std.mem.eql(u8, ep, "cuda")) {
                use_cuda = true;
            } else {
                std.debug.print("invalid ep: {s} (expected cpu or cuda)\n", .{ep});
                return error.InvalidArguments;
            }
            continue;
        }

        if (explicit_model_path == null) {
            explicit_model_path = arg;
            continue;
        }

        std.debug.print("usage: {s} [--ep=cpu|cuda] [model.onnx]\n", .{args[0]});
        return error.InvalidArguments;
    }

    const model_path = if (explicit_model_path) |path| path else blk: {
        const exe_dir = try std.fs.selfExeDirPathAlloc(allocator);
        defer allocator.free(exe_dir);
        break :blk try std.fmt.allocPrint(allocator, "{s}/model.onnx", .{exe_dir});
    };
    defer if (explicit_model_path == null) allocator.free(model_path);

    try runEmptyFrameInference(allocator, model_path, use_cuda);
}
