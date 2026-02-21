const std = @import("std");
const Allocator = std.mem.Allocator;
const yolo = @import("yolo.zig");
const image_decode = @import("image_decode.zig");
const image_preprocess = @import("image_preprocess.zig");

pub const Detection = yolo.Detection;
pub const DetectProfile = struct {
    preprocess_ns: i128 = 0,
    run_ns: i128 = 0,
    decode_ns: i128 = 0,
};

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const OutputSpec = struct {
    boxes: usize,
    classes: usize,
    layout: yolo.OutputLayout,
};

fn inferOutputSpec(dim_count: usize, dims: []const i64, total_values: usize) OutputSpec {
    if (dim_count >= 3) {
        const d1 = @as(usize, @intCast(dims[dim_count - 2]));
        const d2 = @as(usize, @intCast(dims[dim_count - 1]));
        // Typical YOLOv8 outputs:
        // [1, 8400, 84] => boxes_first
        // [1, 84, 8400] => attributes_first
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
    // Fallback: assume boxes_first with COCO classes.
    return .{
        .boxes = total_values / (80 + 4),
        .classes = 80,
        .layout = .boxes_first,
    };
}

pub const Runtime = struct {
    allocator: Allocator,
    api: *const c.OrtApi,
    env: ?*c.OrtEnv,
    session_options: ?*c.OrtSessionOptions,
    session: ?*c.OrtSession,
    input_memory_info: ?*c.OrtMemoryInfo,
    input_name: ?[:0]u8,
    output_name: ?[:0]u8,
    reusable_input_tensor: ?[]f32,

    pub const InitOptions = struct {
        use_cuda: bool = true,
        cuda_device_id: i32 = 0,
        require_cuda: bool = false,
    };

    pub fn init(allocator: Allocator, model_path: []const u8) !@This() {
        return initWithOptions(allocator, model_path, .{});
    }

    pub fn initWithOptions(allocator: Allocator, model_path: []const u8, options: InitOptions) !@This() {
        const base = c.OrtGetApiBase();
        if (base == null) return error.OnnxRuntimeUnavailable;
        const get_api = base.?.*.GetApi orelse return error.OnnxRuntimeUnavailable;
        const api = get_api(c.ORT_API_VERSION) orelse return error.OnnxRuntimeUnavailable;

        var runtime: @This() = .{
            .allocator = allocator,
            .api = api.?,
            .env = null,
            .session_options = null,
            .session = null,
            .input_memory_info = null,
            .input_name = null,
            .output_name = null,
            .reusable_input_tensor = null,
        };
        errdefer runtime.deinit();

        try runtime.check(runtime.api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "zig-wasm", &runtime.env));
        try runtime.check(runtime.api.CreateSessionOptions.?(&runtime.session_options));
        try runtime.configureExecutionProvider(options);

        const model_path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(model_path_z);
        try runtime.check(runtime.api.CreateSession.?(runtime.env, model_path_z.ptr, runtime.session_options, &runtime.session));
        try runtime.initInferenceCache();

        return runtime;
    }

    pub fn deinit(self: *@This()) void {
        if (self.reusable_input_tensor) |tensor| self.allocator.free(tensor);
        if (self.input_name) |name| self.allocator.free(name);
        if (self.output_name) |name| self.allocator.free(name);
        if (self.input_memory_info) |info| self.api.ReleaseMemoryInfo.?(info);
        if (self.session) |session| self.api.ReleaseSession.?(session);
        if (self.session_options) |session_options| self.api.ReleaseSessionOptions.?(session_options);
        if (self.env) |env| self.api.ReleaseEnv.?(env);
        self.* = undefined;
    }

    fn initInferenceCache(self: *@This()) !void {
        const input_tensor_len = 1 * 3 * 640 * 640;
        self.reusable_input_tensor = try self.allocator.alloc(f32, input_tensor_len);
        errdefer {
            if (self.reusable_input_tensor) |tensor| self.allocator.free(tensor);
            self.reusable_input_tensor = null;
        }

        try self.check(self.api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &self.input_memory_info));
        errdefer {
            if (self.input_memory_info) |info| self.api.ReleaseMemoryInfo.?(info);
            self.input_memory_info = null;
        }

        var allocator_ort: ?*c.OrtAllocator = null;
        try self.check(self.api.GetAllocatorWithDefaultOptions.?(&allocator_ort));

        var input_name_alloc: [*c]u8 = null;
        try self.check(self.api.SessionGetInputName.?(self.session, 0, allocator_ort, &input_name_alloc));
        defer _ = self.api.AllocatorFree.?(allocator_ort, input_name_alloc);
        self.input_name = try self.allocator.dupeZ(u8, std.mem.span(input_name_alloc));
        errdefer {
            if (self.input_name) |name| self.allocator.free(name);
            self.input_name = null;
        }

        var output_name_alloc: [*c]u8 = null;
        try self.check(self.api.SessionGetOutputName.?(self.session, 0, allocator_ort, &output_name_alloc));
        defer _ = self.api.AllocatorFree.?(allocator_ort, output_name_alloc);
        self.output_name = try self.allocator.dupeZ(u8, std.mem.span(output_name_alloc));
    }

    fn check(self: *@This(), status: ?*c.OrtStatus) !void {
        if (status == null) return;
        defer self.api.ReleaseStatus.?(status);
        const message = self.api.GetErrorMessage.?(status);
        if (message != null) {
            std.log.err("onnxruntime: {s}", .{std.mem.span(message)});
        }
        return error.OnnxRuntimeError;
    }

    fn configureExecutionProvider(self: *@This(), options: InitOptions) !void {
        if (!options.use_cuda) return;

        const create_cuda_options = self.api.CreateCUDAProviderOptions orelse {
            if (options.require_cuda) return error.CudaExecutionProviderUnavailable;
            std.log.warn("onnxruntime: CUDA provider API unavailable, falling back to CPU", .{});
            return;
        };
        const update_cuda_options = self.api.UpdateCUDAProviderOptions orelse {
            if (options.require_cuda) return error.CudaExecutionProviderUnavailable;
            std.log.warn("onnxruntime: CUDA provider options API unavailable, falling back to CPU", .{});
            return;
        };
        const release_cuda_options = self.api.ReleaseCUDAProviderOptions orelse {
            if (options.require_cuda) return error.CudaExecutionProviderUnavailable;
            std.log.warn("onnxruntime: CUDA provider release API unavailable, falling back to CPU", .{});
            return;
        };
        const append_cuda = self.api.SessionOptionsAppendExecutionProvider_CUDA_V2 orelse {
            if (options.require_cuda) return error.CudaExecutionProviderUnavailable;
            std.log.warn("onnxruntime: CUDA append API unavailable, falling back to CPU", .{});
            return;
        };

        var cuda_options: ?*c.OrtCUDAProviderOptionsV2 = null;
        try self.check(create_cuda_options(&cuda_options));
        defer release_cuda_options(cuda_options);

        var device_id_buf: [16]u8 = undefined;
        const device_id_value = try std.fmt.bufPrintZ(&device_id_buf, "{d}", .{options.cuda_device_id});
        const keys = [_][*c]const u8{"device_id"};
        const values = [_][*c]const u8{device_id_value.ptr};
        try self.check(update_cuda_options(cuda_options, &keys, &values, 1));

        const cuda_status = append_cuda(self.session_options, cuda_options);
        if (cuda_status == null) {
            std.log.info("onnxruntime: CUDA execution provider enabled (device {d})", .{options.cuda_device_id});
            return;
        }

        defer self.api.ReleaseStatus.?(cuda_status);
        const msg = self.api.GetErrorMessage.?(cuda_status);
        const error_message = if (msg != null) std.mem.span(msg) else "unknown error";
        if (options.require_cuda) {
            std.log.err("onnxruntime: CUDA execution provider required but unavailable: {s}", .{error_message});
            return error.CudaExecutionProviderUnavailable;
        }

        std.log.warn("onnxruntime: CUDA execution provider unavailable, falling back to CPU: {s}", .{error_message});
    }

    pub fn decodeYoloOutput(
        _: *@This(),
        allocator: Allocator,
        tensor: []const f32,
        boxes: usize,
        classes: usize,
        layout: yolo.OutputLayout,
        conf_threshold: f32,
        iou_threshold: f32,
    ) ![]Detection {
        const decoded = try yolo.decodeV8(allocator, tensor, boxes, classes, layout, conf_threshold);
        defer allocator.free(decoded);
        return yolo.nms(allocator, decoded, iou_threshold);
    }

    pub fn freeDetections(_: *@This(), allocator: Allocator, detections: []Detection) void {
        allocator.free(detections);
    }

    pub fn detectFromImageBytes(self: *@This(), allocator: Allocator, image_bytes: []const u8) ![]Detection {
        const image = try image_decode.decodeRgb(allocator, image_bytes);
        defer allocator.free(image.rgb);

        return self.detectFromRgb(allocator, image.rgb, image.size);
    }

    pub fn detectFromRgb(
        self: *@This(),
        allocator: Allocator,
        rgb: []const u8,
        image_size: image_preprocess.ImageSize,
    ) ![]Detection {
        var profile: DetectProfile = .{};
        return self.detectFromRgbWithProfile(allocator, rgb, image_size, &profile);
    }

    pub fn detectFromRgbWithProfile(
        self: *@This(),
        allocator: Allocator,
        rgb: []const u8,
        image_size: image_preprocess.ImageSize,
        profile: *DetectProfile,
    ) ![]Detection {
        const input_size = image_preprocess.ImageSize{ .width = 640, .height = 640 };
        const input_tensor = self.reusable_input_tensor orelse return error.RuntimeNotInitialized;
        const preprocess_start_ns = std.time.nanoTimestamp();
        try image_preprocess.rgbU8ToNchwF32Into(input_tensor, rgb, image_size, input_size);
        profile.preprocess_ns = std.time.nanoTimestamp() - preprocess_start_ns;

        const run_start_ns = std.time.nanoTimestamp();
        const output = try self.runSingleInput(input_tensor);
        profile.run_ns = std.time.nanoTimestamp() - run_start_ns;
        defer self.api.ReleaseValue.?(output.output_value);

        const decode_start_ns = std.time.nanoTimestamp();
        const detections = try self.decodeYoloOutput(
            allocator,
            output.values,
            output.boxes,
            output.classes,
            output.layout,
            0.25,
            0.45,
        );
        profile.decode_ns = std.time.nanoTimestamp() - decode_start_ns;
        return detections;
    }

    const InferenceOutputView = struct {
        output_value: *c.OrtValue,
        values: []const f32,
        boxes: usize,
        classes: usize,
        layout: yolo.OutputLayout,
    };

    fn runSingleInput(self: *@This(), input: []f32) !InferenceOutputView {
        const memory_info = self.input_memory_info orelse return error.RuntimeNotInitialized;
        const input_name = self.input_name orelse return error.RuntimeNotInitialized;
        const output_name = self.output_name orelse return error.RuntimeNotInitialized;
        const shape = [_]i64{ 1, 3, 640, 640 };
        var input_value: ?*c.OrtValue = null;
        try self.check(self.api.CreateTensorWithDataAsOrtValue.?(
            memory_info,
            input.ptr,
            input.len * @sizeOf(f32),
            &shape,
            shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value,
        ));
        defer self.api.ReleaseValue.?(input_value);

        var output_value: ?*c.OrtValue = null;
        const input_names = [_][*c]const u8{input_name.ptr};
        const output_names = [_][*c]const u8{output_name.ptr};
        const input_values = [_]?*c.OrtValue{input_value};
        try self.check(self.api.Run.?(
            self.session,
            null,
            &input_names,
            &input_values,
            1,
            &output_names,
            1,
            &output_value,
        ));

        var tensor_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try self.check(self.api.GetTensorTypeAndShape.?(output_value, &tensor_info));
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(tensor_info);

        var dim_count: usize = 0;
        try self.check(self.api.GetDimensionsCount.?(tensor_info, &dim_count));
        var dims_buf: [8]i64 = undefined;
        if (dim_count > dims_buf.len) return error.UnsupportedOutputRank;
        try self.check(self.api.GetDimensions.?(tensor_info, &dims_buf, dim_count));
        const dims = dims_buf[0..dim_count];

        var out_ptr_raw: ?*anyopaque = null;
        try self.check(self.api.GetTensorMutableData.?(output_value, &out_ptr_raw));
        const out_ptr: [*]f32 = @ptrCast(@alignCast(out_ptr_raw));

        var total: usize = 1;
        for (dims) |d| total *= @as(usize, @intCast(d));

        const spec = inferOutputSpec(dim_count, dims, total);
        const output_value_nonnull = output_value orelse return error.OnnxRuntimeError;

        return .{
            .output_value = output_value_nonnull,
            .values = out_ptr[0..total],
            .boxes = spec.boxes,
            .classes = spec.classes,
            .layout = spec.layout,
        };
    }
};

test "inferOutputSpec yolo shapes" {
    const boxes_first_dims = [_]i64{ 1, 8400, 84 };
    const a = inferOutputSpec(boxes_first_dims.len, &boxes_first_dims, 1 * 8400 * 84);
    try std.testing.expectEqual(@as(usize, 8400), a.boxes);
    try std.testing.expectEqual(@as(usize, 80), a.classes);
    try std.testing.expectEqual(yolo.OutputLayout.boxes_first, a.layout);

    const attrs_first_dims = [_]i64{ 1, 84, 8400 };
    const b = inferOutputSpec(attrs_first_dims.len, &attrs_first_dims, 1 * 84 * 8400);
    try std.testing.expectEqual(@as(usize, 8400), b.boxes);
    try std.testing.expectEqual(@as(usize, 80), b.classes);
    try std.testing.expectEqual(yolo.OutputLayout.attributes_first, b.layout);
}
