const std = @import("std");
const Allocator = std.mem.Allocator;
const yolo = @import("yolo.zig");
const image_decode = @import("image_decode.zig");
const image_preprocess = @import("image_preprocess.zig");

pub const Detection = yolo.Detection;

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

    pub fn init(allocator: Allocator, model_path: []const u8) !@This() {
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
        };
        errdefer runtime.deinit();

        try runtime.check(runtime.api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "zig-wasm", &runtime.env));
        try runtime.check(runtime.api.CreateSessionOptions.?(&runtime.session_options));

        const model_path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(model_path_z);
        try runtime.check(runtime.api.CreateSession.?(runtime.env, model_path_z.ptr, runtime.session_options, &runtime.session));

        return runtime;
    }

    pub fn deinit(self: *@This()) void {
        if (self.session) |session| self.api.ReleaseSession.?(session);
        if (self.session_options) |session_options| self.api.ReleaseSessionOptions.?(session_options);
        if (self.env) |env| self.api.ReleaseEnv.?(env);
        self.* = undefined;
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

        const input_size = image_preprocess.ImageSize{ .width = 640, .height = 640 };
        const input_tensor = try image_preprocess.rgbU8ToNchwF32(allocator, image.rgb, image.size, input_size);
        defer allocator.free(input_tensor);

        const output = try self.runSingleInput(allocator, input_tensor, .{ 1, 3, 640, 640 });
        defer allocator.free(output.values);

        return self.decodeYoloOutput(
            allocator,
            output.values,
            output.boxes,
            output.classes,
            output.layout,
            0.25,
            0.45,
        );
    }

    const InferenceOutput = struct {
        values: []f32,
        boxes: usize,
        classes: usize,
        layout: yolo.OutputLayout,
    };

    fn runSingleInput(self: *@This(), allocator: Allocator, input: []f32, shape: [4]i64) !InferenceOutput {
        var memory_info: ?*c.OrtMemoryInfo = null;
        try self.check(self.api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info));
        defer self.api.ReleaseMemoryInfo.?(memory_info);

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

        var allocator_ort: ?*c.OrtAllocator = null;
        try self.check(self.api.GetAllocatorWithDefaultOptions.?(&allocator_ort));

        var input_name_alloc: [*c]u8 = null;
        try self.check(self.api.SessionGetInputName.?(self.session, 0, allocator_ort, &input_name_alloc));
        defer _ = self.api.AllocatorFree.?(allocator_ort, input_name_alloc);

        var output_name_alloc: [*c]u8 = null;
        try self.check(self.api.SessionGetOutputName.?(self.session, 0, allocator_ort, &output_name_alloc));
        defer _ = self.api.AllocatorFree.?(allocator_ort, output_name_alloc);

        var output_value: ?*c.OrtValue = null;
        const input_names = [_][*c]const u8{input_name_alloc};
        const output_names = [_][*c]const u8{output_name_alloc};
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
        defer self.api.ReleaseValue.?(output_value);

        var tensor_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try self.check(self.api.GetTensorTypeAndShape.?(output_value, &tensor_info));
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(tensor_info);

        var dim_count: usize = 0;
        try self.check(self.api.GetDimensionsCount.?(tensor_info, &dim_count));
        const dims = try allocator.alloc(i64, dim_count);
        defer allocator.free(dims);
        try self.check(self.api.GetDimensions.?(tensor_info, dims.ptr, dim_count));

        var out_ptr_raw: ?*anyopaque = null;
        try self.check(self.api.GetTensorMutableData.?(output_value, &out_ptr_raw));
        const out_ptr: [*]f32 = @ptrCast(@alignCast(out_ptr_raw));

        var total: usize = 1;
        for (dims) |d| total *= @as(usize, @intCast(d));
        const copied = try allocator.alloc(f32, total);
        @memcpy(copied, out_ptr[0..total]);

        const spec = inferOutputSpec(dim_count, dims, total);

        return .{
            .values = copied,
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
