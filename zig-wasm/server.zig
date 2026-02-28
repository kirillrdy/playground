const std = @import("std");
const httpz = @import("httpz");
const builtin = @import("builtin");
const onnxruntime = @import("onnxruntime.zig");
const video_yolo = @import("video_yolo.zig");
const Allocator = std.mem.Allocator;
const string = []const u8;
const c = @import("c.zig").c;

const print = std.log.info;
const port = 3000;
const uploads_dir = "uploads";
const processed_dir = "processed";
const max_range_response_bytes: u64 = 8 * 1024 * 1024;
const coco_class_names = [_][]const u8{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

fn detectionClassName(class_id: usize) []const u8 {
    if (class_id >= coco_class_names.len) return "unknown";
    return coco_class_names[class_id];
}

const App = struct {
    allocator: Allocator,
    detector: ?onnxruntime.Runtime,
    processing_mutex: std.Thread.Mutex,

    fn init(allocator: Allocator) !App {
        std.fs.cwd().makePath(uploads_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        std.fs.cwd().makePath(processed_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        var detector: ?onnxruntime.Runtime = null;
        const exe_dir = try std.fs.selfExeDirPathAlloc(allocator);
        defer allocator.free(exe_dir);
        const model_path = try std.fmt.allocPrint(allocator, "{s}/model.onnx", .{exe_dir});
        defer allocator.free(model_path);

        if (onnxruntime.Runtime.init(allocator, model_path)) |runtime| {
            detector = runtime;
            print("onnxruntime detector initialized with model: {s}", .{model_path});
        } else |err| {
            std.log.err("failed to initialize onnxruntime session with default model path {s}: {}", .{ model_path, err });
        }

        return .{
            .allocator = allocator,
            .detector = detector,
            .processing_mutex = .{},
        };
    }

    fn deinit(self: *App) void {
        if (self.detector) |*detector| detector.deinit();
    }
};

const UploadListRecord = struct {
    id: string,
    filename: string,
};

fn allUploadNames(allocator: Allocator) ![]UploadListRecord {
    var dir = try std.fs.cwd().openDir(uploads_dir, .{ .iterate = true });
    defer dir.close();

    var files: std.ArrayList(UploadListRecord) = .empty;
    errdefer {
        for (files.items) |file| allocator.free(file.filename);
        files.deinit(allocator);
    }

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind != .file) continue;
        const name_copy = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name_copy);
        try files.append(allocator, .{
            .id = name_copy,
            .filename = try allocator.dupe(u8, entry.name),
        });
    }

    return files.toOwnedSlice(allocator);
}

fn freeUploadNames(allocator: Allocator, files: []UploadListRecord) void {
    for (files) |file| {
        allocator.free(file.id);
        allocator.free(file.filename);
    }
    allocator.free(files);
}

const templates = struct {
    const Error = error{
        SectionNotFound,
        MissingTemplateArg,
    };

    fn parseSectionName(line: []const u8) ?[]const u8 {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len < 2 or trimmed[0] != '.') return null;
        const name = std.mem.trim(u8, trimmed[1..], " \t");
        if (name.len == 0) return null;
        return name;
    }

    fn findHeaderEnd(template: string) usize {
        var index: usize = 0;
        while (index < template.len) {
            const line_start = index;
            const newline_index = std.mem.indexOfScalarPos(u8, template, index, '\n') orelse template.len;
            const line = template[line_start..newline_index];
            if (parseSectionName(line) != null) return line_start;
            index = if (newline_index == template.len) template.len else newline_index + 1;
        }
        return template.len;
    }

    fn sectionRange(template: string, section: string) ?struct { start: usize, end: usize } {
        var index: usize = 0;
        var section_start: ?usize = null;

        while (index < template.len) {
            const line_start = index;
            const newline_index = std.mem.indexOfScalarPos(u8, template, index, '\n') orelse template.len;
            const line = template[line_start..newline_index];

            if (parseSectionName(line)) |name| {
                if (section_start != null) return .{ .start = section_start.?, .end = line_start };
                if (std.mem.eql(u8, name, section)) {
                    section_start = if (newline_index == template.len) template.len else newline_index + 1;
                }
            }
            index = if (newline_index == template.len) template.len else newline_index + 1;
        }

        if (section_start) |start| return .{ .start = start, .end = template.len };
        return null;
    }

    fn writeArg(writer: anytype, args: anytype, name: []const u8, specifier: []const u8) !bool {
        const Args = @TypeOf(args);
        const info = @typeInfo(Args);
        if (info != .@"struct") @compileError("template args must be a struct");

        inline for (info.@"struct".fields) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                const value = @field(args, field.name);
                if (std.mem.eql(u8, specifier, "s")) {
                    switch (@typeInfo(@TypeOf(value))) {
                        .pointer => |ptr| {
                            switch (ptr.size) {
                                .slice => {
                                    if (ptr.child == u8) {
                                        try writer.writeAll(value);
                                    } else {
                                        try writer.print("{any}", .{value});
                                    }
                                },
                                .many, .c => {
                                    if (ptr.child == u8) {
                                        try writer.writeAll(std.mem.span(value));
                                    } else {
                                        try writer.print("{any}", .{value});
                                    }
                                },
                                .one => {
                                    switch (@typeInfo(ptr.child)) {
                                        .array => |arr| {
                                            if (arr.child == u8) {
                                                try writer.writeAll(value[0..]);
                                            } else {
                                                try writer.print("{any}", .{value});
                                            }
                                        },
                                        else => try writer.print("{any}", .{value}),
                                    }
                                },
                            }
                        },
                        .array => |arr| {
                            if (arr.child == u8) {
                                try writer.writeAll(value[0..]);
                            } else {
                                try writer.print("{any}", .{value});
                            }
                        },
                        else => try writer.print("{any}", .{value}),
                    }
                } else if (std.mem.eql(u8, specifier, "d")) {
                    switch (@typeInfo(@TypeOf(value))) {
                        .int, .comptime_int => try writer.print("{d}", .{value}),
                        else => try writer.print("{any}", .{value}),
                    }
                } else {
                    try writer.print("{any}", .{value});
                }
                return true;
            }
        }
        return false;
    }

    fn renderWithArgs(writer: anytype, template: string, args: anytype) !void {
        var literal_start: usize = 0;
        var index: usize = 0;

        while (index < template.len) : (index += 1) {
            if (template[index] != '{' or index + 1 >= template.len or template[index + 1] != '[') continue;

            const close_bracket = std.mem.indexOfScalarPos(u8, template, index + 2, ']') orelse continue;
            const close_brace = std.mem.indexOfScalarPos(u8, template, close_bracket + 1, '}') orelse continue;

            const name = template[index + 2 .. close_bracket];
            const specifier = template[close_bracket + 1 .. close_brace];
            if (!std.mem.eql(u8, specifier, "s") and !std.mem.eql(u8, specifier, "d") and !std.mem.eql(u8, specifier, "any")) continue;

            try writer.writeAll(template[literal_start..index]);
            if (!try writeArg(writer, args, name, specifier)) return Error.MissingTemplateArg;

            index = close_brace;
            literal_start = index + 1;
        }

        if (literal_start < template.len) {
            try writer.writeAll(template[literal_start..]);
        }
    }

    fn writeHeader(template: string, writer: anytype) !void {
        try writer.writeAll(template[0..findHeaderEnd(template)]);
    }

    fn printHeader(template: string, args: anytype, writer: anytype) !void {
        try renderWithArgs(writer, template[0..findHeaderEnd(template)], args);
    }

    fn write(template: string, section: string, writer: anytype) !void {
        const range = sectionRange(template, section) orelse return Error.SectionNotFound;
        try writer.writeAll(template[range.start..range.end]);
    }

    fn print(template: string, section: string, args: anytype, writer: anytype) !void {
        const range = sectionRange(template, section) orelse return Error.SectionNotFound;
        try renderWithArgs(writer, template[range.start..range.end], args);
    }
};

pub const wasm_app_name = "main";
pub const server_name = "main";

pub const file_names = struct {
    const wasm = wasm_app_name ++ ".wasm";
    const wasm_js = "wasm.js";
};

fn resource(name: string) type {
    return struct {
        const index = "/" ++ name;
        const new = "/" ++ name ++ "/new";
        const show = "/" ++ name ++ "/:id";
    };
}

const Paths = struct {
    const files = resource("files");
    const root = "/";
    const wasm = "/wasm";
    const wasm_file = "/" ++ file_names.wasm;
    const wasm_js_file = "/" ++ file_names.wasm_js;
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();

    var app = try App.init(allocator);
    defer app.deinit();

    var server = try httpz.Server(@TypeOf(&app)).init(allocator, .{
        .port = port,
        .request = .{
            .max_multiform_count = 8,
            .max_body_size = 1024 * 1024 * 1024,
        },
    }, &app);
    defer {
        print("shutting down httpz", .{});
        server.stop();
        server.deinit();
    }

    var router = try server.router(.{});
    router.get(Paths.files.index, makeHandlerWithOptions("Files/index", .{ .default_template = false }), .{});
    router.get(Paths.files.new, makeHandler("Files/new"), .{});
    router.get(Paths.files.show, Files.show, .{});
    router.post(Paths.files.index, Files.create, .{});
    router.get(Paths.root, makeHandler("handlers/home"), .{});
    router.get(Paths.wasm, makeHandler("handlers/wasm"), .{});
    router.get(Paths.wasm_js_file, wasmJsFile, .{});
    router.get(Paths.wasm_file, wasmFile, .{});
    router.get("/uploads/:name", uploadFileHandler, .{});
    router.get("/processed/:name", processedFileHandler, .{});
    router.get("/favicon.ico", faviconHandler, .{});

    print("processor model: {s}", .{builtin.cpu.model.name});
    print("listening on :{d}", .{port});
    try server.listen();
}

const httpzHandler = fn (*App, *httpz.Request, *httpz.Response) anyerror!void;

const responseOptions = struct {
    default_template: bool = true,
};

fn removeUndescores(comptime input: string) [input.len]u8 {
    var lower: [input.len]u8 = undefined;
    for (input, 0..) |char, index| {
        if (char == '_') {
            lower[index] = ' ';
        } else {
            lower[index] = char;
        }
    }
    return lower;
}
fn lowerString(comptime input: string) [input.len]u8 {
    var lower: [input.len]u8 = undefined;
    for (input, 0..) |char, index| {
        if (index == 0) {
            lower[index] = std.ascii.toLower(char);
        } else {
            lower[index] = char;
        }
    }
    return lower;
}

fn responseType(name: string, options: responseOptions) type {
    const lower = lowerString(name);
    const template_path = "views/" ++ lower ++ ".html";

    return struct {
        comptime template: string = if (options.default_template) @embedFile(template_path) else "",
        out: *std.io.Writer,
        app: *App,
        fn print(self: @This(), comptime section: string, args: anytype) !void {
            try templates.print(self.template, section, args, self.out);
        }

        fn printHeader(self: @This(), args: anytype) !void {
            try templates.printHeader(self.template, args, self.out);
        }

        fn writeHeader(self: @This()) !void {
            try templates.writeHeader(self.template, self.out);
        }

        fn write(self: @This(), section: string) !void {
            try templates.write(self.template, section, self.out);
        }
    };
}

fn makeHandler(name: string) httpzHandler {
    return makeHandlerWithOptions(name, .{});
}
fn makeHandlerWithOptions(name: string, options: responseOptions) httpzHandler {
    var it = std.mem.splitScalar(u8, name, '/');
    const space = it.next().?;
    const field = it.next().?;

    const func = @field(@field(@This(), space), field);
    return struct {
        fn handler(app: *App, _: *httpz.Request, httpzResponse: *httpz.Response) !void {
            var writerAllocating = std.Io.Writer.Allocating.init(httpzResponse.arena);
            const writer = &writerAllocating.writer;
            const layout = @embedFile("views/layout.html");
            try templates.writeHeader(layout, writer);
            try templates.print(layout, "title", .{ .title = space }, writer);
            const response = responseType(name, options){ .out = writer, .app = app };
            try func(response);
            try templates.write(layout, "close-body", writer);
            httpzResponse.body = writerAllocating.written();
        }
    }.handler;
}
test "tolower" {
    const input = "Kirill";
    const output = lowerString(input);
    _ = lowerString("somethingverylong");
    try std.testing.expectEqualStrings("kirill", &output);
}

test "http handlers smoke" {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    errdefer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var app = try App.init(allocator);
    defer app.deinit();

    const test_upload_name = "__test_handler_file.txt";
    const upload_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ uploads_dir, test_upload_name });
    defer allocator.free(upload_path);
    {
        var f = try std.fs.cwd().createFile(upload_path, .{ .truncate = true });
        defer f.close();
        try f.writeAll("0123456789");
    }
    defer std.fs.cwd().deleteFile(upload_path) catch {};

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        const handler = makeHandlerWithOptions("Files/index", .{ .default_template = false });
        try handler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
        const body = try ht.getBody();
        try std.testing.expect(std.mem.indexOf(u8, body, test_upload_name) != null);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        const handler = makeHandler("Files/new");
        try handler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        ht.param("id", test_upload_name);
        try Files.show(&app, ht.req, ht.res);
        try ht.expectStatus(200);
        const body = try ht.getBody();
        try std.testing.expect(std.mem.indexOf(u8, body, "No stored detections for this file.") != null);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        ht.param("name", test_upload_name);
        try uploadFileHandler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
        try ht.expectBody("0123456789");
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        ht.param("name", test_upload_name);
        ht.header("Range", "bytes=2-5");
        try uploadFileHandler(&app, ht.req, ht.res);
        try ht.expectStatus(206);
        try ht.expectBody("2345");
        try ht.expectHeader("Content-Range", "bytes 2-5/10");
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        ht.param("name", "__missing__.jsonl");
        try processedFileHandler(&app, ht.req, ht.res);
        try ht.expectStatus(404);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        const handler = makeHandler("handlers/home");
        try handler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        const handler = makeHandler("handlers/wasm");
        try handler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        try wasmJsFile(&app, ht.req, ht.res);
        try ht.expectStatus(200);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        try faviconHandler(&app, ht.req, ht.res);
        try ht.expectStatus(200);
    }

    {
        var ht = httpz.testing.init(.{});
        defer ht.deinit();
        try Files.create(&app, ht.req, ht.res);
        try ht.expectStatus(400);
    }

    try std.testing.expect(gpa.deinit() == .ok);
}

const DetectionRecord = struct {
    class_id: usize,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
};

fn table(writer: *std.io.Writer, title: string, records: anytype) !void {
    const resultRowType = @typeInfo(@TypeOf(records)).pointer.child;
    const template = @embedFile("views/table.html");
    try templates.printHeader(template, .{ .title = title }, writer);
    const query_fields = @typeInfo(resultRowType).@"struct".fields;

    inline for (query_fields) |field| {
        if (!std.mem.startsWith(u8, field.name, "_")) {
            try templates.print(template, "th", .{ .value = removeUndescores(field.name) }, writer);
        }
    }
    try templates.write(template, "th-close", writer);

    for (records) |file| {
        try templates.write(template, "tr", writer);
        inline for (query_fields) |field| {
            if (comptime !std.mem.startsWith(u8, field.name, "_")) {
                const value = @field(file, field.name);

                switch (@TypeOf(value)) {
                    usize => {
                        if (comptime std.mem.eql(u8, field.name, "class_id")) {
                            try templates.print(template, "td-string", .{ .value = detectionClassName(value) }, writer);
                        } else {
                            try templates.print(template, "td-number", .{ .value = value }, writer);
                        }
                    },
                    i32, i64 => try templates.print(template, "td-number", .{ .value = value }, writer),
                    f32 => try templates.print(template, "td", .{ .value = value }, writer),
                    string => {
                        if (comptime std.mem.eql(u8, field.name, "filename")) {
                            if (comptime @hasField(resultRowType, "id")) {
                                if (std.mem.eql(u8, title, "Files") or std.mem.eql(u8, title, "Wasm app")) {
                                    const id_value = @field(file, "id");
                                    try writer.print(
                                        \\<td class="px-6 py-4"><a class="text-blue-600 hover:underline" href="/files/{s}">{s}</a></td>
                                    , .{ id_value, value });
                                } else {
                                    try templates.print(template, "td-string", .{ .value = value }, writer);
                                }
                            } else {
                                try templates.print(template, "td-string", .{ .value = value }, writer);
                            }
                        } else {
                            try templates.print(template, "td-string", .{ .value = value }, writer);
                        }
                    },
                    else => @compileError("what type is this " ++ @typeName(@TypeOf(value))),
                }
            }
        }
        try templates.write(template, "tr-close", writer);
    }
    try templates.write(template, "tbody-close", writer);
}

const Files = struct {
    fn index(response: anytype) !void {
        const files = try allUploadNames(response.app.allocator);
        defer freeUploadNames(response.app.allocator, files);
        try table(response.out, "Files", files);
    }
    fn new(response: anytype) !void {
        try response.writeHeader();
    }

    fn show(app: *App, request: *httpz.Request, response: *httpz.Response) !void {
        var writerAllocating = std.Io.Writer.Allocating.init(response.arena);
        const writer = &writerAllocating.writer;
        const layout = @embedFile("views/layout.html");
        try templates.writeHeader(layout, writer);
        try templates.print(layout, "title", .{ .title = "Files" }, writer);

        const id_raw = request.param("id") orelse {
            response.status = 400;
            try writer.writeAll("missing id");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        };
        const id_str = try decodeUrlComponent(app.allocator, id_raw);
        defer app.allocator.free(id_str);

        if (!isSafeAssetName(id_str)) {
            response.status = 400;
            try writer.writeAll("invalid id");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        }

        const upload_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ uploads_dir, id_str });
        defer app.allocator.free(upload_path);
        std.fs.cwd().access(upload_path, .{}) catch {
            response.status = 404;
            try writer.writeAll("file not found");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        };

        const template = @embedFile("views/files/show.html");
        try templates.printHeader(template, .{ .file_id = id_str, .file_name = id_str }, writer);

        if (isVideoFileName(id_str)) {
            const detections_name = try std.fmt.allocPrint(app.allocator, "{s}.jsonl", .{id_str});
            defer app.allocator.free(detections_name);

            try writer.writeAll(
                \\<div class="relative mx-4 mt-4 inline-block">
                \\  <video id="video-player" class="max-w-full rounded-lg border border-slate-300" controls src="/uploads/
            );
            try writer.print("{s}", .{id_str});
            try writer.writeAll(
                \\"></video>
                \\  <canvas id="video-overlay" class="absolute left-0 top-0 pointer-events-none"></canvas>
                \\</div>
                \\<script src="/wasm.js"></script>
                \\<script>
                \\(async function() {
                \\  const video = document.getElementById('video-player');
                \\  const canvas = document.getElementById('video-overlay');
                \\  const detectionsUrl = '/processed/
            );
            try writer.print("{s}", .{detections_name});
            try writer.writeAll(
                \\';
                \\  await window.videoOverlayWasm.attachVideoOverlay(video, canvas, detectionsUrl);
                \\})();
                \\</script>
            );
        } else {
            const detections_name = try std.fmt.allocPrint(app.allocator, "{s}.jsonl", .{id_str});
            defer app.allocator.free(detections_name);
            const detections_rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ processed_dir, detections_name });
            defer app.allocator.free(detections_rel_path);

            if (readDetectionsJsonl(app.allocator, detections_rel_path)) |detections| {
                defer app.allocator.free(detections);
                try table(writer, "Detections", detections);
            } else |err| {
                if (err != error.FileNotFound) std.log.err("failed to read detections: {}", .{err});
                try writer.writeAll("<div class=\"mx-4 mt-4 text-slate-600\">No stored detections for this file.</div>");
            }
        }

        try templates.write(layout, "close-body", writer);
        response.body = writerAllocating.written();
    }

    fn create(app: *App, request: *httpz.Request, response: *httpz.Response) !void {
        const form = request.multiFormData() catch |err| {
            print("multipart parse failed: {}", .{err});
            response.status = 400;
            response.body = "invalid multipart form data";
            return;
        };
        var iterator = form.iterator();
        var has_file = false;
        while (iterator.next()) |kv| {
            if (std.mem.eql(u8, kv.key, "file")) {
                has_file = true;
                const incoming_name = kv.value.filename orelse "upload.bin";
                const safe_name = try sanitizeFilename(app.allocator, incoming_name);
                defer app.allocator.free(safe_name);

                const rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ uploads_dir, safe_name });
                defer app.allocator.free(rel_path);

                var file = try std.fs.cwd().createFile(rel_path, .{ .truncate = true });
                defer file.close();
                try file.writeAll(kv.value.value);
                print("saved upload to {s}", .{rel_path});

                if (app.detector != null and !isVideoFileName(safe_name)) {
                    var detector = &app.detector.?;
                    const detections = detector.detectFromImageBytes(app.allocator, kv.value.value) catch |err| {
                        print("detection failed for {s}: {}", .{ safe_name, err });
                        continue;
                    };
                    defer detector.freeDetections(app.allocator, detections);

                    const detections_name = try std.fmt.allocPrint(app.allocator, "{s}.jsonl", .{safe_name});
                    defer app.allocator.free(detections_name);
                    const detections_rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ processed_dir, detections_name });
                    defer app.allocator.free(detections_rel_path);

                    writeDetectionsJsonl(detections_rel_path, detections) catch |err| {
                        print("failed to persist detections for {s}: {}", .{ safe_name, err });
                    };
                }
            }
        }
        if (has_file) {
            response.status = 303;
            response.header("Location", Paths.files.index);
            response.body = "redirecting";
        } else {
            response.status = 400;
            response.body = "missing file";
        }
    }
};

fn writeDetectionsJsonl(path: []const u8, detections: []const onnxruntime.Detection) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.heap.c_allocator);
    const writer = buf.writer(std.heap.c_allocator);

    try writer.print("{{\"frame\":0,\"detections\":[", .{});
    for (detections, 0..) |d, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.print(
            "{{\"class_id\":{d},\"score\":{d:.4},\"x1\":{d:.6},\"y1\":{d:.6},\"x2\":{d:.6},\"y2\":{d:.6}}}",
            .{ d.class_id, d.score, d.x1, d.y1, d.x2, d.y2 },
        );
    }
    try writer.writeAll("]}\n");
    try file.writeAll(buf.items);
}

fn readDetectionsJsonl(allocator: Allocator, path: []const u8) ![]DetectionRecord {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(data);

    var list: std.ArrayList(DetectionRecord) = .empty;
    errdefer list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, data, '\n');
    const first_line = lines.next() orelse return error.EndOfStream;
    
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, first_line, .{});
    defer parsed.deinit();

    const detections_val = parsed.value.object.get("detections") orelse return error.InvalidFormat;
    for (detections_val.array.items) |item| {
        try list.append(allocator, .{
            .class_id = @intCast(item.object.get("class_id").?.integer),
            .score = @floatCast(item.object.get("score").?.float),
            .x1 = @floatCast(item.object.get("x1").?.float),
            .y1 = @floatCast(item.object.get("y1").?.float),
            .x2 = @floatCast(item.object.get("x2").?.float),
            .y2 = @floatCast(item.object.get("y2").?.float),
        });
    }
    return list.toOwnedSlice(allocator);
}

const handlers = struct {
    fn home(response: anytype) !void {
        try response.writeHeader();
    }

    fn wasm(response: anytype) !void {
        const all_files = try allUploadNames(response.app.allocator);
        defer freeUploadNames(response.app.allocator, all_files);

        var video_files: std.ArrayList(UploadListRecord) = .empty;
        errdefer {
            for (video_files.items) |file| {
                response.app.allocator.free(file.id);
                response.app.allocator.free(file.filename);
            }
            video_files.deinit(response.app.allocator);
        }

        for (all_files) |file| {
            if (isVideoFileName(file.filename)) {
                try video_files.append(response.app.allocator, .{
                    .id = try response.app.allocator.dupe(u8, file.id),
                    .filename = try response.app.allocator.dupe(u8, file.filename),
                });
            }
        }
        defer {
            for (video_files.items) |file| {
                response.app.allocator.free(file.id);
                response.app.allocator.free(file.filename);
            }
            video_files.deinit(response.app.allocator);
        }

        try table(response.out, "Wasm app", video_files.items);
    }
};

fn sanitizeFilename(allocator: Allocator, name: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (name) |ch| {
        const safe = switch (ch) {
            'a'...'z', 'A'...'Z', '0'...'9', '.', '-', '_' => ch,
            else => '_',
        };
        try out.append(allocator, safe);
    }
    if (out.items.len == 0) try out.append(allocator, '_');
    return out.toOwnedSlice(allocator);
}

fn ensureVideoDetections(app: *App, stored_upload_name: []const u8) ![]u8 {
    app.processing_mutex.lock();
    defer app.processing_mutex.unlock();

    const detections_name = try std.fmt.allocPrint(app.allocator, "{s}.jsonl", .{stored_upload_name});
    errdefer app.allocator.free(detections_name);
    const detections_rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ processed_dir, detections_name });
    defer app.allocator.free(detections_rel_path);

    var need_generate = false;
    std.fs.cwd().access(detections_rel_path, .{}) catch {
        need_generate = true;
    };
    if (!need_generate) {
        if (!try isLikelyValidDetectionsJsonl(app.allocator, detections_rel_path)) {
            std.fs.cwd().deleteFile(detections_rel_path) catch {};
            need_generate = true;
        }
    }

    if (need_generate) {
        const upload_rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ uploads_dir, stored_upload_name });
        defer app.allocator.free(upload_rel_path);

        const tmp_rel_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}.tmp", .{ processed_dir, detections_name });
        defer app.allocator.free(tmp_rel_path);
        std.fs.cwd().deleteFile(tmp_rel_path) catch {};

        const shared = if (app.detector) |d| video_yolo.SharedInferenceResources{
            .api = d.api,
            .env = d.env.?,
            .session = d.session.?,
        } else null;

        video_yolo.inferVideoToJsonl(app.allocator, upload_rel_path, tmp_rel_path, .{}, shared) catch |err| {
            std.fs.cwd().deleteFile(tmp_rel_path) catch {};
            return err;
        };

        std.fs.cwd().rename(tmp_rel_path, detections_rel_path) catch |err| switch (err) {
            error.PathAlreadyExists => {
                std.fs.cwd().deleteFile(tmp_rel_path) catch {};
            },
            else => return err,
        };
    }

    return detections_name;
}

fn isLikelyValidDetectionsJsonl(allocator: Allocator, rel_path: []const u8) !bool {
    const data = std.fs.cwd().readFileAlloc(allocator, rel_path, 256 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer allocator.free(data);
    if (data.len == 0) return false;

    var saw_any = false;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        saw_any = true;
        if (!std.mem.startsWith(u8, line, "{\"frame\":")) return false;
        if (!std.mem.endsWith(u8, line, "}")) return false;
        if (std.mem.indexOf(u8, line, "\"detections\":") == null) return false;
    }
    return saw_any;
}

fn sendFile(allocator: Allocator, comptime name: string) !string {
    const dir = try std.fs.selfExeDirPathAlloc(allocator);
    const file_path = try std.fmt.allocPrint(allocator, "{s}/" ++ name, .{dir});
    const wasm_file = try std.fs.cwd().openFile(file_path, .{});
    defer wasm_file.close();

    const file_size = try wasm_file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    const bytesRead = try wasm_file.readAll(buffer);

    if (bytesRead != file_size) {
        return error.UnexpectedEndOfFile;
    }
    return buffer;
}

fn wasmJsFile(_: *App, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .JS;
    res.body = @embedFile("wasm.js");
}
fn wasmFile(_: *App, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .WASM;
    res.body = try sendFile(res.arena, file_names.wasm);
}
fn faviconHandler(_: *App, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .SVG;
    res.body = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\"></svg>";
}

fn isVideoFileName(name: []const u8) bool {
    return std.ascii.endsWithIgnoreCase(name, ".mp4") or
        std.ascii.endsWithIgnoreCase(name, ".mov") or
        std.ascii.endsWithIgnoreCase(name, ".m4v") or
        std.ascii.endsWithIgnoreCase(name, ".webm") or
        std.ascii.endsWithIgnoreCase(name, ".avi") or
        std.ascii.endsWithIgnoreCase(name, ".mkv");
}

fn isSafeAssetName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (std.mem.indexOfScalar(u8, name, '/')) |_| return false;
    if (std.mem.indexOfScalar(u8, name, '\\')) |_| return false;
    if (std.mem.indexOf(u8, name, "..")) |_| return false;
    return true;
}

fn assetContentType(name: []const u8) []const u8 {
    if (std.ascii.endsWithIgnoreCase(name, ".mp4")) return "video/mp4";
    if (std.ascii.endsWithIgnoreCase(name, ".mov")) return "video/quicktime";
    if (std.ascii.endsWithIgnoreCase(name, ".webm")) return "video/webm";
    if (std.ascii.endsWithIgnoreCase(name, ".jsonl")) return "application/json";
    return "application/octet-stream";
}

fn readAssetToArena(arena: Allocator, dir_name: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dir_name, name });
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const size = try file.getEndPos();
    const buffer = try arena.alloc(u8, size);
    const read = try file.readAll(buffer);
    if (read != size) return error.UnexpectedEndOfFile;
    return buffer;
}

fn decodeUrlComponent(allocator: Allocator, raw: []const u8) ![]u8 {
    const buf = try allocator.dupe(u8, raw);
    const decoded = std.Uri.percentDecodeInPlace(buf);
    if (decoded.ptr == buf.ptr and decoded.len == buf.len) return buf;
    const out = try allocator.dupe(u8, decoded);
    allocator.free(buf);
    return out;
}

const ByteRange = struct {
    start: u64,
    end: u64,
};

fn parseRangeHeader(value: []const u8, file_size: u64) ?ByteRange {
    if (!std.mem.startsWith(u8, value, "bytes=")) return null;
    const spec = value["bytes=".len..];
    if (spec.len == 0) return null;
    if (std.mem.indexOfScalar(u8, spec, ',')) |_| return null; // single range only

    const dash_idx = std.mem.indexOfScalar(u8, spec, '-') orelse return null;
    const start_part = spec[0..dash_idx];
    const end_part = spec[dash_idx + 1 ..];

    if (file_size == 0) return null;

    if (start_part.len == 0) {
        const suffix_len = std.fmt.parseInt(u64, end_part, 10) catch return null;
        if (suffix_len == 0) return null;
        if (suffix_len >= file_size) return .{ .start = 0, .end = file_size - 1 };
        return .{ .start = file_size - suffix_len, .end = file_size - 1 };
    }

    const start = std.fmt.parseInt(u64, start_part, 10) catch return null;
    if (start >= file_size) return null;
    if (end_part.len == 0) return .{ .start = start, .end = file_size - 1 };

    var end = std.fmt.parseInt(u64, end_part, 10) catch return null;
    if (end < start) return null;
    if (end >= file_size) end = file_size - 1;
    return .{ .start = start, .end = end };
}

fn invalidRange(res: *httpz.Response, file_size: u64) !void {
    res.status = 416;
    const content_range = try std.fmt.allocPrint(res.arena, "bytes */{d}", .{file_size});
    res.header("Content-Range", content_range);
    res.body = "range not satisfiable";
}

fn uploadFileHandler(_: *App, req: *httpz.Request, res: *httpz.Response) !void {
    const name_raw = req.param("name") orelse {
        res.status = 400;
        res.body = "missing file name";
        return;
    };
    const name = try decodeUrlComponent(res.arena, name_raw);
    if (!isSafeAssetName(name)) {
        res.status = 400;
        res.body = "invalid file name";
        return;
    }
    const path = try std.fmt.allocPrint(res.arena, "{s}/{s}", .{ uploads_dir, name });
    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            res.status = 404;
            res.body = "not found";
            return;
        },
        else => return err,
    };
    defer file.close();

    const size = try file.getEndPos();
    res.header("Accept-Ranges", "bytes");
    res.header("Content-Type", assetContentType(name));

    if (req.header("range")) |range_header| {
        var byte_range = parseRangeHeader(range_header, size) orelse {
            try invalidRange(res, size);
            return;
        };

        const max_end = byte_range.start +| (max_range_response_bytes - 1);
        if (byte_range.end > max_end) byte_range.end = max_end;
        const len_u64 = (byte_range.end - byte_range.start) + 1;
        const len = std.math.cast(usize, len_u64) orelse return error.FileTooBig;
        const buffer = try res.arena.alloc(u8, len);

        try file.seekTo(byte_range.start);
        const read = try file.readAll(buffer);
        if (read != len) return error.UnexpectedEndOfFile;

        res.status = 206;
        const content_range = try std.fmt.allocPrint(res.arena, "bytes {d}-{d}/{d}", .{
            byte_range.start,
            byte_range.end,
            size,
        });
        res.header("Content-Range", content_range);
        const content_len = try std.fmt.allocPrint(res.arena, "{d}", .{len});
        res.header("Content-Length", content_len);
        res.body = buffer;
        return;
    }

    const full_len = std.math.cast(usize, size) orelse return error.FileTooBig;
    const full = try res.arena.alloc(u8, full_len);
    const read = try file.readAll(full);
    if (read != full_len) return error.UnexpectedEndOfFile;
    res.body = full;
}

fn processedFileHandler(app: *App, req: *httpz.Request, res: *httpz.Response) !void {
    const name_raw = req.param("name") orelse {
        res.status = 400;
        res.body = "missing file name";
        return;
    };
    const name = try decodeUrlComponent(res.arena, name_raw);
    if (!isSafeAssetName(name)) {
        res.status = 400;
        res.body = "invalid file name";
        return;
    }

    if (std.ascii.endsWithIgnoreCase(name, ".jsonl")) {
        const upload_name = parseDetectionsSourceName(name) orelse {
            res.status = 400;
            res.body = "invalid detections file name";
            return;
        };

        const q_params = try req.query();
        const start_param = q_params.get("start");
        const end_param = q_params.get("end");

        if (start_param != null and end_param != null) {
            const start = std.fmt.parseFloat(f64, start_param.?) catch 0.0;
            const end = std.fmt.parseFloat(f64, end_param.?) catch (start + 2.0);
            
            const segment_name = try std.fmt.allocPrint(res.arena, "{s}.{d:.2}-{d:.2}.jsonl", .{upload_name, start, end});
            const segment_path = try std.fmt.allocPrint(res.arena, "{s}/{s}", .{processed_dir, segment_name});
            
            var need_gen = false;
            std.fs.cwd().access(segment_path, .{}) catch { need_gen = true; };
            
            if (need_gen) {
                const upload_path = try std.fmt.allocPrint(res.arena, "{s}/{s}", .{uploads_dir, upload_name});
                const tmp_path = try std.fmt.allocPrint(res.arena, "{s}.tmp", .{segment_path});
                
                const shared = if (app.detector) |d| video_yolo.SharedInferenceResources{
                    .api = d.api,
                    .env = d.env.?,
                    .session = d.session.?,
                } else null;

                video_yolo.inferVideoToJsonl(app.allocator, upload_path, tmp_path, .{
                    .start_s = start,
                    .duration_s = end - start,
                }, shared) catch |err| {
                    std.log.err("segment inference failed: {}", .{err});
                    res.status = 500;
                    res.body = "inference failed";
                    return;
                };
                try std.fs.cwd().rename(tmp_path, segment_path);
            }
            
            res.header("Content-Type", "application/json");
            res.body = readAssetToArena(res.arena, processed_dir, segment_name) catch |err| {
                std.log.err("failed to read segment: {}", .{err});
                res.status = 404;
                return;
            };
            return;
        }

        if (!isSafeAssetName(upload_name)) {
            res.status = 400;
            res.body = "invalid detections file name";
            return;
        }
        const upload_path = try std.fmt.allocPrint(app.allocator, "{s}/{s}", .{ uploads_dir, upload_name });
        defer app.allocator.free(upload_path);
        std.fs.cwd().access(upload_path, .{}) catch {
            res.status = 404;
            res.body = "not found";
            return;
        };
        _ = try ensureVideoDetections(app, upload_name);
    }

    res.header("Content-Type", assetContentType(name));
    res.body = readAssetToArena(res.arena, processed_dir, name) catch |err| switch (err) {
        error.FileNotFound => {
            res.status = 404;
            res.body = "not found";
            return;
        },
        else => return err,
    };
}

fn parseDetectionsSourceName(name: []const u8) ?[]const u8 {
    if (!std.ascii.endsWithIgnoreCase(name, ".jsonl")) return null;
    if (name.len <= ".jsonl".len) return null;
    return name[0 .. name.len - ".jsonl".len];
}
