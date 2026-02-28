const std = @import("std");
const httpz = @import("httpz");
const builtin = @import("builtin");
const onnxruntime = @import("onnxruntime.zig");
const video_yolo = @import("video_yolo.zig");
const Allocator = std.mem.Allocator;
const string = []const u8;
const c = @cImport({
    @cInclude("sqlite3.h");
});

const print = std.log.info;
const port = 3000;
const uploads_dir = "uploads";
const processed_dir = "processed";
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

fn detectionClassName(class_id: i32) []const u8 {
    if (class_id < 0) return "unknown";
    const idx: usize = @intCast(class_id);
    if (idx >= coco_class_names.len) return "unknown";
    return coco_class_names[idx];
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

    var repo = try Repo.init(allocator);
    defer repo.deinit();
    try schemaPlusSeeds(&repo);

    var server = try httpz.Server(@TypeOf(&repo)).init(allocator, .{
        .port = port,
        .request = .{
            .max_multiform_count = 8,
            .max_body_size = 1024 * 1024 * 1024,
        },
    }, &repo);
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

const httpzHandler = fn (*Repo, *httpz.Request, *httpz.Response) anyerror!void;

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
        repo: *Repo,
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
        fn handler(repo: *Repo, _: *httpz.Request, httpzResponse: *httpz.Response) !void {
            var writerAllocating = std.Io.Writer.Allocating.init(httpzResponse.arena);
            const writer = &writerAllocating.writer;
            const layout = @embedFile("views/layout.html");
            try templates.writeHeader(layout, writer);
            try templates.print(layout, "title", .{ .title = space }, writer);
            const response = responseType(name, options){ .out = writer, .repo = repo };
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
const Repo = struct {
    allocator: Allocator,
    db: *c.sqlite3,
    detector: ?onnxruntime.Runtime,
    processing_mutex: std.Thread.Mutex,

    const FileRecord = struct {
        id: i32,
        filename: string,
        detections_count: i32,
        created_at: string,
        updated_at: string,
    };

    const UploadedFile = struct {
        id: i64,
        filename: string,
    };

    const UploadListRecord = struct {
        id: i64,
        filename: string,
    };

    const DetectionRecord = struct {
        id: i32,
        class_id: i32,
        score: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        created_at: string,
    };

    fn init(allocator: Allocator) !Repo {
        std.fs.cwd().makePath(uploads_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        std.fs.cwd().makePath(processed_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        var db_ptr: ?*c.sqlite3 = null;
        if (c.sqlite3_open("app.db", &db_ptr) != c.SQLITE_OK) {
            if (db_ptr) |db| _ = c.sqlite3_close(db);
            return error.SqliteOpenFailed;
        }

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
            .db = db_ptr.?,
            .detector = detector,
            .processing_mutex = .{},
        };
    }

    fn deinit(self: *Repo) void {
        if (self.detector) |*detector| detector.deinit();
        _ = c.sqlite3_close(self.db);
    }

    fn exec(self: *Repo, sql: [*:0]const u8) !void {
        var err_msg: [*c]u8 = null;
        if (c.sqlite3_exec(self.db, sql, null, null, &err_msg) != c.SQLITE_OK) {
            if (err_msg != null) {
                const msg: [*:0]u8 = @ptrCast(err_msg);
                std.log.err("sqlite exec failed: {s}", .{std.mem.span(msg)});
                c.sqlite3_free(err_msg);
            }
            return error.SqliteExecFailed;
        }
    }

    fn allFiles(self: *Repo) ![]FileRecord {
        const sql =
            \\SELECT
            \\  f.id,
            \\  f.filename,
            \\  CAST(COUNT(d.id) AS INTEGER) AS detections_count,
            \\  f.created_at,
            \\  f.updated_at
            \\FROM files f
            \\LEFT JOIN detections d ON d.file_id = f.id
            \\GROUP BY f.id, f.filename, f.created_at, f.updated_at
            \\ORDER BY f.id DESC
        ;
        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
            std.log.err("sqlite prepare failed in allFiles: {s}", .{std.mem.span(c.sqlite3_errmsg(self.db))});
            return error.SqlitePrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        var files = try std.ArrayList(FileRecord).initCapacity(self.allocator, 0);
        errdefer {
            for (files.items) |file| {
                self.allocator.free(file.filename);
                self.allocator.free(file.created_at);
                self.allocator.free(file.updated_at);
            }
            files.deinit(self.allocator);
        }

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            const filename = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 1)));
            const created_at = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 3)));
            const updated_at = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 4)));
            try files.append(self.allocator, .{
                .id = c.sqlite3_column_int(stmt, 0),
                .filename = filename,
                .detections_count = c.sqlite3_column_int(stmt, 2),
                .created_at = created_at,
                .updated_at = updated_at,
            });
        }
        return files.toOwnedSlice(self.allocator);
    }

    fn allUploadNames(self: *Repo) ![]UploadListRecord {
        var dir = try std.fs.cwd().openDir(uploads_dir, .{ .iterate = true });
        defer dir.close();

        var files = try std.ArrayList(UploadListRecord).initCapacity(self.allocator, 0);
        errdefer {
            for (files.items) |file| self.allocator.free(file.filename);
            files.deinit(self.allocator);
        }

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind != .file) continue;
            const underscore_index = std.mem.indexOfScalar(u8, entry.name, '_');
            if (underscore_index == null) continue;
            const idx = underscore_index.?;
            const id: i64 = std.fmt.parseInt(i64, entry.name[0..idx], 10) catch continue;
            if (id <= 0) continue;
            const display_name: []const u8 = entry.name[idx + 1 ..];
            try files.append(self.allocator, .{
                .id = id,
                .filename = try self.allocator.dupe(u8, display_name),
            });
        }

        return files.toOwnedSlice(self.allocator);
    }

    fn freeFiles(self: *Repo, files: []FileRecord) void {
        for (files) |file| {
            self.allocator.free(file.filename);
            self.allocator.free(file.created_at);
            self.allocator.free(file.updated_at);
        }
        self.allocator.free(files);
    }

    fn freeUploadNames(self: *Repo, files: []UploadListRecord) void {
        for (files) |file| self.allocator.free(file.filename);
        self.allocator.free(files);
    }

    fn insertOrGetFile(self: *Repo, filename: []const u8) !UploadedFile {
        const escaped = try escapeSqlString(self.allocator, filename);
        defer self.allocator.free(escaped);

        const insert_sql = try std.fmt.allocPrint(
            self.allocator,
            "INSERT OR IGNORE INTO files(filename) VALUES ('{s}')",
            .{escaped},
        );
        defer self.allocator.free(insert_sql);
        try self.execZ(insert_sql);

        const select_sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT id, filename FROM files WHERE filename = '{s}' LIMIT 1",
            .{escaped},
        );
        defer self.allocator.free(select_sql);

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, select_sql.ptr, @intCast(select_sql.len), &stmt, null) != c.SQLITE_OK) {
            return error.SqlitePrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) return error.SqliteRowNotFound;

        return .{
            .id = c.sqlite3_column_int64(stmt, 0),
            .filename = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 1))),
        };
    }

    fn freeUploadedFile(self: *Repo, file: UploadedFile) void {
        self.allocator.free(file.filename);
    }

    fn saveUploadedFile(self: *Repo, file_id: i64, filename: []const u8, bytes: []const u8) ![]u8 {
        const safe_name = try sanitizeFilename(self.allocator, filename);
        defer self.allocator.free(safe_name);

        const rel_path = try std.fmt.allocPrint(self.allocator, "{s}/{d}_{s}", .{ uploads_dir, file_id, safe_name });
        errdefer self.allocator.free(rel_path);

        var file = try std.fs.cwd().createFile(rel_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(bytes);
        return rel_path;
    }

    fn sanitizeFilename(allocator: Allocator, name: []const u8) ![]u8 {
        var out = try std.ArrayList(u8).initCapacity(allocator, name.len);
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

    fn findStoredUploadNameById(self: *Repo, file_id: i64) !?[]u8 {
        var dir = try std.fs.cwd().openDir(uploads_dir, .{ .iterate = true });
        defer dir.close();
        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind != .file) continue;
            const underscore_index = std.mem.indexOfScalar(u8, entry.name, '_') orelse continue;
            const id = std.fmt.parseInt(i64, entry.name[0..underscore_index], 10) catch continue;
            if (id == file_id) return try self.allocator.dupe(u8, entry.name);
        }
        return null;
    }

    fn ensureVideoDetections(self: *Repo, stored_upload_name: []const u8, file_id: i64) ![]u8 {
        self.processing_mutex.lock();
        defer self.processing_mutex.unlock();

        const detections_name = try std.fmt.allocPrint(self.allocator, "{d}.jsonl", .{file_id});
        errdefer self.allocator.free(detections_name);
        const detections_rel_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ processed_dir, detections_name });
        defer self.allocator.free(detections_rel_path);

        var need_generate = false;
        std.fs.cwd().access(detections_rel_path, .{}) catch {
            need_generate = true;
        };
        if (!need_generate) {
            if (!try self.isLikelyValidDetectionsJsonl(detections_rel_path)) {
                std.fs.cwd().deleteFile(detections_rel_path) catch {};
                need_generate = true;
            }
        }

        if (need_generate) {
            const upload_rel_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ uploads_dir, stored_upload_name });
            defer self.allocator.free(upload_rel_path);

            const tmp_rel_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.tmp", .{ processed_dir, detections_name });
            defer self.allocator.free(tmp_rel_path);
            std.fs.cwd().deleteFile(tmp_rel_path) catch {};

            video_yolo.inferVideoToJsonl(self.allocator, upload_rel_path, tmp_rel_path) catch |err| {
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
    fn isLikelyValidDetectionsJsonl(self: *Repo, rel_path: []const u8) !bool {
        const data = std.fs.cwd().readFileAlloc(self.allocator, rel_path, 256 * 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => return false,
            else => return err,
        };
        defer self.allocator.free(data);
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

    fn getFileNameById(self: *Repo, file_id: i64) !?[]u8 {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT filename FROM files WHERE id = {d} LIMIT 1",
            .{file_id},
        );
        defer self.allocator.free(sql);

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql.ptr, @intCast(sql.len), &stmt, null) != c.SQLITE_OK) {
            return error.SqlitePrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        if (c.sqlite3_step(stmt) != c.SQLITE_ROW) return null;
        return try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 0)));
    }

    fn allDetectionsByFileId(self: *Repo, file_id: i64) ![]DetectionRecord {
        const sql = try std.fmt.allocPrint(
            self.allocator,
            \\SELECT id, class_id, score, x1, y1, x2, y2, created_at
            \\FROM detections
            \\WHERE file_id = {d}
            \\ORDER BY score DESC, id DESC
        , .{file_id});
        defer self.allocator.free(sql);

        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql.ptr, @intCast(sql.len), &stmt, null) != c.SQLITE_OK) {
            return error.SqlitePrepareFailed;
        }
        defer _ = c.sqlite3_finalize(stmt);

        var rows = try std.ArrayList(DetectionRecord).initCapacity(self.allocator, 0);
        errdefer {
            for (rows.items) |row| self.allocator.free(row.created_at);
            rows.deinit(self.allocator);
        }

        while (c.sqlite3_step(stmt) == c.SQLITE_ROW) {
            try rows.append(self.allocator, .{
                .id = c.sqlite3_column_int(stmt, 0),
                .class_id = c.sqlite3_column_int(stmt, 1),
                .score = @floatCast(c.sqlite3_column_double(stmt, 2)),
                .x1 = @floatCast(c.sqlite3_column_double(stmt, 3)),
                .y1 = @floatCast(c.sqlite3_column_double(stmt, 4)),
                .x2 = @floatCast(c.sqlite3_column_double(stmt, 5)),
                .y2 = @floatCast(c.sqlite3_column_double(stmt, 6)),
                .created_at = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 7))),
            });
        }

        return rows.toOwnedSlice(self.allocator);
    }

    fn freeDetectionsByFile(self: *Repo, rows: []DetectionRecord) void {
        for (rows) |row| self.allocator.free(row.created_at);
        self.allocator.free(rows);
    }

    fn insertDetections(self: *Repo, file_id: i64, detections: []const onnxruntime.Detection) !void {
        for (detections) |d| {
            const sql = try std.fmt.allocPrint(
                self.allocator,
                "INSERT INTO detections(file_id,class_id,score,x1,y1,x2,y2) VALUES ({d},{d},{d},{d},{d},{d},{d})",
                .{
                    file_id,
                    @as(i64, @intCast(d.class_id)),
                    d.score,
                    d.x1,
                    d.y1,
                    d.x2,
                    d.y2,
                },
            );
            defer self.allocator.free(sql);
            try self.execZ(sql);
        }
    }

    fn execZ(self: *Repo, sql: []const u8) !void {
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        try self.exec(sql_z);
    }

    fn escapeSqlString(allocator: Allocator, input: []const u8) ![]u8 {
        var out = try std.ArrayList(u8).initCapacity(allocator, input.len);
        errdefer out.deinit(allocator);
        for (input) |ch| {
            try out.append(allocator, ch);
            if (ch == '\'') try out.append(allocator, '\'');
        }
        return out.toOwnedSlice(allocator);
    }
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
                    i32 => {
                        if (comptime std.mem.eql(u8, field.name, "class_id")) {
                            if (std.mem.eql(u8, title, "Detections")) {
                                try templates.print(template, "td-string", .{ .value = detectionClassName(value) }, writer);
                            } else {
                                try templates.print(template, "td-number", .{ .value = value }, writer);
                            }
                        } else {
                            try templates.print(template, "td-number", .{ .value = value }, writer);
                        }
                    },
                    i64 => try templates.print(template, "td-number", .{ .value = value }, writer),
                    f32 => try templates.print(template, "td", .{ .value = value }, writer),
                    string => {
                        if (comptime std.mem.eql(u8, field.name, "filename")) {
                            if (comptime @hasField(resultRowType, "id")) {
                                if (std.mem.eql(u8, title, "Files") and @field(file, "id") > 0) {
                                    try writer.print(
                                        \\<td class="px-6 py-4"><a class="text-blue-600 hover:underline" href="/files/{d}">{s}</a></td>
                                    , .{ @field(file, "id"), value });
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
        const files = try response.repo.allUploadNames();
        defer response.repo.freeUploadNames(files);
        try table(response.out, "Files", files);
    }
    fn new(response: anytype) !void {
        try response.writeHeader();
    }

    fn show(repo: *Repo, request: *httpz.Request, response: *httpz.Response) !void {
        var writerAllocating = std.Io.Writer.Allocating.init(response.arena);
        const writer = &writerAllocating.writer;
        const layout = @embedFile("views/layout.html");
        try templates.writeHeader(layout, writer);
        try templates.print(layout, "title", .{ .title = "Files" }, writer);

        const id_str = request.param("id") orelse {
            response.status = 400;
            try writer.writeAll("missing id");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        };

        const file_id = std.fmt.parseInt(i64, id_str, 10) catch {
            response.status = 400;
            try writer.writeAll("invalid id");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        };

        const maybe_file_name = try repo.getFileNameById(file_id);
        if (maybe_file_name == null) {
            response.status = 404;
            try writer.writeAll("file not found");
            try templates.write(layout, "close-body", writer);
            response.body = writerAllocating.written();
            return;
        }
        const file_name = maybe_file_name.?;
        defer repo.allocator.free(file_name);

        const template = @embedFile("views/files/show.html");
        try templates.printHeader(template, .{ .file_id = file_id, .file_name = file_name }, writer);

        const stored_name = try repo.findStoredUploadNameById(file_id);
        if (stored_name) |upload_name| {
            defer repo.allocator.free(upload_name);
            if (isVideoFileName(upload_name)) {
                const detections_name = try std.fmt.allocPrint(repo.allocator, "{d}.jsonl", .{file_id});
                defer repo.allocator.free(detections_name);

                try writer.writeAll(
                    \\<div class="relative mx-4 mt-4 inline-block">
                    \\  <video id="video-player" class="max-w-full rounded-lg border border-slate-300" controls src="/uploads/
                );
                try writer.print("{s}", .{upload_name});
                try writer.writeAll(
                    \\"></video>
                    \\  <canvas id="video-overlay" class="absolute left-0 top-0 pointer-events-none"></canvas>
                    \\</div>
                    \\<script>
                    \\(async function() {
                    \\  const video = document.getElementById('video-player');
                    \\  const canvas = document.getElementById('video-overlay');
                    \\  const ctx = canvas.getContext('2d');
                    \\  const classNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];
                    \\  const raw = await fetch('/processed/
                );
                try writer.print("{s}", .{detections_name});
                try writer.writeAll(
                    \\').then(r => r.text());
                    \\  const frames = [];
                    \\  for (const line of raw.split('\n')) {
                    \\    const t = line.trim();
                    \\    if (!t) continue;
                    \\    try { frames.push(JSON.parse(t)); } catch (_) {}
                    \\  }
                    \\  function resizeOverlay() {
                    \\    const w = video.clientWidth || video.videoWidth || 0;
                    \\    const h = video.clientHeight || video.videoHeight || 0;
                    \\    if (!w || !h) return;
                    \\    canvas.width = w;
                    \\    canvas.height = h;
                    \\    canvas.style.width = w + 'px';
                    \\    canvas.style.height = h + 'px';
                    \\  }
                    \\  function drawCurrentFrame() {
                    \\    resizeOverlay();
                    \\    if (frames.length === 0 || !video.duration || !canvas.width || !canvas.height) return;
                    \\    const fps = frames.length / video.duration;
                    \\    const frameIndex = Math.max(1, Math.floor(video.currentTime * fps));
                    \\    const frame = frames[Math.min(frames.length - 1, frameIndex - 1)];
                    \\    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    \\    ctx.lineWidth = 2;
                    \\    ctx.strokeStyle = '#ef4444';
                    \\    ctx.fillStyle = '#ef4444';
                    \\    ctx.font = '12px sans-serif';
                    \\    for (const det of frame.detections) {
                    \\      const x = det.x1 * canvas.width;
                    \\      const y = det.y1 * canvas.height;
                    \\      const w = (det.x2 - det.x1) * canvas.width;
                    \\      const h = (det.y2 - det.y1) * canvas.height;
                    \\      if (w <= 0 || h <= 0) continue;
                    \\      ctx.strokeRect(x, y, w, h);
                    \\      const name = classNames[det.class_id] || ('class ' + det.class_id);
                    \\      ctx.fillText(name + ' ' + det.score.toFixed(2), x + 2, Math.max(12, y - 4));
                    \\    }
                    \\  }
                    \\  function tick() {
                    \\    drawCurrentFrame();
                    \\    requestAnimationFrame(tick);
                    \\  }
                    \\  video.addEventListener('loadedmetadata', resizeOverlay);
                    \\  video.addEventListener('timeupdate', drawCurrentFrame);
                    \\  video.addEventListener('seeked', drawCurrentFrame);
                    \\  video.addEventListener('play', drawCurrentFrame);
                    \\  video.addEventListener('pause', drawCurrentFrame);
                    \\  window.addEventListener('resize', resizeOverlay);
                    \\  resizeOverlay();
                    \\  requestAnimationFrame(tick);
                    \\})();
                    \\</script>
                );
            } else {
                const detections = try repo.allDetectionsByFileId(file_id);
                defer repo.freeDetectionsByFile(detections);
                try table(writer, "Detections", detections);
            }
        } else {
            const detections = try repo.allDetectionsByFileId(file_id);
            defer repo.freeDetectionsByFile(detections);
            try table(writer, "Detections", detections);
        }

        try templates.write(layout, "close-body", writer);
        response.body = writerAllocating.written();
    }

    fn create(repo: *Repo, request: *httpz.Request, response: *httpz.Response) !void {
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
                const stored = repo.insertOrGetFile(incoming_name) catch |err| {
                    print("file insert failed for {s}: {}", .{ incoming_name, err });
                    response.status = 500;
                    response.body = "failed to store file";
                    return;
                };
                defer repo.freeUploadedFile(stored);

                const saved_path = repo.saveUploadedFile(stored.id, stored.filename, kv.value.value) catch |err| {
                    print("file save failed for {s}: {}", .{ stored.filename, err });
                    response.status = 500;
                    response.body = "failed to save uploaded file";
                    return;
                };
                defer repo.allocator.free(saved_path);
                print("saved upload to {s}", .{saved_path});

                if (repo.detector != null) {
                    var detector = &repo.detector.?;
                    const detections = detector.detectFromImageBytes(repo.allocator, kv.value.value) catch |err| {
                        print("detection failed for {s}: {}", .{ stored.filename, err });
                        continue;
                    };
                    defer detector.freeDetections(repo.allocator, detections);
                    repo.insertDetections(stored.id, detections) catch |err| {
                        print("failed to persist detections for file id={d}: {}", .{ stored.id, err });
                        continue;
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

const handlers = struct {
    fn home(response: anytype) !void {
        try response.writeHeader();
    }

    fn wasm(response: anytype) !void {
        try response.writeHeader();
    }
};

fn schemaPlusSeeds(repo: *Repo) !void {
    try repo.exec(
        \\CREATE TABLE IF NOT EXISTS files (
        \\  id INTEGER PRIMARY KEY AUTOINCREMENT,
        \\  filename TEXT NOT NULL UNIQUE,
        \\  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \\  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        \\)
    );
    try repo.exec("INSERT OR IGNORE INTO files(filename) VALUES ('bar')");
    try repo.exec("INSERT OR IGNORE INTO files(filename) VALUES ('foo')");
    try repo.exec(
        \\CREATE TABLE IF NOT EXISTS detections (
        \\  id INTEGER PRIMARY KEY AUTOINCREMENT,
        \\  file_id INTEGER NOT NULL,
        \\  class_id INTEGER NOT NULL,
        \\  score REAL NOT NULL,
        \\  x1 REAL NOT NULL,
        \\  y1 REAL NOT NULL,
        \\  x2 REAL NOT NULL,
        \\  y2 REAL NOT NULL,
        \\  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        \\  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        \\)
    );
}

fn sendFile(allocator: Allocator, comptime name: string) !string {
    const dir = try std.fs.selfExeDirPathAlloc(allocator);
    const file_path = try std.fmt.allocPrint(allocator, "{s}/" ++ name, .{dir});
    const wasm_file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    defer wasm_file.close();

    const file_size = try wasm_file.getEndPos();
    const buffer = try allocator.alloc(u8, file_size);
    const bytesRead = try wasm_file.readAll(buffer);

    if (bytesRead != file_size) {
        return error.UnexpectedEndOfFile;
    }
    return buffer;
}

fn wasmJsFile(_: *Repo, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .JS;
    res.body = @embedFile("wasm.js");
}
fn wasmFile(_: *Repo, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .WASM;
    res.body = try sendFile(res.arena, file_names.wasm);
}
fn faviconHandler(_: *Repo, _: *httpz.Request, res: *httpz.Response) !void {
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
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    const size = try file.getEndPos();
    const buffer = try arena.alloc(u8, size);
    const read = try file.readAll(buffer);
    if (read != size) return error.UnexpectedEndOfFile;
    return buffer;
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

fn uploadFileHandler(_: *Repo, req: *httpz.Request, res: *httpz.Response) !void {
    const name = req.param("name") orelse {
        res.status = 400;
        res.body = "missing file name";
        return;
    };
    if (!isSafeAssetName(name)) {
        res.status = 400;
        res.body = "invalid file name";
        return;
    }
    const path = try std.fmt.allocPrint(res.arena, "{s}/{s}", .{ uploads_dir, name });
    const file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch |err| switch (err) {
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
        const byte_range = parseRangeHeader(range_header, size) orelse {
            try invalidRange(res, size);
            return;
        };

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

fn processedFileHandler(repo: *Repo, req: *httpz.Request, res: *httpz.Response) !void {
    const name = req.param("name") orelse {
        res.status = 400;
        res.body = "missing file name";
        return;
    };
    if (!isSafeAssetName(name)) {
        res.status = 400;
        res.body = "invalid file name";
        return;
    }
    if (std.ascii.endsWithIgnoreCase(name, ".jsonl")) {
        const file_id = parseDetectionsFileId(name) orelse {
            res.status = 400;
            res.body = "invalid detections file name";
            return;
        };
        const stored_name = try repo.findStoredUploadNameById(file_id);
        if (stored_name == null) {
            res.status = 404;
            res.body = "not found";
            return;
        }
        defer repo.allocator.free(stored_name.?);
        _ = repo.ensureVideoDetections(stored_name.?, file_id) catch |err| {
            std.log.err("video detections failed for id={d}: {}", .{ file_id, err });
            res.status = 500;
            res.body = "failed to process video";
            return;
        };
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

fn parseDetectionsFileId(name: []const u8) ?i64 {
    if (!std.ascii.endsWithIgnoreCase(name, ".jsonl")) return null;
    if (name.len <= ".jsonl".len) return null;
    const stem = name[0 .. name.len - ".jsonl".len];
    return std.fmt.parseInt(i64, stem, 10) catch null;
}
