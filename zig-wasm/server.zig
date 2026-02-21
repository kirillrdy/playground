const std = @import("std");
const httpz = @import("httpz");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const string = []const u8;
const c = @cImport({
    @cInclude("sqlite3.h");
});

const print = std.log.info;
const port = 3000;

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

    var server = try httpz.Server(@TypeOf(&repo)).init(allocator, .{ .port = port, .request = .{ .max_multiform_count = 1 } }, &repo);
    defer {
        print("shutting down httpz", .{});
        server.stop();
        server.deinit();
    }

    var router = try server.router(.{});
    router.get(Paths.files.index, makeHandlerWithOptions("Files/index", .{ .default_template = false }), .{});
    router.get(Paths.files.new, makeHandler("Files/new"), .{});
    router.post(Paths.files.index, Files.create, .{});
    router.get(Paths.root, makeHandler("handlers/home"), .{});
    router.get(Paths.wasm, makeHandler("handlers/wasm"), .{});
    router.get(Paths.wasm_js_file, wasmJsFile, .{});
    router.get(Paths.wasm_file, wasmFile, .{});
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

    const FileRecord = struct {
        id: i32,
        filename: string,
        created_at: string,
        updated_at: string,
    };

    fn init(allocator: Allocator) !Repo {
        var db_ptr: ?*c.sqlite3 = null;
        if (c.sqlite3_open("app.db", &db_ptr) != c.SQLITE_OK) {
            if (db_ptr) |db| _ = c.sqlite3_close(db);
            return error.SqliteOpenFailed;
        }
        return .{
            .allocator = allocator,
            .db = db_ptr.?,
        };
    }

    fn deinit(self: *Repo) void {
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
            \\SELECT id, filename, created_at, updated_at
            \\FROM files
            \\ORDER BY id DESC
        ;
        var stmt: ?*c.sqlite3_stmt = null;
        if (c.sqlite3_prepare_v2(self.db, sql, -1, &stmt, null) != c.SQLITE_OK) {
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
            const created_at = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 2)));
            const updated_at = try self.allocator.dupe(u8, std.mem.span(c.sqlite3_column_text(stmt, 3)));
            try files.append(self.allocator, .{
                .id = c.sqlite3_column_int(stmt, 0),
                .filename = filename,
                .created_at = created_at,
                .updated_at = updated_at,
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
                    i32 => try templates.print(template, "td-number", .{ .value = value }, writer),
                    string => try templates.print(template, "td-string", .{ .value = value }, writer),
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
        const files = try response.repo.allFiles();
        defer response.repo.freeFiles(files);
        try table(response.out, @typeName(@This()), files);
    }
    fn new(response: anytype) !void {
        try response.writeHeader();
    }

    fn create(_: *Repo, request: *httpz.Request, response: *httpz.Response) !void {
        const form = try request.multiFormData();
        print("started {d} \n", .{form.len});
        var iterator = form.iterator();
        while (iterator.next()) |kv| {
            print("hello {s} {s} {any}\n", .{ kv.key, kv.value.value, kv.value.filename });
        }
        print("ended \n", .{});
        response.body = "hello";
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
