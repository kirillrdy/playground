const std = @import("std");
const httpz = @import("httpz");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const jetquery = @import("jetquery");
const Schema = @import("Schema.zig");
const postgres = @import("postgres.zig");
const string = []const u8;
const zts = @import("zts");

const print = std.log.info;
const port = 3000;

pub const wasm_app_name = "main";
pub const server_name = "main";

pub const file_names = struct {
    const wasm = wasm_app_name ++ ".wasm";
    const wasm_js = "wasm.js";
    pub const css = "main.css";
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
    const css_file = "/" ++ file_names.css;
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();

    // TODO somehow use arena allocator ? repo per request ???
    var repo = try Repo.init(allocator, .{
        .adapter = .{
            .database = postgres.db_name,
            .username = postgres.db_user,
            .password = "",
            .hostname = "127.0.0.1",
            .port = 5432,
        },
    });

    schemaPlusSeeds(&repo) catch {};

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
    router.get(Paths.css_file, cssFile, .{});

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
        out: std.ArrayList(u8).Writer,
        repo: *Repo,
        fn print(self: @This(), comptime section: string, args: anytype) !void {
            try zts.print(self.template, section, args, self.out);
        }

        fn printHeader(self: @This(), args: anytype) !void {
            try zts.printHeader(self.template, args, self.out);
        }

        fn writeHeader(self: @This()) !void {
            try zts.writeHeader(self.template, self.out);
        }

        fn write(self: @This(), section: string) !void {
            try zts.write(self.template, section, self.out);
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
        fn handler(repo: *Repo, _: *httpz.Request, response: *httpz.Response) !void {
            var buffer = std.ArrayList(u8){};
            const out = buffer.writer(response.arena);
            const layout = @embedFile("views/layout.html");
            try zts.writeHeader(layout, out);
            try zts.print(layout, "title", .{ .title = space, .css = Paths.css_file }, out);
            try func(responseType(name, options){ .out = out, .repo = repo });
            try zts.write(layout, "close-body", out);
            response.body = buffer.items;
        }
    }.handler;
}
test "tolower" {
    const input = "Kirill";
    const output = lowerString(input);
    _ = lowerString("somethingverylong");
    try std.testing.expectEqualStrings("kirill", &output);
}
const Repo = jetquery.Repo(.postgresql, Schema);

fn table(writer: std.ArrayList(u8).Writer, title: string, records: anytype) !void {
    const resultRowType = @typeInfo(@TypeOf(records)).pointer.child;
    const template = @embedFile("views/table.html");
    try zts.printHeader(template, .{ .title = title }, writer);
    const query_fields = @typeInfo(resultRowType).@"struct".fields;

    inline for (query_fields) |field| {
        if (!std.mem.startsWith(u8, field.name, "_")) {
            try zts.print(template, "th", .{ .value = removeUndescores(field.name) }, writer);
        }
    }
    try zts.write(template, "th-close", writer);

    for (records) |file| {
        try zts.write(template, "tr", writer);
        inline for (query_fields) |field| {
            if (comptime !std.mem.startsWith(u8, field.name, "_")) {
                const value = @field(file, field.name);

                switch (@TypeOf(value)) {
                    i32 => try zts.print(template, "td-number", .{ .value = value }, writer),
                    string => try zts.print(template, "td-string", .{ .value = value }, writer),
                    jetquery.DateTime => {
                        try writer.writeAll(
                            \\<td class="px-6 py-4">
                        );
                        try value.strftime(writer, "%Y-%m-%d %H:%M:%S");
                        try writer.writeAll("</td>");
                    },
                    else => @compileError("what type is this " ++ @typeName(@TypeOf(value))),
                }
            }
        }
        try zts.write(template, "tr-close", writer);
    }
    try zts.write(template, "tbody-close", writer);
}

const Files = struct {
    fn index(response: anytype) !void {
        const files = try jetquery.Query(.postgresql, Schema, .File).all(response.repo);
        defer response.repo.free(files);
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
        try response.printHeader(.{ .wasm = Paths.wasm, .files = Paths.files.index });
    }

    fn wasm(response: anytype) !void {
        try response.writeHeader();
    }
};

fn schemaPlusSeeds(repo: *Repo) !void {
    const t = jetquery.schema.table;
    try repo.createTable(
        "files",
        &.{
            t.primaryKey("id", .{}),
            t.column("filename", .string, .{ .unique = true }),
            t.timestamps(.{}),
        },
        .{ .if_not_exists = true },
    );
    try repo.insert(.File, .{ .filename = "bar" });
    try repo.insert(.File, .{ .filename = "foo" });
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
fn cssFile(_: *Repo, _: *httpz.Request, res: *httpz.Response) !void {
    res.content_type = .CSS;
    res.body = try sendFile(res.arena, file_names.css);
}
