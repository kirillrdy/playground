const std = @import("std");
const web_server = @import("web_server");

pub fn main() !void {
    const address = try std.net.Address.parseIp4("0.0.0.0", 3000);
    var server = try address.listen(.{ .reuse_address = true });
    while (true) {
        var connection = try server.accept();
        defer connection.stream.close();
        var read_buffer: [5000]u8 = undefined;
        var write_buffer: [5000]u8 = undefined;

        var reader = connection.stream.reader(&read_buffer);
        var writer = connection.stream.writer(&write_buffer);
        var http_server = std.http.Server.init(reader.interface(), &writer.interface);
        while (true) {
            var request = http_server.receiveHead() catch |err| {
                switch (err) {
                    error.HttpConnectionClosing => break,
                    else => return,
                }
            };
            std.debug.print("{s}\n", .{request.head.target});
            const html =
                \\<!DOCTYPE html>
                \\<html lang="en">
                \\<head>
                \\    <meta charset="UTF-8">
                \\    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                \\    <title>Zig & Tailwind</title>
                \\    <script src="https://cdn.tailwindcss.com"></script>
                \\</head>
                \\<body class="bg-slate-900 text-slate-200 h-screen flex items-center justify-center">
                \\    <div class="text-center max-w-lg p-8 bg-slate-800 rounded-2xl shadow-2xl border border-slate-700">
                \\        <h1 class="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-500 mb-6">
                \\            Zig Web Server
                \\        </h1>
                \\        <p class="text-lg text-slate-400 mb-8 leading-relaxed">
                \\            You are viewing a response from a high-performance server written in <span class="text-yellow-400 font-mono font-bold">Zig</span>.
                \\            Styled beautifully with <span class="text-cyan-400 font-bold">Tailwind CSS</span>.
                \\        </p>
                \\        <a href="https://ziglang.org" target="_blank" class="inline-block px-8 py-3 bg-orange-500 hover:bg-orange-600 text-white font-bold rounded-lg transition-colors duration-200 shadow-lg hover:shadow-orange-500/20">
                \\            Learn More
                \\        </a>
                \\    </div>
                \\</body>
                \\</html>
            ;
            try request.respond(html, .{
                .extra_headers = &[_]std.http.Header{
                    .{ .name = "Content-Type", .value = "text/html" },
                },
            });
        }
    }
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
