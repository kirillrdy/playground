const std = @import("std");
const sam3 = @import("sam3");
const zimo = @import("zimo");
const app_mod = @import("app.zig");
const native_main = @import("native_main");

export fn sam_query_completions(text: [*:0]const u8, caret: usize, start: *usize, end: *usize, output: [*][*:0]const u8, capacity: usize) usize {
    const matches = @import("vdb").completion.suggest(std.mem.span(text), caret);
    start.* = matches.start;
    end.* = matches.end;
    const count = @min(matches.len, capacity);
    for (matches.items[0..count], 0..) |word, i| output[i] = word.ptr;
    return count;
}

pub fn main(init: std.process.Init) !void {
    try native_main.run(init, Platform);
}

const Platform = struct {
    pub const name = "macOS";
    pub const launch_message = "Launching native macOS interface…";

    pub fn launch(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) !void {
        zimo.open(allocator, io, ".sam3-zimo") catch {};
        defer zimo.close();

        var app = app_mod.App.init(allocator, io, model, example_path);
        defer app.deinit();

        try app.start();
    }
};
