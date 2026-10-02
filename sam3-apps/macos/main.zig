const std = @import("std");
const sam3 = @import("sam3");
const zimo = @import("zimo");
const app_mod = @import("app.zig");
const native_main = @import("native_main");
var completion_io: std.Io = undefined;
var completion_buffers: [8][4096:0]u8 = undefined;

export fn sam_query_completions(text: [*:0]const u8, caret: usize, start: *usize, end: *usize, output: [*][*:0]const u8, capacity: usize) usize {
    var matches: @import("vdb").completion.Matches = @import("vdb").completion.complete(std.heap.page_allocator, completion_io, std.mem.span(text), caret) catch .{};
    defer matches.deinit(std.heap.page_allocator);
    start.* = matches.start;
    end.* = matches.end;
    const count = @min(matches.len, @min(capacity, completion_buffers.len));
    for (matches.items[0..count], 0..) |word, i| {
        if (word.len > completion_buffers[i].len) return i;
        @memcpy(completion_buffers[i][0..word.len], word);
        completion_buffers[i][word.len] = 0;
        output[i] = &completion_buffers[i];
    }
    return count;
}

pub fn main(init: std.process.Init) !void {
    completion_io = init.io;
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
