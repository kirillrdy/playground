const std = @import("std");
const sam3 = @import("sam3");
const app_mod = @import("app.zig");
const native_main = @import("native_main");

pub fn main(init: std.process.Init) !void {
    try native_main.run(init, Platform);
}

const Platform = struct {
    pub const name = "macOS";
    pub const launch_message = "Launching native macOS interface…";

    pub fn launch(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) !void {
        var app = app_mod.App.init(allocator, io, model, example_path);
        defer app.deinit();

        try app.start();
    }
};
