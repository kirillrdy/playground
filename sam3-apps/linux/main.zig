const std = @import("std");
const sam3 = @import("sam3");
const zimo = @import("zimo");
const app_mod = @import("app.zig");
const native_main = @import("native_main");

pub fn main(init: std.process.Init) !void {
    try native_main.run(init, Platform);
}

const Platform = struct {
    pub const name = "Wayland";
    pub const launch_message = "Connecting to Wayland display…";

    pub fn launch(allocator: std.mem.Allocator, io: std.Io, model: *sam3.Model, example_path: []const u8) !void {
        zimo.open(allocator, io, ".sam3-zimo") catch {};
        defer zimo.close();

        var app = app_mod.App.init(allocator, io, model, example_path, 1000, 720) catch |err| {
            std.debug.print("Failed to connect to Wayland display: {t}\n", .{err});
            std.debug.print("Hint: Make sure a Wayland compositor (e.g. Weston) is running and WAYLAND_DISPLAY is set.\n", .{});
            return err;
        };
        defer app.deinit();

        try app.run();
    }
};
