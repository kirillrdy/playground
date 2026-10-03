const std = @import("std");
const c = @import("c");

var mutex: std.Io.Mutex = .init;

pub fn info(io: std.Io, comptime format: []const u8, args: anytype) void {
    const now = std.Io.Timestamp.now(io, .real).toSeconds();
    const epoch_seconds: c.time_t = @intCast(now);
    var local: c.struct_tm = undefined;
    if (c.localtime_r(&epoch_seconds, &local) == null) return;

    var message_buffer: [2048]u8 = undefined;
    const message = std.fmt.bufPrint(&message_buffer, format, args) catch return;
    var line_buffer: [2112]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, "{d:0>4}/{d:0>2}/{d:0>2} {d:0>2}:{d:0>2}:{d:0>2} {s}\n", .{
        @as(u16, @intCast(local.tm_year + 1900)),
        @as(u8, @intCast(local.tm_mon + 1)),
        @as(u8, @intCast(local.tm_mday)),
        @as(u8, @intCast(local.tm_hour)),
        @as(u8, @intCast(local.tm_min)),
        @as(u8, @intCast(local.tm_sec)),
        message,
    }) catch return;

    mutex.lock(io) catch return;
    defer mutex.unlock(io);
    std.Io.File.stdout().writeStreamingAll(io, line) catch {};
}
