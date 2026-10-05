const std = @import("std");

const tm = extern struct {
    tm_sec: c_int,
    tm_min: c_int,
    tm_hour: c_int,
    tm_mday: c_int,
    tm_mon: c_int,
    tm_year: c_int,
    tm_wday: c_int,
    tm_yday: c_int,
    tm_isdst: c_int,
    tm_gmtoff: c_long,
    tm_zone: ?[*:0]const u8,
};

extern "c" fn localtime_r(timer: *const isize, result: *tm) ?*tm;

var mutex: std.Io.Mutex = .init;

pub fn info(io: std.Io, comptime format: []const u8, args: anytype) void {
    const now = std.Io.Timestamp.now(io, .real).toSeconds();
    const epoch_seconds: isize = @intCast(now);
    var local: tm = undefined;
    if (localtime_r(&epoch_seconds, &local) == null) return;

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
