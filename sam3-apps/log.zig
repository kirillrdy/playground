const std = @import("std");

var mutex: std.Io.Mutex = .init;

pub fn info(io: std.Io, comptime format: []const u8, args: anytype) void {
    const now = std.Io.Timestamp.now(io, .real).toSeconds();
    if (now < 0) return;
    const epoch = std.time.epoch.EpochSeconds{ .secs = @intCast(now) };
    const year_day = epoch.getEpochDay().calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_time = epoch.getDaySeconds();

    var message_buffer: [2048]u8 = undefined;
    const message = std.fmt.bufPrint(&message_buffer, format, args) catch return;
    var line_buffer: [2112]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, "{d:0>4}/{d:0>2}/{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}Z {s}\n", .{
        year_day.year,
        month_day.month.numeric(),
        month_day.day_index + 1,
        day_time.getHoursIntoDay(),
        day_time.getMinutesIntoHour(),
        day_time.getSecondsIntoMinute(),
        message,
    }) catch return;

    mutex.lock(io) catch return;
    defer mutex.unlock(io);
    std.Io.File.stdout().writeStreamingAll(io, line) catch {};
}
