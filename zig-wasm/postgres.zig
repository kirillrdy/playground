const std = @import("std");
const Allocator = std.mem.Allocator;

const db_dir = ".db";
pub const db_name = "app_db";
pub const db_user = "db_user";

pub var allocator: Allocator = undefined;

pub fn startPostgres() !void {
    var env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();
    try env_map.put("PGHOST", "/tmp");

    var exists = true;
    _ = std.fs.cwd().access(db_dir, .{}) catch {
        exists = false;
    };
    if (!exists) {
        var child = std.process.Child.init(&.{ "initdb", db_dir }, allocator);
        try child.spawn();
        _ = try child.wait();
    }

    var child = std.process.Child.init(&.{ "pg_ctl", "-D", ".db", "-o", "-k /tmp", "start", "-l", "postgres.log" }, allocator);
    try child.spawn();
    _ = try child.wait();

    if (!exists) {
        child = std.process.Child.init(&.{ "createuser", "-s", db_user }, allocator);
        child.env_map = &env_map;
        try child.spawn();
        _ = try child.wait();

        child = std.process.Child.init(&.{ "createdb", db_name }, allocator);
        child.env_map = &env_map;
        try child.spawn();
        _ = try child.wait();
    }
}

pub fn stopPostgres() void {
    var child = std.process.Child.init(&.{ "pg_ctl", "-D", db_dir, "stop" }, allocator);
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    child.spawn() catch |err| {
        std.debug.print("{}", .{err});
        return;
    };
    _ = child.wait() catch |err| {
        std.debug.print("{}", .{err});
    };
}
pub fn onSigIntStopPostgres() void {
    const act: std.os.linux.Sigaction = .{
        // Set handler to a noop function instead of `SIG.IGN` to prevent
        // leaking signal disposition to a child process.
        .handler = .{ .handler = signalHandler },
        .mask = [1]c_ulong{0},
        .flags = 0,
    };
    _ = std.os.linux.sigaction(std.os.linux.SIG.INT, &act, null);
}

fn signalHandler(signal: i32) callconv(.c) void {
    //TODO technically we will only ever get 2
    if (signal == 2) {
        stopPostgres();
    }
    std.process.exit(0);
}
