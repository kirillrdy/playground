const std = @import("std");
const Child = std.process.Child;
const print = std.log.info;

const server_name = @import("app_names.zig").server_name;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    const exe_path = try std.process.executablePathAlloc(io, allocator);
    const dir = std.Io.Dir.path.dirname(exe_path) orelse ".";
    const server_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, server_name });

    var last_mod_time: i96 = 0;

    const cwd: std.Io.Dir = .cwd();
    const initial_file_info = try cwd.statFile(io, server_path, .{});
    last_mod_time = initial_file_info.mtime.nanoseconds;
    var current_child_process = try startBinary(io, server_path);

    //TODO replace with inotify
    while (true) {
        try io.sleep(.{ .nanoseconds = 100 * std.time.ns_per_ms }, .awake);

        const stat_result = try cwd.statFile(io, server_path, .{});

        if (stat_result.mtime.nanoseconds != last_mod_time) {
            print("Detected change in '{s}'!\n", .{server_path});

            current_child_process.kill(io);
            last_mod_time = stat_result.mtime.nanoseconds;

            current_child_process = try startBinary(io, server_path);
            print("Started new process\n", .{});
        }
    }
}

fn startBinary(io: std.Io, binary_path: []const u8) !Child {
    return std.process.spawn(io, .{ .argv = &.{binary_path} });
}
