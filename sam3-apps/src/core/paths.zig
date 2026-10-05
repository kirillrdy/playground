const std = @import("std");

pub fn resolveVideoPath(allocator: std.mem.Allocator, input_path: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, input_path, " \t\r\n'\"");
    if (trimmed.len == 0) return null;

    // 1. Direct path as-is (e.g. absolute or already correct relative path)
    if (allocator.dupeSentinel(u8, trimmed, 0)) |zpath| {
        defer allocator.free(zpath);
        if (std.c.access(zpath.ptr, 0) == 0) {
            return allocator.dupe(u8, trimmed) catch null;
        }
    } else |_| {}

    // 2. Expand ~/
    const maybe_home = std.c.getenv("HOME");
    if (std.mem.startsWith(u8, trimmed, "~/")) {
        if (maybe_home) |h| {
            const home = std.mem.span(h);
            if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed[2..] })) |expanded| {
                if (allocator.dupeSentinel(u8, expanded, 0)) |zpath| {
                    defer allocator.free(zpath);
                    if (std.c.access(zpath.ptr, 0) == 0) {
                        return expanded;
                    }
                } else |_| {}
                allocator.free(expanded);
            } else |_| {}
        }
    }

    // 3. Check in home directory (~/<trimmed>)
    if (maybe_home) |h| {
        const home = std.mem.span(h);
        if (std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, trimmed })) |in_home| {
            if (allocator.dupeSentinel(u8, in_home, 0)) |zpath| {
                defer allocator.free(zpath);
                if (std.c.access(zpath.ptr, 0) == 0) {
                    return in_home;
                }
            } else |_| {}
            allocator.free(in_home);
        } else |_| {}
    }

    // 4. Check in current working directory (./<trimmed>)
    if (std.fmt.allocPrint(allocator, "./{s}", .{trimmed})) |in_cwd| {
        if (allocator.dupeSentinel(u8, in_cwd, 0)) |zpath| {
            defer allocator.free(zpath);
            if (std.c.access(zpath.ptr, 0) == 0) {
                return in_cwd;
            }
        } else |_| {}
        allocator.free(in_cwd);
    } else |_| {}

    return null;
}
