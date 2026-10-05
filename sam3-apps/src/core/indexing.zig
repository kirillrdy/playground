const std = @import("std");
const vdb = @import("vdb");

pub fn saveVideoIndex(
    allocator: std.mem.Allocator,
    io: std.Io,
    index_mutex: *std.Io.Mutex,
    source_path: []const u8,
    builder: *vdb.index.IndexBuilder,
) !void {
    try index_mutex.lock(io);
    defer index_mutex.unlock(io);
    const sidecar = try std.fmt.allocPrint(allocator, "{s}.vdb", .{source_path});
    defer allocator.free(sidecar);
    if (vdb.index.InvertedIndex.loadFromFile(allocator, sidecar)) |loaded| {
        var latest = loaded;
        defer latest.deinit();
        try builder.importIndex(&latest);
    } else |_| {}
    var built = try builder.build();
    defer built.deinit();
    try built.saveToFile(sidecar);
}
