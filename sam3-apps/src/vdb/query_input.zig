const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");

// Expand desktop shorthand into SQL using token boundaries, so keywords inside
// prompts and comments cannot be mistaken for clauses.
pub fn normalize(allocator: std.mem.Allocator, input: []const u8, video_path: []const u8) ![]u8 {
    const sql = std.mem.trim(u8, input, " \t\r\n");
    var tokens = lexer.Lexer.init(sql);
    const first = tokens.next();
    if (first.tag == .kw_where) {
        return std.fmt.allocPrint(allocator, "SELECT frame FROM \"{s}\" {s}", .{ video_path, sql });
    }
    if (first.tag != .kw_select) return allocator.dupe(u8, sql);

    var depth: usize = 0;
    var has_source = false;
    var insert_at: usize = first.pos + first.text.len;
    while (true) {
        const token = tokens.next();
        if (depth == 0) {
            if (token.tag == .kw_from) has_source = true;
            if (token.tag == .kw_where or token.tag == .kw_limit or token.tag == .semicolon or token.tag == .eof) {
                insert_at = token.pos;
                break;
            }
        }
        if (token.tag == .eof) break;
        if (token.tag == .lparen) depth += 1;
        if (token.tag == .rparen and depth > 0) depth -= 1;
    }
    const sourced = if (has_source)
        try allocator.dupe(u8, sql)
    else
        try std.fmt.allocPrint(allocator, "{s}\nFROM \"{s}\" {s}", .{ sql[0..insert_at], video_path, sql[insert_at..] });
    errdefer allocator.free(sourced);

    // The desktop displays frames from streamed rows even when only a mask was
    // requested. Include the frame that produced that mask.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = parser.Parser.init(arena.allocator(), sourced);
    const statement = try p.parse();
    for (statement.select_stmt.projections) |projection| {
        switch (projection.expr.*) {
            .column_ref => |name| if (std.ascii.eqlIgnoreCase(name, "frame")) return sourced,
            .call => |call| if (std.ascii.eqlIgnoreCase(call.name, "frame")) return sourced,
            else => {},
        }
    }
    const select_end = first.pos + first.text.len;
    const result = try std.fmt.allocPrint(allocator, "{s} frame, {s}", .{ sourced[0..select_end], sourced[select_end..] });
    allocator.free(sourced);
    return result;
}

test "mask only shorthand inserts frame and source before limit" {
    const allocator = std.testing.allocator;
    const sql = try normalize(allocator, "SELECT sam3(frame, \"nose\") limit 1", "video.mp4");
    defer allocator.free(sql);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = parser.Parser.init(arena.allocator(), sql);
    const statement = try p.parse();
    try std.testing.expectEqual(@as(usize, 2), statement.select_stmt.projections.len);
    try std.testing.expectEqualStrings("frame", statement.select_stmt.projections[0].expr.column_ref);
    try std.testing.expectEqualStrings("nose", statement.select_stmt.projections[1].expr.call.args[1].literal_string);
    try std.testing.expectEqualStrings("video.mp4", statement.select_stmt.source.file_path);
    try std.testing.expectEqual(@as(?usize, 1), statement.select_stmt.limit);
}

test "normalization handles predicates terminators and keywords in prompts" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        "SELECT frame, sam3(frame, 'from where limit') LIMIT 1;",
        "SELECT frame WHERE sam3(frame, 'nose') > 0.4 LIMIT 1;",
        "SELECT frame;",
        "SELECT frame -- FROM ignored\nLIMIT 1",
        "WHERE sam3(frame, 'nose') > 0.4 LIMIT 1",
        "SELECT frame FROM 'explicit.mp4' LIMIT 1;",
        "SELECT frame -- trailing comment",
    };
    for (inputs, 0..) |input, i| {
        const sql = try normalize(allocator, input, "video.mp4");
        defer allocator.free(sql);
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var p = parser.Parser.init(arena.allocator(), sql);
        const statement = try p.parse();
        try std.testing.expectEqualStrings(if (i == 5) "explicit.mp4" else "video.mp4", statement.select_stmt.source.file_path);
        try std.testing.expectEqual(@as(?usize, if (i == 2 or i == 6) null else 1), statement.select_stmt.limit);
    }
}
