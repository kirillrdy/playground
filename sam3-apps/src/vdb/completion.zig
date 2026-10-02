const std = @import("std");
const lexer = @import("lexer.zig");

pub const words = [_][:0]const u8{
    "SELECT",   "FROM",   "WHERE",     "LIMIT",    "AS",           "AND",   "OR",   "NOT",
    "CONTAINS", "CREATE", "INDEX",     "ON",       "USING",        "WITH",  "TRUE", "FALSE",
    "NULL",     "frame",  "frame_idx", "frame_id", "frame_number", "index", "pts",  "timestamp",
    "sam3",     "yolov8", "BETWEEN",
};

pub const Matches = struct {
    start: usize = 0,
    end: usize = 0,
    items: [words.len][:0]const u8 = undefined,
    len: usize = 0,
    owned: bool = false,

    pub fn deinit(self: *Matches, allocator: std.mem.Allocator) void {
        if (self.owned) for (self.items[0..self.len]) |item| allocator.free(item);
        self.* = .{};
    }
};

pub const FileContext = struct {
    start: usize,
    end: usize,
    prefix: []const u8,
    quote: u8,
};

/// Identify the source being edited, excluding prompts, comments and later clauses.
pub fn fileContext(text: []const u8, caret: usize) ?FileContext {
    if (caret > text.len) return null;
    var tokens = lexer.Lexer.init(text[0..caret]);
    var from: ?usize = null;
    while (true) {
        const token = tokens.next();
        if (token.tag == .eof) break;
        if (token.tag == .kw_from and from == null) from = token.pos + token.text.len;
    }
    const after_from = from orelse return null;
    if (after_from == caret or !std.ascii.isWhitespace(text[after_from])) return null;
    var start = after_from;
    while (start < caret and std.ascii.isWhitespace(text[start])) start += 1;
    const quoted = start < caret and (text[start] == '\'' or text[start] == '"');
    const quote: u8 = if (quoted) text[start] else '\'';
    const prefix_start = start + @as(usize, if (quoted) 1 else 0);
    const prefix = text[prefix_start..caret];
    for (prefix) |ch| {
        if ((quoted and ch == quote) or (!quoted and (std.ascii.isWhitespace(ch) or ch == '(' or ch == ';' or ch == '\'' or ch == '"'))) return null;
    }
    // A comment is not a source filename.
    if (!quoted and std.mem.startsWith(u8, prefix, "--")) return null;
    var end = caret;
    if (quoted) {
        end = if (std.mem.indexOfScalar(u8, text[caret..], quote)) |offset| caret + offset + 1 else text.len;
    } else {
        while (end < text.len and !std.ascii.isWhitespace(text[end]) and text[end] != ';') end += 1;
    }
    return .{ .start = start, .end = end, .prefix = prefix, .quote = quote };
}

/// Filesystem suggestions own their strings; SQL suggestions borrow static words.
pub fn complete(allocator: std.mem.Allocator, io: std.Io, text: []const u8, caret: usize) !Matches {
    const context = fileContext(text, caret) orelse return suggest(text, caret);
    var result: Matches = .{ .start = context.start, .end = context.end, .owned = true };
    errdefer result.deinit(allocator);
    const slash = std.mem.lastIndexOfScalar(u8, context.prefix, '/');
    const directory_prefix = if (slash) |pos| context.prefix[0 .. pos + 1] else "";
    const name_prefix = if (slash) |pos| context.prefix[pos + 1 ..] else context.prefix;
    const directory = if (directory_prefix.len > 0) directory_prefix else ".";
    const expanded = if (std.mem.startsWith(u8, directory, "~/")) blk: {
        const home = if (std.c.getenv("HOME")) |value| std.mem.span(value) else break :blk null;
        break :blk try std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, directory[2..] });
    } else null;
    defer if (expanded) |path| allocator.free(path);
    const insertion_prefix = expanded orelse directory_prefix;
    var dir = std.Io.Dir.cwd().openDir(io, expanded orelse directory, .{ .iterate = true }) catch return result;
    defer dir.close(io);
    const Candidate = struct { name: []u8, directory: bool };
    var entries: std.ArrayList(Candidate) = .empty;
    defer {
        for (entries.items) |entry| allocator.free(entry.name);
        entries.deinit(allocator);
    }
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.name.len == 0 or (entry.name[0] == '.' and !std.mem.startsWith(u8, name_prefix, "."))) continue;
        if (!std.ascii.startsWithIgnoreCase(entry.name, name_prefix)) continue;
        const is_directory = entry.kind == .directory or (entry.kind == .sym_link and blk: {
            const stat = dir.statFile(io, entry.name, .{}) catch break :blk false;
            break :blk stat.kind == .directory;
        });
        const name = try allocator.dupe(u8, entry.name);
        entries.append(allocator, .{ .name = name, .directory = is_directory }) catch |err| {
            allocator.free(name);
            return err;
        };
    }
    std.mem.sort(Candidate, entries.items, {}, struct {
        fn less(_: void, a: Candidate, b: Candidate) bool {
            if (a.directory != b.directory) return a.directory;
            return std.ascii.lessThanIgnoreCase(a.name, b.name);
        }
    }.less);
    for (entries.items) |entry| {
        if (result.len == 8) break;
        var quote = context.quote;
        if (std.mem.indexOfScalar(u8, insertion_prefix, quote) != null or std.mem.indexOfScalar(u8, entry.name, quote) != null) quote = if (quote == '\'') '"' else '\'';
        if (std.mem.indexOfScalar(u8, insertion_prefix, quote) != null or std.mem.indexOfScalar(u8, entry.name, quote) != null) continue;
        result.items[result.len] = if (entry.directory)
            try std.fmt.allocPrintSentinel(allocator, "{c}{s}{s}/{c}", .{ quote, insertion_prefix, entry.name, quote }, 0)
        else
            try std.fmt.allocPrintSentinel(allocator, "{c}{s}{s}{c}", .{ quote, insertion_prefix, entry.name, quote }, 0);
        result.len += 1;
    }
    return result;
}

fn wordChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch >= 128;
}

/// Complete only an SQL word at the caret; prompts, paths and comments are literal.
pub fn suggest(text: []const u8, caret: usize) Matches {
    var result: Matches = .{};
    if (caret == 0 or caret > text.len) return result;
    var quote: ?u8 = null;
    var comment = false;
    var i: usize = 0;
    while (i < caret) : (i += 1) {
        const ch = text[i];
        if (comment) {
            if (ch == '\n') comment = false;
        } else if (quote) |q| {
            if (ch == q) quote = null;
        } else if (ch == '\'' or ch == '"') {
            quote = ch;
        } else if (ch == '-' and i + 1 < caret and text[i + 1] == '-') {
            comment = true;
            i += 1;
        }
    }
    if (quote != null or comment or !wordChar(text[caret - 1])) return result;
    result.start = caret;
    while (result.start > 0 and wordChar(text[result.start - 1])) result.start -= 1;
    result.end = caret;
    while (result.end < text.len and wordChar(text[result.end])) result.end += 1;
    const prefix = text[result.start..caret];
    for (words) |word| {
        if (word.len >= prefix.len and std.ascii.eqlIgnoreCase(prefix, word[0..prefix.len]) and
            !std.ascii.eqlIgnoreCase(text[result.start..result.end], word))
        {
            result.items[result.len] = word;
            result.len += 1;
        }
    }
    return result;
}

test "completion respects caret, case, word suffixes and literal boundaries" {
    const matches = suggest("SELECT fra WHERE", 10);
    try std.testing.expectEqual(@as(usize, 7), matches.start);
    try std.testing.expectEqual(@as(usize, 10), matches.end);
    try std.testing.expectEqualStrings("frame", matches.items[0]);
    try std.testing.expectEqualStrings("SELECT", suggest("se", 2).items[0]);
    const middle = suggest("SELECT frme", 9);
    try std.testing.expectEqual(@as(usize, 11), middle.end);
    for ([_][]const u8{ "", "SELECT ", "SELECT", "'fra", "\"fra", "-- fra", "123", "sam3(frame, 'person wh" }) |text| {
        try std.testing.expectEqual(@as(usize, 0), suggest(text, text.len).len);
    }
    try std.testing.expectEqualStrings("WHERE", suggest("-- comment\nwh", 13).items[0]);
    try std.testing.expectEqualStrings("WHERE", suggest("'person' wh", 11).items[0]);
    try std.testing.expectEqual(@as(usize, 0), suggest("wh", 99).len);
}

test "FROM completion identifies quoted and bare source paths without touching predicates" {
    const source = "SELECT frame FROM 'videos/clips.mp4' WHERE frame_id > 10";
    const caret = "SELECT frame FROM 'videos/cl".len;
    const context = fileContext(source, caret).?;
    try std.testing.expectEqualStrings("videos/cl", context.prefix);
    try std.testing.expectEqualStrings("'videos/clips.mp4'", source[context.start..context.end]);
    try std.testing.expectEqualStrings("../vi", fileContext("SELECT frame FROM ../vi", 23).?.prefix);
    try std.testing.expectEqualStrings("", fileContext("SELECT frame FROM ", 18).?.prefix);
    for ([_][]const u8{
        "SELECT frame FROM",                       "SELECT sam3(frame, 'FROM vi",  "SELECT frame -- FROM vi",
        "SELECT frame FROM 'video.mp4' WHERE fra", "SELECT frame FROM videos('vi", "SELECT frame FROM -- comment",
    }) |input| {
        try std.testing.expect(fileContext(input, input.len) == null);
    }
}

test "FROM autocomplete lists real directories and files and quotes paths with spaces" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.createDir(io, "clips", .default_dir);
    for ([_][]const u8{ "clip one.mp4", "clip two.mov", "other.mp4", ".hidden.mp4", "can't.mp4" }) |name| {
        const file = try temp.dir.createFile(io, name, .{});
        file.close(io);
    }
    var path_buffer: [4096]u8 = undefined;
    const length = try temp.dir.realPath(io, &path_buffer);
    const path = path_buffer[0..length];
    const query = try std.fmt.allocPrint(allocator, "SELECT frame FROM '{s}/cl", .{path});
    defer allocator.free(query);
    var matches = try complete(allocator, io, query, query.len);
    defer matches.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), matches.len);
    const directory = try std.fmt.allocPrint(allocator, "'{s}/clips/'", .{path});
    defer allocator.free(directory);
    try std.testing.expectEqualStrings(directory, matches.items[0]);
    const first_file = try std.fmt.allocPrint(allocator, "'{s}/clip one.mp4'", .{path});
    defer allocator.free(first_file);
    try std.testing.expectEqualStrings(first_file, matches.items[1]);
    const apostrophe = try std.fmt.allocPrint(allocator, "SELECT frame FROM '{s}/can", .{path});
    defer allocator.free(apostrophe);
    var quoted = try complete(allocator, io, apostrophe, apostrophe.len);
    defer quoted.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), quoted.len);
    try std.testing.expect(quoted.items[0][0] == '"');
    const missing = try std.fmt.allocPrint(allocator, "SELECT frame FROM '{s}/missing/", .{path});
    defer allocator.free(missing);
    var empty = try complete(allocator, io, missing, missing.len);
    defer empty.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}

test "accepting a directory keeps the closing quote and following SQL intact" {
    const allocator = std.testing.allocator;
    const text = "SELECT frame FROM 'cl' WHERE frame_id BETWEEN 10 AND 20";
    const caret = "SELECT frame FROM 'cl".len;
    const context = fileContext(text, caret).?;
    const directory = "'clips/'";
    const edited = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ text[0..context.start], directory, text[context.end..] });
    defer allocator.free(edited);
    const next = fileContext(edited, context.start + directory.len - 1).?;
    try std.testing.expectEqualStrings("clips/", next.prefix);
    try std.testing.expectEqualStrings(" WHERE frame_id BETWEEN 10 AND 20", edited[next.end..]);
}
