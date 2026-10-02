const std = @import("std");

pub const words = [_][:0]const u8{
    "SELECT",   "FROM",   "WHERE",     "LIMIT",    "AS",           "AND",   "OR",   "NOT",
    "CONTAINS", "CREATE", "INDEX",     "ON",       "USING",        "WITH",  "TRUE", "FALSE",
    "NULL",     "frame",  "frame_idx", "frame_id", "frame_number", "index", "pts",  "timestamp",
    "sam3",     "yolov8",
};

pub const Matches = struct {
    start: usize = 0,
    end: usize = 0,
    items: [words.len][:0]const u8 = undefined,
    len: usize = 0,
};

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
