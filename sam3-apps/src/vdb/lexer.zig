const std = @import("std");

pub const TokenType = enum {
    // Keywords
    kw_select,
    kw_from,
    kw_where,
    kw_limit,
    kw_as,
    kw_and,
    kw_or,
    kw_not,
    kw_contains,
    kw_create,
    kw_index,
    kw_on,
    kw_using,
    kw_with,
    kw_true,
    kw_false,
    kw_null,

    // Identifiers & Literals
    identifier,
    string_lit,
    int_lit,
    float_lit,

    // Punctuation & Operators
    lparen,
    rparen,
    comma,
    semicolon,
    star,
    eq,
    neq,
    lt,
    gt,
    lte,
    gte,

    eof,
    invalid,
};

pub const Token = struct {
    tag: TokenType,
    text: []const u8,
    pos: usize,
};

pub const Lexer = struct {
    source: []const u8,
    cursor: usize = 0,

    pub fn init(source: []const u8) Lexer {
        return .{ .source = source, .cursor = 0 };
    }

    fn peekChar(self: *const Lexer) ?u8 {
        if (self.cursor < self.source.len) {
            return self.source[self.cursor];
        }
        return null;
    }

    fn nextChar(self: *Lexer) ?u8 {
        if (self.cursor < self.source.len) {
            const ch = self.source[self.cursor];
            self.cursor += 1;
            return ch;
        }
        return null;
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.cursor < self.source.len) {
            const ch = self.source[self.cursor];
            if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') {
                self.cursor += 1;
            } else if (ch == '-' and self.cursor + 1 < self.source.len and self.source[self.cursor + 1] == '-') {
                // SQL single line comment --
                self.cursor += 2;
                while (self.cursor < self.source.len and self.source[self.cursor] != '\n') {
                    self.cursor += 1;
                }
            } else {
                break;
            }
        }
    }

    pub fn next(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.cursor >= self.source.len) {
            return .{ .tag = .eof, .text = "", .pos = self.cursor };
        }

        const start = self.cursor;
        const ch = self.nextChar().?;

        switch (ch) {
            '(' => return .{ .tag = .lparen, .text = self.source[start..self.cursor], .pos = start },
            ')' => return .{ .tag = .rparen, .text = self.source[start..self.cursor], .pos = start },
            ',' => return .{ .tag = .comma, .text = self.source[start..self.cursor], .pos = start },
            ';' => return .{ .tag = .semicolon, .text = self.source[start..self.cursor], .pos = start },
            '*' => return .{ .tag = .star, .text = self.source[start..self.cursor], .pos = start },
            '=' => return .{ .tag = .eq, .text = self.source[start..self.cursor], .pos = start },
            '!' => {
                if (self.peekChar() == '=') {
                    _ = self.nextChar();
                    return .{ .tag = .neq, .text = self.source[start..self.cursor], .pos = start };
                }
                return .{ .tag = .invalid, .text = self.source[start..self.cursor], .pos = start };
            },
            '<' => {
                if (self.peekChar() == '=') {
                    _ = self.nextChar();
                    return .{ .tag = .lte, .text = self.source[start..self.cursor], .pos = start };
                } else if (self.peekChar() == '>') {
                    _ = self.nextChar();
                    return .{ .tag = .neq, .text = self.source[start..self.cursor], .pos = start };
                }
                return .{ .tag = .lt, .text = self.source[start..self.cursor], .pos = start };
            },
            '>' => {
                if (self.peekChar() == '=') {
                    _ = self.nextChar();
                    return .{ .tag = .gte, .text = self.source[start..self.cursor], .pos = start };
                }
                return .{ .tag = .gt, .text = self.source[start..self.cursor], .pos = start };
            },
            '\'', '"' => {
                const quote = ch;
                const lit_start = self.cursor;
                while (self.cursor < self.source.len) {
                    const c = self.nextChar().?;
                    if (c == quote) {
                        return .{
                            .tag = .string_lit,
                            .text = self.source[lit_start .. self.cursor - 1],
                            .pos = start,
                        };
                    }
                }
                return .{ .tag = .invalid, .text = self.source[start..self.cursor], .pos = start };
            },
            '0'...'9' => {
                var is_float = false;
                while (self.peekChar()) |next_c| {
                    if (next_c >= '0' and next_c <= '9') {
                        _ = self.nextChar();
                    } else if (next_c == '.' and !is_float) {
                        // Check if following char is a digit
                        if (self.cursor + 1 < self.source.len and self.source[self.cursor + 1] >= '0' and self.source[self.cursor + 1] <= '9') {
                            is_float = true;
                            _ = self.nextChar();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                return .{
                    .tag = if (is_float) .float_lit else .int_lit,
                    .text = self.source[start..self.cursor],
                    .pos = start,
                };
            },
            'a'...'z', 'A'...'Z', '_' => {
                while (self.peekChar()) |next_c| {
                    if (std.ascii.isAlphanumeric(next_c) or next_c == '_' or next_c == '.') {
                        _ = self.nextChar();
                    } else {
                        break;
                    }
                }
                const text = self.source[start..self.cursor];
                const tag = classifyKeyword(text);
                return .{ .tag = tag, .text = text, .pos = start };
            },
            else => return .{ .tag = .invalid, .text = self.source[start..self.cursor], .pos = start },
        }
    }

    fn classifyKeyword(text: []const u8) TokenType {
        if (std.ascii.eqlIgnoreCase(text, "SELECT")) return .kw_select;
        if (std.ascii.eqlIgnoreCase(text, "FROM")) return .kw_from;
        if (std.ascii.eqlIgnoreCase(text, "WHERE")) return .kw_where;
        if (std.ascii.eqlIgnoreCase(text, "LIMIT")) return .kw_limit;
        if (std.ascii.eqlIgnoreCase(text, "AS")) return .kw_as;
        if (std.ascii.eqlIgnoreCase(text, "AND")) return .kw_and;
        if (std.ascii.eqlIgnoreCase(text, "OR")) return .kw_or;
        if (std.ascii.eqlIgnoreCase(text, "NOT")) return .kw_not;
        if (std.ascii.eqlIgnoreCase(text, "CONTAINS")) return .kw_contains;
        if (std.ascii.eqlIgnoreCase(text, "CREATE")) return .kw_create;
        if (std.ascii.eqlIgnoreCase(text, "INDEX")) return .kw_index;
        if (std.ascii.eqlIgnoreCase(text, "ON")) return .kw_on;
        if (std.ascii.eqlIgnoreCase(text, "USING")) return .kw_using;
        if (std.ascii.eqlIgnoreCase(text, "WITH")) return .kw_with;
        if (std.ascii.eqlIgnoreCase(text, "TRUE")) return .kw_true;
        if (std.ascii.eqlIgnoreCase(text, "FALSE")) return .kw_false;
        if (std.ascii.eqlIgnoreCase(text, "NULL")) return .kw_null;
        return .identifier;
    }
};
