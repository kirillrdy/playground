const std = @import("std");
const lex_mod = @import("lexer.zig");
const ast = @import("ast.zig");
const types = @import("types.zig");

pub const ParseError = error{
    UnexpectedToken,
    ExpectedIdentifier,
    ExpectedString,
    ExpectedLParen,
    ExpectedRParen,
    ExpectedKeyword,
    InvalidExpression,
    OutOfMemory,
} || std.fmt.ParseIntError || std.fmt.ParseFloatError;

pub const Parser = struct {
    allocator: std.mem.Allocator,
    lexer: lex_mod.Lexer,
    tok: lex_mod.Token,

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Parser {
        var lexer = lex_mod.Lexer.init(source);
        const tok = lexer.next();
        return .{
            .allocator = allocator,
            .lexer = lexer,
            .tok = tok,
        };
    }

    fn advance(self: *Parser) void {
        self.tok = self.lexer.next();
    }

    fn eat(self: *Parser, tag: lex_mod.TokenType) bool {
        if (self.tok.tag == tag) {
            self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, tag: lex_mod.TokenType) !lex_mod.Token {
        if (self.tok.tag != tag) {
            return ParseError.UnexpectedToken;
        }
        const prev = self.tok;
        self.advance();
        return prev;
    }

    pub fn parse(self: *Parser) !ast.Statement {
        switch (self.tok.tag) {
            .kw_select => {
                const s = try self.parseSelect();
                return .{ .select_stmt = s };
            },
            .kw_create => {
                const c = try self.parseCreateIndex();
                return .{ .create_index = c };
            },
            else => return ParseError.UnexpectedToken,
        }
    }

    pub fn parseSelect(self: *Parser) !ast.SelectStmt {
        _ = try self.expect(.kw_select);

        var projections: std.ArrayList(ast.Projection) = .empty;
        defer projections.deinit(self.allocator);

        while (true) {
            const expr = try self.parseExpr();
            var alias: ?[]const u8 = null;
            if (self.eat(.kw_as)) {
                const ident = try self.expect(.identifier);
                alias = ident.text;
            } else if (self.tok.tag == .identifier and self.tok.tag != .kw_from) {
                const ident = self.tok;
                self.advance();
                alias = ident.text;
            }

            try projections.append(self.allocator, .{
                .expr = expr,
                .alias = alias,
            });

            if (!self.eat(.comma)) {
                break;
            }
        }

        _ = try self.expect(.kw_from);

        const source = try self.parseTableSource();

        var where_clause: ?*ast.Expr = null;
        if (self.eat(.kw_where)) {
            where_clause = try self.parseExpr();
        }

        var limit: ?usize = null;
        if (self.eat(.kw_limit)) {
            const int_tok = try self.expect(.int_lit);
            limit = try std.fmt.parseInt(usize, int_tok.text, 10);
        }

        _ = self.eat(.semicolon);

        return .{
            .projections = try self.allocator.dupe(ast.Projection, projections.items),
            .source = source,
            .where_clause = where_clause,
            .limit = limit,
        };
    }

    fn parseTableSource(self: *Parser) !ast.TableSource {
        if (self.tok.tag == .string_lit) {
            const path = self.tok.text;
            self.advance();
            return .{ .file_path = path };
        }

        if (self.tok.tag == .identifier) {
            const name = self.tok.text;
            self.advance();
            if (self.eat(.lparen)) {
                const path_tok = try self.expect(.string_lit);
                var step: usize = 1;
                if (self.eat(.comma)) {
                    const step_tok = try self.expect(.int_lit);
                    step = try std.fmt.parseInt(usize, step_tok.text, 10);
                }
                _ = try self.expect(.rparen);
                return .{
                    .call = .{
                        .name = name,
                        .path = path_tok.text,
                        .step = step,
                    },
                };
            }
            return .{ .file_path = name };
        }

        return ParseError.ExpectedString;
    }

    pub fn parseCreateIndex(self: *Parser) !ast.CreateIndexStmt {
        _ = try self.expect(.kw_create);
        _ = try self.expect(.kw_index);

        var name: []const u8 = "default_idx";
        if (self.tok.tag == .identifier and self.tok.tag != .kw_on) {
            name = self.tok.text;
            self.advance();
        }

        _ = try self.expect(.kw_on);

        var source_file: []const u8 = "";
        var model_name: []const u8 = "sam3";
        var prompt: ?[]const u8 = null;
        var min_conf: f32 = 0.25;
        var sample_step: usize = 1;

        if (self.tok.tag == .string_lit) {
            source_file = self.tok.text;
            self.advance();

            if (self.eat(.kw_using)) {
                const model_tok = try self.expect(.identifier);
                model_name = model_tok.text;
                if (self.eat(.lparen)) {
                    while (self.tok.tag != .rparen and self.tok.tag != .eof) {
                        if (self.tok.tag == .string_lit) {
                            prompt = self.tok.text;
                        }
                        self.advance();
                    }
                    _ = try self.expect(.rparen);
                }
            } else if (self.eat(.lparen)) {
                if (self.tok.tag == .identifier) {
                    model_name = self.tok.text;
                    self.advance();
                    if (self.eat(.lparen)) {
                        while (self.tok.tag != .rparen and self.tok.tag != .eof) {
                            if (self.tok.tag == .string_lit) {
                                prompt = self.tok.text;
                            }
                            self.advance();
                        }
                        _ = try self.expect(.rparen);
                    }
                }
                _ = try self.expect(.rparen);
            }
        } else if (self.tok.tag == .identifier) {
            if (std.ascii.eqlIgnoreCase(self.tok.text, "sam3") or std.ascii.eqlIgnoreCase(self.tok.text, "yolov8")) {
                model_name = self.tok.text;
                self.advance();
                if (self.eat(.lparen)) {
                    while (self.tok.tag != .rparen and self.tok.tag != .eof) {
                        if (self.tok.tag == .string_lit) {
                            prompt = self.tok.text;
                        }
                        self.advance();
                    }
                    _ = try self.expect(.rparen);
                }
            } else {
                source_file = self.tok.text;
                self.advance();
                if (self.eat(.kw_using)) {
                    const model_tok = try self.expect(.identifier);
                    model_name = model_tok.text;
                    if (self.eat(.lparen)) {
                        while (self.tok.tag != .rparen and self.tok.tag != .eof) {
                            if (self.tok.tag == .string_lit) {
                                prompt = self.tok.text;
                            }
                            self.advance();
                        }
                        _ = try self.expect(.rparen);
                    }
                }
            }
        } else if (self.eat(.lparen)) {
            if (self.tok.tag == .identifier) {
                model_name = self.tok.text;
                self.advance();
                if (self.eat(.lparen)) {
                    while (self.tok.tag != .rparen and self.tok.tag != .eof) {
                        if (self.tok.tag == .string_lit) {
                            prompt = self.tok.text;
                        }
                        self.advance();
                    }
                    _ = try self.expect(.rparen);
                }
            }
            _ = try self.expect(.rparen);
        } else {
            return ParseError.ExpectedString;
        }

        if (self.eat(.kw_with)) {
            _ = try self.expect(.lparen);
            while (true) {
                const key = try self.expect(.identifier);
                _ = try self.expect(.eq);
                if (std.ascii.eqlIgnoreCase(key.text, "conf") or std.ascii.eqlIgnoreCase(key.text, "min_conf")) {
                    if (self.tok.tag == .float_lit) {
                        min_conf = try std.fmt.parseFloat(f32, self.tok.text);
                        self.advance();
                    } else if (self.tok.tag == .int_lit) {
                        min_conf = @floatFromInt(try std.fmt.parseInt(i64, self.tok.text, 10));
                        self.advance();
                    }
                } else if (std.ascii.eqlIgnoreCase(key.text, "step") or std.ascii.eqlIgnoreCase(key.text, "sample_step")) {
                    const val = try self.expect(.int_lit);
                    sample_step = try std.fmt.parseInt(usize, val.text, 10);
                } else if (std.ascii.eqlIgnoreCase(key.text, "prompt")) {
                    const val = try self.expect(.string_lit);
                    prompt = val.text;
                }
                if (!self.eat(.comma)) break;
            }
            _ = try self.expect(.rparen);
        }

        _ = self.eat(.semicolon);

        return .{
            .name = name,
            .source_file = source_file,
            .model_name = model_name,
            .prompt = prompt,
            .min_conf = min_conf,
            .sample_step = sample_step,
        };
    }

    pub fn parseExpr(self: *Parser) ParseError!*ast.Expr {
        return self.parseLogicalOr();
    }

    fn parseLogicalOr(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseLogicalAnd();
        while (self.eat(.kw_or)) {
            const right = try self.parseLogicalAnd();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .binary = .{
                    .op = .or_op,
                    .left = left,
                    .right = right,
                },
            };
            left = expr;
        }
        return left;
    }

    fn parseLogicalAnd(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseComparison();
        while (self.eat(.kw_and)) {
            const right = try self.parseComparison();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .binary = .{
                    .op = .and_op,
                    .left = left,
                    .right = right,
                },
            };
            left = expr;
        }
        return left;
    }

    fn parseComparison(self: *Parser) ParseError!*ast.Expr {
        const left = try self.parsePrimary();

        // Lower inclusive BETWEEN into comparisons, preserving logical AND precedence.
        if (self.eat(.kw_between)) {
            const lower = try self.parsePrimary();
            _ = try self.expect(.kw_and);
            const upper = try self.parsePrimary();
            const gte = try self.allocator.create(ast.Expr);
            gte.* = .{ .binary = .{ .op = .gte, .left = left, .right = lower } };
            const lte = try self.allocator.create(ast.Expr);
            lte.* = .{ .binary = .{ .op = .lte, .left = left, .right = upper } };
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{ .binary = .{ .op = .and_op, .left = gte, .right = lte } };
            return expr;
        }

        const op: ?ast.BinOp = switch (self.tok.tag) {
            .kw_contains => .contains,
            .eq => .eq,
            .neq => .neq,
            .lt => .lt,
            .gt => .gt,
            .lte => .lte,
            .gte => .gte,
            else => null,
        };

        if (op) |bin_op| {
            self.advance();
            const right = try self.parsePrimary();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .binary = .{
                    .op = bin_op,
                    .left = left,
                    .right = right,
                },
            };
            return expr;
        }

        return left;
    }

    fn parsePrimary(self: *Parser) ParseError!*ast.Expr {
        const expr = try self.allocator.create(ast.Expr);

        switch (self.tok.tag) {
            .int_lit => {
                const v = try std.fmt.parseInt(i64, self.tok.text, 10);
                self.advance();
                expr.* = .{ .literal_int = v };
                return expr;
            },
            .float_lit => {
                const v = try std.fmt.parseFloat(f64, self.tok.text);
                self.advance();
                expr.* = .{ .literal_float = v };
                return expr;
            },
            .string_lit => {
                const str = self.tok.text;
                self.advance();
                expr.* = .{ .literal_string = str };
                return expr;
            },
            .kw_true => {
                self.advance();
                expr.* = .{ .literal_bool = true };
                return expr;
            },
            .kw_false => {
                self.advance();
                expr.* = .{ .literal_bool = false };
                return expr;
            },
            .kw_null => {
                self.advance();
                expr.* = .literal_null;
                return expr;
            },
            .kw_not => {
                self.advance();
                const sub = try self.parsePrimary();
                expr.* = .{
                    .unary = .{
                        .op = .not_op,
                        .expr = sub,
                    },
                };
                return expr;
            },
            .lparen => {
                self.advance();
                const inside = try self.parseExpr();
                _ = try self.expect(.rparen);
                return inside;
            },
            .identifier => {
                const name = self.tok.text;
                self.advance();
                if (self.eat(.lparen)) {
                    // Function call: e.g. sam3(frame, "cat"), yolov8(frame), frame(n)
                    var args: std.ArrayList(*ast.Expr) = .empty;
                    defer args.deinit(self.allocator);

                    if (self.tok.tag != .rparen) {
                        while (true) {
                            const arg = try self.parseExpr();
                            try args.append(self.allocator, arg);
                            if (!self.eat(.comma)) break;
                        }
                    }
                    _ = try self.expect(.rparen);

                    expr.* = .{
                        .call = .{
                            .name = name,
                            .args = try self.allocator.dupe(*ast.Expr, args.items),
                        },
                    };
                    return expr;
                } else {
                    // Column or identifier reference
                    expr.* = .{ .column_ref = name };
                    return expr;
                }
            },
            else => return ParseError.InvalidExpression,
        }
    }
};

// Expand desktop shorthand into SQL using token boundaries, so keywords inside
// prompts and comments cannot be mistaken for clauses.
pub fn normalize(allocator: std.mem.Allocator, input: []const u8, video_path: []const u8) ![]u8 {
    const sql = std.mem.trim(u8, input, " \t\r\n");
    var tokens = lex_mod.Lexer.init(sql);
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
    var p = Parser.init(arena.allocator(), sourced);
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
    var p = Parser.init(arena.allocator(), sql);
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
        var p = Parser.init(arena.allocator(), sql);
        const statement = try p.parse();
        try std.testing.expectEqualStrings(if (i == 5) "explicit.mp4" else "video.mp4", statement.select_stmt.source.file_path);
        try std.testing.expectEqual(@as(?usize, if (i == 2 or i == 6) null else 1), statement.select_stmt.limit);
    }
}
