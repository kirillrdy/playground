const std = @import("std");
const ast = @import("ast.zig");
const parser = @import("parser.zig");

// Owned prompt storage can be copied safely between the query and video workers.
pub const Prompts = struct {
    storage: [16][256]u8 = undefined,
    lengths: [16]usize = undefined,
    count: usize = 0,

    pub fn get(self: *const Prompts, index: usize) []const u8 {
        return self.storage[index][0..self.lengths[index]];
    }

    fn collect(self: *Prompts, expr: *const ast.Expr) !void {
        switch (expr.*) {
            .call => |call| {
                if (std.ascii.eqlIgnoreCase(call.name, "sam3") and call.args.len >= 2 and call.args[1].* == .literal_string) {
                    const prompt = call.args[1].literal_string;
                    for (0..self.count) |i| {
                        if (std.mem.eql(u8, self.get(i), prompt)) return;
                    }
                    if (self.count == self.storage.len or prompt.len > self.storage[0].len) return error.TooManyOverlayPrompts;
                    @memcpy(self.storage[self.count][0..prompt.len], prompt);
                    self.lengths[self.count] = prompt.len;
                    self.count += 1;
                } else {
                    for (call.args) |arg| try self.collect(arg);
                }
            },
            .binary => |binary| {
                try self.collect(binary.left);
                try self.collect(binary.right);
            },
            .unary => |unary| try self.collect(unary.expr),
            else => {},
        }
    }

    pub fn fromSql(allocator: std.mem.Allocator, sql: []const u8) !Prompts {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var p = parser.Parser.init(arena.allocator(), sql);
        const stmt = try p.parse();
        var result: Prompts = .{};
        if (stmt == .select_stmt) {
            for (stmt.select_stmt.projections) |projection| try result.collect(projection.expr);
        }
        return result;
    }
};

test "overlay prompts follow projections with mixed SQL quoting" {
    const prompts = try Prompts.fromSql(std.testing.allocator,
        \\SELECT frame, sam3(frame, 'person'), sam3(frame, 'glasses')
        \\FROM "holes_3min.mp4"
        \\WHERE sam3(frame, 'person') > 0.4 AND sam3(frame, 'glasses') > 0.4 LIMIT 1
    );
    try std.testing.expectEqual(@as(usize, 2), prompts.count);
    try std.testing.expectEqualStrings("person", prompts.get(0));
    try std.testing.expectEqualStrings("glasses", prompts.get(1));
}

test "overlay prompts deduplicate selected masks and ignore predicate masks" {
    const prompts = try Prompts.fromSql(std.testing.allocator,
        \\SELECT frame, sam3(frame, 'person'), sam3(frame, 'person') FROM 'video.mp4'
        \\WHERE sam3(frame, "hat") > 0.9
    );
    try std.testing.expectEqual(@as(usize, 1), prompts.count);
    try std.testing.expectEqualStrings("person", prompts.get(0));
    const plain = try Prompts.fromSql(std.testing.allocator,
        \\SELECT frame FROM 'video.mp4' WHERE sam3(frame, "hat") > 0.9
    );
    try std.testing.expectEqual(@as(usize, 0), plain.count);
}
