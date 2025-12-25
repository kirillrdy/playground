const std = @import("std");

pub fn main() !void {
    var debug: std.heap.DebugAllocator(.{}) = .init;
    const allocator = debug.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_fw = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_fw.interface;

    if (args.len < 2) {
        try stdout.print("Usage: {s} <avro_file>\n", .{args[0]});
        try stdout.flush();
        return;
    }

    const file_path = args[1];
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buf: [4096]u8 = undefined;
    var raw_reader = file.reader(&file_buf);
    const reader = &raw_reader.interface;

    // 1. Read Magic "Obj\x01"
    var magic: [4]u8 = undefined;
    try reader.readSliceAll(&magic);
    if (!std.mem.eql(u8, &magic, "Obj\x01")) {
        try stdout.print("Not an Avro OCF file. Magic bytes mismatch.\n", .{});
        try stdout.flush();
        return;
    }

    try stdout.print("Found Avro Magic bytes.\n", .{});

    // 2. Read Metadata Map
    try stdout.print("Reading Metadata...\n", .{});

    var codec_is_zstd = false;
    var schema_json_string: []u8 = undefined;

    while (true) {
        const block_count = try readLong(reader);
        if (block_count == 0) break;

        var count = block_count;
        if (count < 0) {
            count = -count;
            _ = try readLong(reader); // block size in bytes, ignore
        }

        var i: i64 = 0;
        while (i < count) : (i += 1) {
            const key = try readStringAlloc(allocator, reader);
            defer allocator.free(key);
            const value = try readStringAlloc(allocator, reader);

            try stdout.print("Metadata Key: {s}\n", .{key});
            if (std.mem.eql(u8, key, "avro.schema")) {
                schema_json_string = value; // Take ownership
                try stdout.print("Schema found.\n", .{});
            } else {
                if (std.mem.eql(u8, key, "avro.codec")) {
                    try stdout.print("Codec: {s}\n", .{value});
                    if (std.mem.eql(u8, value, "zstandard")) {
                        codec_is_zstd = true;
                    }
                }
                allocator.free(value);
            }
        }
    }

    // Parse Schema
    var parsed_schema = try std.json.parseFromSlice(std.json.Value, allocator, schema_json_string, .{});
    defer parsed_schema.deinit();
    defer allocator.free(schema_json_string);

    // Build Schema Registry
    var registry = SchemaRegistry.init(allocator);
    defer registry.deinit();
    try registry.build(parsed_schema.value);

    // 3. Read Sync Marker
    var sync_marker: [16]u8 = undefined;
    try reader.readSliceAll(&sync_marker);

    // 4. Read Data Blocks
    try stdout.print("Reading Data Blocks...\n", .{});
    
    var window_buffer: []u8 = &.{};
    if (codec_is_zstd) {
        window_buffer = try allocator.alloc(u8, std.compress.zstd.default_window_len + std.compress.zstd.block_size_max);
    }
    defer if (codec_is_zstd) allocator.free(window_buffer);

    var limit_buf: [4096]u8 = undefined;

    while (true) {
        // Read block count
        var block_count = readLong(reader) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };

        var block_size: i64 = 0;
        if (block_count < 0) {
            block_count = -block_count;
            block_size = try readLong(reader);
        } else {
            block_size = try readLong(reader);
        }

        try stdout.print("Block: {d} records, {d} bytes.\n", .{ block_count, block_size });

        if (codec_is_zstd) {
            var limited_state = reader.limited(@enumFromInt(@as(usize, @intCast(block_size))), &limit_buf);
            const limited = &limited_state.interface;
            var decom = std.compress.zstd.Decompress.init(limited, window_buffer, .{});

            var r: i64 = 0;
            while (r < block_count) : (r += 1) {
                try readDatum(parsed_schema.value, &decom.reader, &registry, 0, stdout);
                try stdout.print("\n", .{});
            }

            // Consume remainder if any (limited reader ensures we don't over-read file, but we should ensure stream is at end)
            _ = try limited.discardRemaining();
        } else {
            var limited_state = reader.limited(@enumFromInt(@as(usize, @intCast(block_size))), &limit_buf);
            const limited = &limited_state.interface;

            var r: i64 = 0;
            while (r < block_count) : (r += 1) {
                try readDatum(parsed_schema.value, limited, &registry, 0, stdout);
                try stdout.print("\n", .{});
            }
            _ = try limited.discardRemaining();
        }

        // Read sync marker
        var marker: [16]u8 = undefined;
        try reader.readSliceAll(&marker);

        if (!std.mem.eql(u8, &marker, &sync_marker)) {
            try stdout.print("Sync marker mismatch at end of block!\n", .{});
            try stdout.flush();
            return;
        }
    }
    try stdout.flush();
}

// --- Avro Primitive Readers ---

fn readLong(reader: anytype) !i64 {
    var variable: u64 = 0;
    var offset: u6 = 0;
    while (true) {
        const b = try reader.takeByte();
        variable |= @as(u64, b & 0x7F) << offset;
        if ((b & 0x80) == 0) break;
        offset += 7;
    }
    const encoded = variable;
    const decoded = (encoded >> 1) ^ (0 -% (encoded & 1));
    return @as(i64, @bitCast(decoded));
}

fn readInt(reader: anytype) !i32 {
    const l = try readLong(reader);
    return @intCast(l);
}

fn readFloat(reader: anytype) !f32 {
    const i = try reader.takeInt(u32, .little);
    return @bitCast(i);
}

fn readDouble(reader: anytype) !f64 {
    const i = try reader.takeInt(u64, .little);
    return @bitCast(i);
}

fn readStringAlloc(allocator: std.mem.Allocator, reader: anytype) ![]u8 {
    const len = try readLong(reader);
    if (len < 0) return error.InvalidLength;
    const buffer = try allocator.alloc(u8, @intCast(len));
    try reader.readSliceAll(buffer);
    return buffer;
}

fn printString(reader: anytype, writer: anytype) !void {
    const len = try readLong(reader);
    if (len < 0) return error.InvalidLength;
    var i: i64 = 0;
    var buf: [4096]u8 = undefined;
    while (i < len) {
        const to_read = @min(len - i, buf.len);
        const read_slice = buf[0..@intCast(to_read)];
        const n = try reader.readSliceShort(read_slice);
        if (n == 0) return error.EndOfStream;
        try writer.print("{s}", .{read_slice[0..n]});
        i += @intCast(n);
    }
}

// --- Schema Registry & Walker ---

const SchemaRegistry = struct {
    map: std.StringHashMap(std.json.Value),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SchemaRegistry {
        return .{
            .map = std.StringHashMap(std.json.Value).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SchemaRegistry) void {
        self.map.deinit();
    }

    pub fn build(self: *SchemaRegistry, schema: std.json.Value) !void {
        switch (schema) {
            .array => |arr| {
                for (arr.items) |item| {
                    try self.build(item);
                }
            },
            .object => |obj| {
                if (obj.get("name")) |name_val| {
                    if (name_val == .string) {
                        try self.map.put(name_val.string, schema);
                    }
                }

                // Recurse into fields if record
                if (obj.get("fields")) |fields| {
                    if (fields == .array) {
                        for (fields.array.items) |field| {
                            if (field.object.get("type")) |ft| {
                                try self.build(ft);
                            }
                        }
                    }
                }
                // Recurse into items if array
                if (obj.get("items")) |items| {
                    try self.build(items);
                }
                // Recurse into values if map
                if (obj.get("values")) |vals| {
                    try self.build(vals);
                }
            },
            .string => {}, // primitive or reference
            else => {},
        }
    }

    pub fn get(self: *SchemaRegistry, name: []const u8) ?std.json.Value {
        return self.map.get(name);
    }
};

fn readDatum(schema: std.json.Value, reader: anytype, registry: *SchemaRegistry, indent: usize, writer: anytype) anyerror!void {
    switch (schema) {
        .string => |s| {
            if (std.mem.eql(u8, s, "null")) {
                try writer.print("null", .{});
            } else if (std.mem.eql(u8, s, "boolean")) {
                const b = try reader.takeByte();
                try writer.print("{}", .{b != 0});
            } else if (std.mem.eql(u8, s, "int")) {
                try writer.print("{}", .{try readInt(reader)});
            } else if (std.mem.eql(u8, s, "long")) {
                try writer.print("{}", .{try readLong(reader)});
            } else if (std.mem.eql(u8, s, "float")) {
                try writer.print("{d:.3}", .{try readFloat(reader)});
            } else if (std.mem.eql(u8, s, "double")) {
                try writer.print("{d:.3}", .{try readDouble(reader)});
            } else if (std.mem.eql(u8, s, "bytes")) {
                const len = try readLong(reader);
                if (len < 0) return error.InvalidLength;
                try reader.discardAll64(@intCast(len));
                try writer.print("<bytes len={d}>", .{len});
            } else if (std.mem.eql(u8, s, "string")) {
                try writer.print("\"", .{});
                try printString(reader, writer);
                try writer.print("\"", .{});
            } else {
                // Named type reference?
                if (registry.get(s)) |def| {
                    try readDatum(def, reader, registry, indent, writer);
                } else {
                    try writer.print("Unknown type: {s}", .{s});
                }
            }
        },
        .array => |arr| {
            // Union
            const index = try readLong(reader);
            if (index < 0 or index >= arr.items.len) return error.InvalidUnionIndex;
            try readDatum(arr.items[@intCast(index)], reader, registry, indent, writer);
        },
        .object => |obj| {
            const type_val = obj.get("type");
            if (type_val == null) {
                // Should not happen for valid schema object
                return;
            }
            const t = type_val.?.string;

            if (std.mem.eql(u8, t, "record")) {
                try writer.print("{{\n", .{});
                const fields = obj.get("fields").?.array;
                for (fields.items) |field| {
                    const fname = field.object.get("name").?.string;
                    const ftype = field.object.get("type").?;

                    for (0..indent + 2) |_| try writer.print(" ", .{});
                    try writer.print("\"{s}\": ", .{fname});
                    try readDatum(ftype, reader, registry, indent + 2, writer);
                    try writer.print(",\n", .{});
                }
                for (0..indent) |_| try writer.print(" ", .{});
                try writer.print("}}", .{});
            } else if (std.mem.eql(u8, t, "array")) {
                try writer.print("[", .{});
                const item_type = obj.get("items").?;

                var block_count = try readLong(reader);
                while (block_count != 0) {
                    if (block_count < 0) {
                        block_count = -block_count;
                        _ = try readLong(reader); // block size
                    }

                    for (0..@intCast(block_count)) |_| {
                        try readDatum(item_type, reader, registry, indent, writer);
                        try writer.print(", ", .{});
                    }
                    block_count = try readLong(reader);
                }
                try writer.print("]", .{});
            } else if (std.mem.eql(u8, t, "enum")) {
                const index = try readInt(reader);
                const symbols = obj.get("symbols").?.array;
                try writer.print("\"{s}\"", .{symbols.items[@intCast(index)].string});
            } else if (std.mem.eql(u8, t, "map")) {
                try writer.print("{{ ", .{});
                const val_type = obj.get("values").?;
                var block_count = try readLong(reader);
                while (block_count != 0) {
                    if (block_count < 0) {
                        block_count = -block_count;
                        _ = try readLong(reader);
                    }
                    for (0..@intCast(block_count)) |_| {
                        // Keys are strings
                        try writer.print("\"", .{});
                        try printString(reader, writer);
                        try writer.print("\": ", .{});
                        try readDatum(val_type, reader, registry, indent, writer);
                        try writer.print(", ", .{});
                    }
                    block_count = try readLong(reader);
                }
                try writer.print("}}", .{});
            } else {
                // logical types often appear as object with "type": "long", "logicalType": ...
                // Or simple types defined as objects
                if (std.mem.eql(u8, t, "long")) {
                    try writer.print("{}", .{try readLong(reader)});
                } else if (std.mem.eql(u8, t, "string")) {
                    try writer.print("\"", .{});
                    try printString(reader, writer);
                    try writer.print("\"", .{});
                } else if (std.mem.eql(u8, t, "int")) {
                    try writer.print("{}", .{try readInt(reader)});
                } else {
                    try writer.print("Unknown obj type: {s}", .{t});
                }
            }
        },
        else => {
            try writer.print("?", .{});
        },
    }
}
