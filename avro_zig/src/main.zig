const std = @import("std");

pub fn main() !void {
    var debug: std.heap.DebugAllocator(.{}) = .init;
    const allocator = debug.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <avro_file>\n", .{args[0]});
        return;
    }

    const file_path = args[1];
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var reader = ReaderType{ .context = file };

    // 1. Read Magic "Obj\x01"
    var magic: [4]u8 = undefined;
    _ = try reader.readNoEof(&magic);
    if (!std.mem.eql(u8, &magic, "Obj\x01")) {
        std.debug.print("Not an Avro OCF file. Magic bytes mismatch.\n", .{});
        return;
    }

    std.debug.print("Found Avro Magic bytes.\n", .{});

    // 2. Read Metadata Map
    std.debug.print("Reading Metadata...\n", .{});

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
            const key = try readString(allocator, reader);
            const value = try readBytes(allocator, reader);

            std.debug.print("Metadata Key: {s}\n", .{key});
            if (std.mem.eql(u8, key, "avro.schema")) {
                schema_json_string = try allocator.dupe(u8, value);
                std.debug.print("Schema found.\n", .{});
            } else if (std.mem.eql(u8, key, "avro.codec")) {
                std.debug.print("Codec: {s}\n", .{value});
                if (std.mem.eql(u8, value, "zstandard")) {
                    codec_is_zstd = true;
                }
            }
        }
    }

    // Parse Schema
    var parsed_schema = try std.json.parseFromSlice(std.json.Value, allocator, schema_json_string, .{});
    defer parsed_schema.deinit();

    // Build Schema Registry
    var registry = SchemaRegistry.init(allocator);
    defer registry.deinit();
    try registry.build(parsed_schema.value);

    // 3. Read Sync Marker
    var sync_marker: [16]u8 = undefined;
    _ = try reader.readNoEof(&sync_marker);

    // 4. Read Data Blocks
    std.debug.print("Reading Data Blocks...\n", .{});
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

        std.debug.print("Block: {d} records, {d} bytes.\n", .{ block_count, block_size });

        if (codec_is_zstd) {
            const compressed_buf = try allocator.alloc(u8, @intCast(block_size));
            defer allocator.free(compressed_buf);

            try reader.readNoEof(compressed_buf);

            var in_reader = std.io.Reader.fixed(compressed_buf);
            var decom = std.compress.zstd.Decompress.init(&in_reader, &.{}, .{});
            var out_writer = std.io.Writer.Allocating.init(allocator);
            defer out_writer.deinit();

            _ = decom.reader.streamRemaining(&out_writer.writer) catch |err| {
                std.debug.print("  Decompression error: {any}\n", .{err});
            };

            const decompressed_data = try out_writer.toOwnedSlice();
            defer allocator.free(decompressed_data);

            std.debug.print("  Decompressed {d} bytes. Parsing records...\n", .{decompressed_data.len});

            var fbs = std.io.fixedBufferStream(decompressed_data);
            const block_reader = fbs.reader();

            var r: i64 = 0;
            while (r < block_count) : (r += 1) {
                // std.debug.print("Record {d}:\n", .{r});
                try readDatum(allocator, parsed_schema.value, block_reader, &registry, 0);
                std.debug.print("\n", .{});
            }
        } else {
            try reader.skipBytes(@intCast(block_size), .{});
        }

        // Read sync marker
        var marker: [16]u8 = undefined;
        _ = try reader.readNoEof(&marker);

        if (!std.mem.eql(u8, &marker, &sync_marker)) {
            std.debug.print("Sync marker mismatch at end of block!\n", .{});
            return;
        }
    }
}

const ReaderType = std.io.GenericReader(std.fs.File, std.fs.File.ReadError, std.fs.File.read);

// --- Avro Primitive Readers ---

fn readLong(reader: anytype) !i64 {
    var variable: u64 = 0;
    var offset: u6 = 0;
    while (true) {
        const b = try reader.readByte();
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
    var bytes: [4]u8 = undefined;
    try reader.readNoEof(&bytes);
    return @bitCast(std.mem.readInt(u32, &bytes, .little));
}

fn readDouble(reader: anytype) !f64 {
    var bytes: [8]u8 = undefined;
    try reader.readNoEof(&bytes);
    return @bitCast(std.mem.readInt(u64, &bytes, .little));
}

fn readString(allocator: std.mem.Allocator, reader: anytype) ![]u8 {
    const len = try readLong(reader);
    if (len < 0) return error.InvalidLength;
    const buffer = try allocator.alloc(u8, @intCast(len));
    try reader.readNoEof(buffer);
    return buffer;
}

fn readBytes(allocator: std.mem.Allocator, reader: anytype) ![]u8 {
    return readString(allocator, reader);
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

fn readDatum(allocator: std.mem.Allocator, schema: std.json.Value, reader: anytype, registry: *SchemaRegistry, indent: usize) anyerror!void {
    switch (schema) {
        .string => |s| {
            if (std.mem.eql(u8, s, "null")) {
                std.debug.print("null", .{});
            } else if (std.mem.eql(u8, s, "boolean")) {
                const b = try reader.readByte();
                std.debug.print("{}", .{b != 0});
            } else if (std.mem.eql(u8, s, "int")) {
                std.debug.print("{}", .{try readInt(reader)});
            } else if (std.mem.eql(u8, s, "long")) {
                std.debug.print("{}", .{try readLong(reader)});
            } else if (std.mem.eql(u8, s, "float")) {
                std.debug.print("{d:.3}", .{try readFloat(reader)});
            } else if (std.mem.eql(u8, s, "double")) {
                std.debug.print("{d:.3}", .{try readDouble(reader)});
            } else if (std.mem.eql(u8, s, "bytes")) {
                const b = try readBytes(allocator, reader);
                std.debug.print("<bytes len={d}>", .{b.len});
            } else if (std.mem.eql(u8, s, "string")) {
                const str = try readString(allocator, reader);
                std.debug.print("\"{s}\"", .{str});
            } else {
                // Named type reference?
                if (registry.get(s)) |def| {
                    try readDatum(allocator, def, reader, registry, indent);
                } else {
                    std.debug.print("Unknown type: {s}", .{s});
                }
            }
        },
        .array => |arr| {
            // Union
            const index = try readLong(reader);
            if (index < 0 or index >= arr.items.len) return error.InvalidUnionIndex;
            try readDatum(allocator, arr.items[@intCast(index)], reader, registry, indent);
        },
        .object => |obj| {
            const type_val = obj.get("type");
            if (type_val == null) {
                // Should not happen for valid schema object
                return;
            }
            const t = type_val.?.string;

            if (std.mem.eql(u8, t, "record")) {
                std.debug.print("{{\n", .{});
                const fields = obj.get("fields").?.array;
                for (fields.items) |field| {
                    const fname = field.object.get("name").?.string;
                    const ftype = field.object.get("type").?;

                    for (0..indent + 2) |_| std.debug.print(" ", .{});
                    std.debug.print("\"{s}\": ", .{fname});
                    try readDatum(allocator, ftype, reader, registry, indent + 2);
                    std.debug.print(",\n", .{});
                }
                for (0..indent) |_| std.debug.print(" ", .{});
                std.debug.print("}}", .{});
            } else if (std.mem.eql(u8, t, "array")) {
                std.debug.print("[", .{});
                const item_type = obj.get("items").?;

                var block_count = try readLong(reader);
                while (block_count != 0) {
                    if (block_count < 0) {
                        block_count = -block_count;
                        _ = try readLong(reader); // block size
                    }

                    for (0..@intCast(block_count)) |_| {
                        try readDatum(allocator, item_type, reader, registry, indent);
                        std.debug.print(", ", .{});
                    }
                    block_count = try readLong(reader);
                }
                std.debug.print("]", .{});
            } else if (std.mem.eql(u8, t, "enum")) {
                const index = try readInt(reader);
                const symbols = obj.get("symbols").?.array;
                std.debug.print("\"{s}\"", .{symbols.items[@intCast(index)].string});
            } else if (std.mem.eql(u8, t, "map")) {
                std.debug.print("{{ ", .{});
                const val_type = obj.get("values").?;
                var block_count = try readLong(reader);
                while (block_count != 0) {
                    if (block_count < 0) {
                        block_count = -block_count;
                        _ = try readLong(reader);
                    }
                    for (0..@intCast(block_count)) |_| {
                        const key = try readString(allocator, reader);
                        std.debug.print("\"{s}\": ", .{key});
                        try readDatum(allocator, val_type, reader, registry, indent);
                        std.debug.print(", ", .{});
                    }
                    block_count = try readLong(reader);
                }
                std.debug.print("}}", .{});
            } else {
                // logical types often appear as object with "type": "long", "logicalType": ...
                // Or simple types defined as objects
                if (std.mem.eql(u8, t, "long")) {
                    std.debug.print("{}", .{try readLong(reader)});
                } else if (std.mem.eql(u8, t, "string")) {
                    const str = try readString(allocator, reader);
                    std.debug.print("\"{s}\"", .{str});
                } else if (std.mem.eql(u8, t, "int")) {
                    std.debug.print("{}", .{try readInt(reader)});
                } else {
                    std.debug.print("Unknown obj type: {s}", .{t});
                }
            }
        },
        else => {
            std.debug.print("?", .{});
        },
    }
}

