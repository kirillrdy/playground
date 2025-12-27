const std = @import("std");

const Point = struct {
    x: f64,
    y: f64,
};

const Bounds = struct {
    min: Point,
    max: Point,
};

const Detection = struct {
    frame_id: i32,
    bounds: ?Bounds,
    geometry_type: i32,
    wkt: ?[]u8,
    confidence: f32,
};

const EntityDatumSource = struct {
    confidence: f64,
    frameId: i32,
};

const EavtValue = union(enum) {
    string: []u8,
    int: i32,
    long: i64,
    float: f32,
    double: f64,
    null: void,
};

const Eavt = struct {
    entityId: []u8,
    entityAttributeId: []u8,
    entityAttributeEnumId: []u8,
    entityDatumSource: EntityDatumSource,
    value: EavtValue,
    time: i64,
};

const Track = struct {
    track_id: []u8,
    data_file_id: []u8,
    detections: []Detection,
    eavts: []Eavt,
};

const Entity = struct {
    id: []u8,
    object_class: []u8,
    tracks: []Track,
    embeddings: [][]i32,
};

pub fn main() !void {
    var debug: std.heap.DebugAllocator(.{}) = .init;
    // We use an arena for all parsed data to make cleanup easy
    var arena = std.heap.ArenaAllocator.init(debug.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    // We don't need to free args explicitly as they are in the arena,
    // but typically argsAlloc uses the passed allocator.
    // Arena deinit will handle it.

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
    // We don't need to parse the schema JSON to build a registry anymore,
    // but we still need to consume the metadata blocks.

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
            const value = try readStringAlloc(allocator, reader);

            if (std.mem.eql(u8, key, "avro.codec")) {
                try stdout.print("Codec: {s}\n", .{value});
                if (std.mem.eql(u8, value, "zstandard")) {
                    codec_is_zstd = true;
                }
            }
        }
    }

    // 3. Read Sync Marker
    var sync_marker: [16]u8 = undefined;
    try reader.readSliceAll(&sync_marker);

    // 4. Read Data Blocks
    try stdout.print("Reading Data Blocks...\n", .{});

    var window_buffer: []u8 = &.{};
    if (codec_is_zstd) {
        window_buffer = try allocator.alloc(u8, std.compress.zstd.default_window_len + std.compress.zstd.block_size_max);
    }
    // Arena will free window_buffer

    var entities: std.ArrayList(Entity) = .{};

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

        // try stdout.print("Block: {d} records, {d} bytes.\n", .{ block_count, block_size });

        if (codec_is_zstd) {
            var limited_state = reader.limited(@enumFromInt(@as(usize, @intCast(block_size))), &limit_buf);
            const limited = &limited_state.interface;
            var decom = std.compress.zstd.Decompress.init(limited, window_buffer, .{});

            var r: i64 = 0;
            while (r < block_count) : (r += 1) {
                const union_index = try readLong(&decom.reader);
                if (union_index != 6) return error.UnexpectedUnionType;
                const entity = try readEntity(allocator, &decom.reader);
                try entities.append(allocator, entity);
            }

            _ = try limited.discardRemaining();
        } else {
            var limited_state = reader.limited(@enumFromInt(@as(usize, @intCast(block_size))), &limit_buf);
            const limited = &limited_state.interface;

            var r: i64 = 0;
            while (r < block_count) : (r += 1) {
                const union_index = try readLong(limited);
                if (union_index != 6) return error.UnexpectedUnionType;
                const entity = try readEntity(allocator, limited);
                try entities.append(allocator, entity);
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

    try stdout.print("Successfully parsed {d} Entities.\n", .{entities.items.len});
    if (entities.items.len > 0) {
        try stdout.print("First Entity ID: {s}\n", .{entities.items[0].id});
        try stdout.print("First Entity Track Count: {d}\n", .{entities.items[0].tracks.len});
    }

    try stdout.flush();
}

// --- Avro Typed Readers ---

fn readEntity(allocator: std.mem.Allocator, reader: anytype) !Entity {
    const id = try readStringAlloc(allocator, reader);
    const object_class = try readStringAlloc(allocator, reader);
    const tracks = try readArrayAlloc(Track, allocator, reader, readTrack);
    const embeddings = try readArrayAlloc([]i32, allocator, reader, readIntArray);

    return Entity{
        .id = id,
        .object_class = object_class,
        .tracks = tracks,
        .embeddings = embeddings,
    };
}

fn readTrack(allocator: std.mem.Allocator, reader: anytype) !Track {
    const track_id = try readStringAlloc(allocator, reader);
    const data_file_id = try readStringAlloc(allocator, reader);
    const detections = try readArrayAlloc(Detection, allocator, reader, readDetection);
    const eavts = try readArrayAlloc(Eavt, allocator, reader, readEavt);
    return Track{ .track_id = track_id, .data_file_id = data_file_id, .detections = detections, .eavts = eavts };
}

fn readDetection(allocator: std.mem.Allocator, reader: anytype) !Detection {
    const frame_id = try readInt(reader);

    const bounds_idx = try readLong(reader);
    var bounds: ?Bounds = null;
    if (bounds_idx == 1) {
        bounds = try readBounds(reader);
    } else if (bounds_idx != 0) {
        return error.InvalidUnionIndex;
    }

    const geometry_type = try readInt(reader);

    const wkt_idx = try readLong(reader);
    var wkt: ?[]u8 = null;
    if (wkt_idx == 1) {
        wkt = try readStringAlloc(allocator, reader);
    } else if (wkt_idx != 0) {
        return error.InvalidUnionIndex;
    }

    const confidence = try readFloat(reader);

    return Detection{
        .frame_id = frame_id,
        .bounds = bounds,
        .geometry_type = geometry_type,
        .wkt = wkt,
        .confidence = confidence,
    };
}

fn readBounds(reader: anytype) !Bounds {
    const min = try readPoint(reader);
    const max = try readPoint(reader);
    return Bounds{ .min = min, .max = max };
}

fn readPoint(reader: anytype) !Point {
    const x = try readDouble(reader);
    const y = try readDouble(reader);
    return Point{ .x = x, .y = y };
}

fn readEavt(allocator: std.mem.Allocator, reader: anytype) !Eavt {
    const entityId = try readStringAlloc(allocator, reader);
    const entityAttributeId = try readStringAlloc(allocator, reader);
    const entityAttributeEnumId = try readStringAlloc(allocator, reader);
    const entityDatumSource = try readEntityDatumSource(allocator, reader);

    const val_idx = try readLong(reader);
    const value: EavtValue = switch (val_idx) {
        0 => .{ .string = try readStringAlloc(allocator, reader) },
        1 => .{ .int = try readInt(reader) },
        2 => .{ .long = try readLong(reader) },
        3 => .{ .float = try readFloat(reader) },
        4 => .{ .double = try readDouble(reader) },
        5 => .{ .null = {} },
        else => return error.InvalidUnionIndex,
    };

    const time = try readLong(reader);

    return Eavt{
        .entityId = entityId,
        .entityAttributeId = entityAttributeId,
        .entityAttributeEnumId = entityAttributeEnumId,
        .entityDatumSource = entityDatumSource,
        .value = value,
        .time = time,
    };
}

fn readEntityDatumSource(allocator: std.mem.Allocator, reader: anytype) !EntityDatumSource {
    _ = allocator;
    const confidence = try readDouble(reader);
    const frameId = try readInt(reader);
    return EntityDatumSource{ .confidence = confidence, .frameId = frameId };
}

fn readIntArray(allocator: std.mem.Allocator, reader: anytype) ![]i32 {
    return readArrayAlloc(i32, allocator, reader, readIntWrapper);
}

fn readIntWrapper(allocator: std.mem.Allocator, reader: anytype) !i32 {
    _ = allocator;
    return readInt(reader);
}

// --- Generic Array Reader ---

fn readArrayAlloc(comptime T: type, allocator: std.mem.Allocator, reader: anytype, readFn: fn (std.mem.Allocator, anytype) anyerror!T) ![]T {
    var list: std.ArrayListUnmanaged(T) = .{};
    // Avro arrays are blocks
    var block_count = try readLong(reader);
    while (block_count != 0) {
        if (block_count < 0) {
            block_count = -block_count;
            _ = try readLong(reader); // block size in bytes
        }

        var i: i64 = 0;
        while (i < block_count) : (i += 1) {
            const item = try readFn(allocator, reader);
            try list.append(allocator, item);
        }
        block_count = try readLong(reader);
    }
    return list.toOwnedSlice(allocator);
}

// --- Primitives ---

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
