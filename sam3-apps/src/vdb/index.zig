const std = @import("std");
const types = @import("types.zig");
const BBox = types.BBox;
const Detection = types.Detection;

pub const IndexMagic = "VDB_YOLO";
pub const IndexMagicSam = "VDB_SAM3";
pub const IndexVersion: u32 = 1;

pub const Posting = struct {
    frame_idx: u32,
    pts_ms: u32,
    conf: f32,
    bbox: BBox,
};

pub const ClassIndex = struct {
    label: []const u8,
    postings: []const Posting,
    /// Bitset of frame indices that contain this class
    bitset: std.DynamicBitSetUnmanaged,
};

pub const InvertedIndex = struct {
    allocator: std.mem.Allocator,
    frame_count: u32,
    classes: std.StringHashMapUnmanaged(ClassIndex),

    pub fn init(allocator: std.mem.Allocator, frame_count: u32) InvertedIndex {
        return .{
            .allocator = allocator,
            .frame_count = frame_count,
            .classes = .{},
        };
    }

    pub fn deinit(self: *InvertedIndex) void {
        var it = self.classes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.postings);
            entry.value_ptr.bitset.deinit(self.allocator);
        }
        self.classes.deinit(self.allocator);
    }

    pub fn lookup(self: *const InvertedIndex, label: []const u8) ?[]const Posting {
        // Case-insensitive lookup
        var it = self.classes.iterator();
        while (it.next()) |entry| {
            if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, label)) {
                return entry.value_ptr.postings;
            }
        }
        return null;
    }

    pub fn getBitset(self: *const InvertedIndex, label: []const u8) ?*const std.DynamicBitSetUnmanaged {
        var it = self.classes.iterator();
        while (it.next()) |entry| {
            if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, label)) {
                return &entry.value_ptr.bitset;
            }
        }
        return null;
    }

    pub fn getMatchingFrames(self: *const InvertedIndex, allocator: std.mem.Allocator, label: []const u8, min_conf: f32) ![]u32 {
        const postings = self.lookup(label) orelse return &.{};
        var frames: std.ArrayList(u32) = .empty;
        errdefer frames.deinit(allocator);

        var last_frame: ?u32 = null;
        for (postings) |p| {
            if (p.conf >= min_conf) {
                if (last_frame == null or last_frame.? != p.frame_idx) {
                    try frames.append(allocator, p.frame_idx);
                    last_frame = p.frame_idx;
                }
            }
        }
        return frames.toOwnedSlice(allocator);
    }

    /// Serializes index to an allocated byte slice
    pub fn serialize(self: *const InvertedIndex, allocator: std.mem.Allocator) ![]u8 {
        var total_detections: u32 = 0;
        var total_size: usize = 8 + 4 + 4 + 4 + 4; // magic, version, frame_count, class_count, total_detections

        var it = self.classes.iterator();
        while (it.next()) |entry| {
            total_detections += @intCast(entry.value_ptr.postings.len);
            total_size += 2 + entry.key_ptr.len + 4 + entry.value_ptr.postings.len * 28;
        }

        const out = try allocator.alloc(u8, total_size);
        errdefer allocator.free(out);

        var offset: usize = 0;
        @memcpy(out[offset .. offset + 8], IndexMagic);
        offset += 8;

        std.mem.writeInt(u32, out[offset..][0..4], IndexVersion, .little);
        offset += 4;

        std.mem.writeInt(u32, out[offset..][0..4], self.frame_count, .little);
        offset += 4;

        std.mem.writeInt(u32, out[offset..][0..4], @intCast(self.classes.count()), .little);
        offset += 4;

        std.mem.writeInt(u32, out[offset..][0..4], total_detections, .little);
        offset += 4;

        // Write classes and postings
        it = self.classes.iterator();
        while (it.next()) |entry| {
            const label = entry.key_ptr.*;
            std.mem.writeInt(u16, out[offset..][0..2], @intCast(label.len), .little);
            offset += 2;

            @memcpy(out[offset .. offset + label.len], label);
            offset += label.len;

            const postings = entry.value_ptr.postings;
            std.mem.writeInt(u32, out[offset..][0..4], @intCast(postings.len), .little);
            offset += 4;

            for (postings) |p| {
                std.mem.writeInt(u32, out[offset..][0..4], p.frame_idx, .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], p.pts_ms, .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], @bitCast(p.conf), .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], @bitCast(p.bbox.x), .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], @bitCast(p.bbox.y), .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], @bitCast(p.bbox.w), .little);
                offset += 4;

                std.mem.writeInt(u32, out[offset..][0..4], @bitCast(p.bbox.h), .little);
                offset += 4;
            }
        }

        return out;
    }

    /// Deserializes index from a byte slice
    pub fn deserialize(allocator: std.mem.Allocator, bytes: []const u8) !InvertedIndex {
        if (bytes.len < 24) return error.UnexpectedEof;
        if (!std.mem.eql(u8, bytes[0..8], IndexMagic) and !std.mem.eql(u8, bytes[0..8], IndexMagicSam)) {
            return error.InvalidIndexMagic;
        }

        var offset: usize = 8;
        const version = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        offset += 4;
        if (version != IndexVersion) {
            return error.UnsupportedIndexVersion;
        }

        const frame_count = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        offset += 4;

        const class_count = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        offset += 4;

        _ = std.mem.readInt(u32, bytes[offset..][0..4], .little); // total detections
        offset += 4;

        var index = InvertedIndex.init(allocator, frame_count);
        errdefer index.deinit();

        var i: u32 = 0;
        while (i < class_count) : (i += 1) {
            if (offset + 2 > bytes.len) return error.UnexpectedEof;
            const label_len = std.mem.readInt(u16, bytes[offset..][0..2], .little);
            offset += 2;

            if (offset + label_len > bytes.len) return error.UnexpectedEof;
            const label = try allocator.dupe(u8, bytes[offset .. offset + label_len]);
            errdefer allocator.free(label);
            offset += label_len;

            if (offset + 4 > bytes.len) return error.UnexpectedEof;
            const postings_count = std.mem.readInt(u32, bytes[offset..][0..4], .little);
            offset += 4;

            const postings = try allocator.alloc(Posting, postings_count);
            errdefer allocator.free(postings);

            var bitset = try std.DynamicBitSetUnmanaged.initEmpty(allocator, frame_count);
            errdefer bitset.deinit(allocator);

            var p_idx: u32 = 0;
            while (p_idx < postings_count) : (p_idx += 1) {
                if (offset + 28 > bytes.len) return error.UnexpectedEof;

                const f_idx = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const pts_ms = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const conf_bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const x_bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const y_bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const w_bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                const h_bits = std.mem.readInt(u32, bytes[offset..][0..4], .little);
                offset += 4;

                postings[p_idx] = .{
                    .frame_idx = f_idx,
                    .pts_ms = pts_ms,
                    .conf = @bitCast(conf_bits),
                    .bbox = .{
                        .x = @bitCast(x_bits),
                        .y = @bitCast(y_bits),
                        .w = @bitCast(w_bits),
                        .h = @bitCast(h_bits),
                    },
                };

                if (f_idx < frame_count) {
                    bitset.set(f_idx);
                }
            }

            try index.classes.put(allocator, label, .{
                .label = label,
                .postings = postings,
                .bitset = bitset,
            });
        }

        return index;
    }

const c_io = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
    extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
    extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn unlink(path: [*:0]const u8) c_int;
};

    pub fn saveToFile(self: *const InvertedIndex, path: []const u8) !void {
        const bytes = try self.serialize(self.allocator);
        defer self.allocator.free(bytes);

        const path_z = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(path_z);

        const builtin = @import("builtin");
        const O_WRONLY: c_int = 0x0001;
        const O_CREAT: c_int = if (builtin.os.tag.isDarwin()) 0x0200 else 0x0040;
        const O_TRUNC: c_int = if (builtin.os.tag.isDarwin()) 0x0400 else 0x0200;

        const fd = c_io.open(path_z.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_uint, 0o644));
        if (fd < 0) return error.CannotCreateFile;
        defer _ = c_io.close(fd);

        var written: usize = 0;
        while (written < bytes.len) {
            const ret = c_io.write(fd, bytes.ptr + written, bytes.len - written);
            if (ret <= 0) return error.WriteFailed;
            written += @intCast(ret);
        }
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !InvertedIndex {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        const O_RDONLY: c_int = 0x0000;
        const fd = c_io.open(path_z.ptr, O_RDONLY, @as(c_uint, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = c_io.close(fd);

        const SEEK_END: c_int = 2;
        const SEEK_SET: c_int = 0;
        const size_i64 = c_io.lseek(fd, 0, SEEK_END);
        if (size_i64 < 0) return error.SeekFailed;
        const size: usize = @intCast(size_i64);
        _ = c_io.lseek(fd, 0, SEEK_SET);

        const bytes = try allocator.alloc(u8, size);
        defer allocator.free(bytes);

        var total_read: usize = 0;
        while (total_read < size) {
            const ret = c_io.read(fd, bytes.ptr + total_read, size - total_read);
            if (ret <= 0) return error.ReadFailed;
            total_read += @intCast(ret);
        }

        return try deserialize(allocator, bytes);
    }

    pub fn deleteFile(allocator: std.mem.Allocator, path: []const u8) !void {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        if (c_io.unlink(path_z.ptr) != 0) return error.DeleteFailed;
    }
};

pub const IndexBuilder = struct {
    allocator: std.mem.Allocator,
    frame_count: u32 = 0,
    classes: std.StringHashMapUnmanaged(std.ArrayList(Posting)),

    pub fn init(allocator: std.mem.Allocator) IndexBuilder {
        return .{
            .allocator = allocator,
            .classes = .{},
        };
    }

    pub fn deinit(self: *IndexBuilder) void {
        var it = self.classes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var list = entry.value_ptr.*;
            list.deinit(self.allocator);
        }
        self.classes.deinit(self.allocator);
    }

    pub fn importIndex(self: *IndexBuilder, existing: *const InvertedIndex) !void {
        var it = existing.classes.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.postings) |p| {
                try self.addDetection(p.frame_idx, p.pts_ms, entry.key_ptr.*, p.conf, p.bbox);
            }
        }
        if (existing.frame_count > self.frame_count) {
            self.frame_count = existing.frame_count;
        }
    }

    pub fn addDetection(
        self: *IndexBuilder,
        frame_idx: u32,
        pts_ms: u32,
        label: []const u8,
        conf: f32,
        bbox: BBox,
    ) !void {
        if (frame_idx + 1 > self.frame_count) {
            self.frame_count = frame_idx + 1;
        }

        const entry = try self.classes.getOrPut(self.allocator, label);
        if (!entry.found_existing) {
            entry.key_ptr.* = try self.allocator.dupe(u8, label);
            entry.value_ptr.* = .empty;
        }

        try entry.value_ptr.append(self.allocator, .{
            .frame_idx = frame_idx,
            .pts_ms = pts_ms,
            .conf = conf,
            .bbox = bbox,
        });
    }

    pub fn build(self: *IndexBuilder) !InvertedIndex {
        var index = InvertedIndex.init(self.allocator, self.frame_count);
        errdefer index.deinit();

        var it = self.classes.iterator();
        while (it.next()) |entry| {
            const label_copy = try self.allocator.dupe(u8, entry.key_ptr.*);
            errdefer self.allocator.free(label_copy);

            const postings_slice = try self.allocator.dupe(Posting, entry.value_ptr.items);
            errdefer self.allocator.free(postings_slice);

            var bitset = try std.DynamicBitSetUnmanaged.initEmpty(self.allocator, self.frame_count);
            errdefer bitset.deinit(self.allocator);

            for (postings_slice) |p| {
                if (p.frame_idx < self.frame_count) {
                    bitset.set(p.frame_idx);
                }
            }

            try index.classes.put(self.allocator, label_copy, .{
                .label = label_copy,
                .postings = postings_slice,
                .bitset = bitset,
            });
        }

        return index;
    }
};
