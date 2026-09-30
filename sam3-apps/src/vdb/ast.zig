const std = @import("std");
const types = @import("types.zig");

pub const BinOp = enum {
    eq,
    neq,
    lt,
    gt,
    lte,
    gte,
    contains,
    and_op,
    or_op,
};

pub const UnaryOp = enum {
    not_op,
    negate,
};

pub const Expr = union(enum) {
    literal_null: void,
    literal_bool: bool,
    literal_int: i64,
    literal_float: f64,
    literal_string: []const u8,
    column_ref: []const u8,
    call: struct {
        name: []const u8,
        args: []const *Expr,
    },
    binary: struct {
        op: BinOp,
        left: *Expr,
        right: *Expr,
    },
    unary: struct {
        op: UnaryOp,
        expr: *Expr,
    },
};

pub const Projection = struct {
    expr: *Expr,
    alias: ?[]const u8 = null,
};

pub const TableSource = union(enum) {
    file_path: []const u8,
    call: struct {
        name: []const u8,
        path: []const u8,
        step: usize = 1,
        sample_fps: ?f64 = null,
    },
};

pub const SelectStmt = struct {
    projections: []const Projection,
    source: TableSource,
    where_clause: ?*Expr = null,
    limit: ?usize = null,
};

pub const CreateIndexStmt = struct {
    name: []const u8 = "default_idx",
    source_file: []const u8 = "",
    model_name: []const u8 = "sam3",
    prompt: ?[]const u8 = null,
    min_conf: f32 = 0.25,
    sample_step: usize = 1,
};

pub const Statement = union(enum) {
    select_stmt: SelectStmt,
    create_index: CreateIndexStmt,
};
