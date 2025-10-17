const jetquery = @import("jetquery");

pub const File = jetquery.Model(
    @This(),
    "files",
    struct {
        id: i32,
        filename: []const u8,
        created_at: jetquery.DateTime,
        updated_at: jetquery.DateTime,
    },
    .{},
);
