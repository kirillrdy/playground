const std = @import("std");
const sam3 = @import("sam3");
const log = @import("log");

pub fn run(init: std.process.Init, comptime Platform: type) !void {
    const allocator = init.gpa;

    log.info(init.io, "SAM 3 {s} app", .{Platform.name});
    log.info(init.io, "Model runtime: {s}", .{sam3.onnx.version()});

    var model = try sam3.Model.open(allocator, init.io);
    defer model.deinit();

    const example_path = try sam3.assets.cat.get(allocator, init.io);
    defer allocator.free(example_path);

    log.info(init.io, "Loaded segmentation and text lookup graphs", .{});
    log.info(init.io, "{s}", .{Platform.launch_message});

    try Platform.launch(allocator, init.io, &model, example_path);
}
