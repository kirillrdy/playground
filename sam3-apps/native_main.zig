const std = @import("std");
const sam3 = @import("sam3");

pub fn run(init: std.process.Init, comptime Platform: type) !void {
    const allocator = init.gpa;

    std.debug.print("\n=== SAM 3 {s} App ===\n\n", .{Platform.name});
    std.debug.print("  Model runtime: {s}\n", .{sam3.onnx.version()});

    var model = try sam3.Model.open(allocator, init.io);
    defer model.deinit();

    const example_path = try sam3.assets.cat.get(allocator, init.io);
    defer allocator.free(example_path);

    std.debug.print("  Loaded segmentation and text lookup graphs\n", .{});
    std.debug.print("  {s}\n\n", .{Platform.launch_message});

    try Platform.launch(allocator, init.io, &model, example_path);
}
