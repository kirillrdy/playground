const std = @import("std");
const wasm_app_name = @import("server.zig").wasm_app_name;
const server_name = @import("server.zig").server_name;
const string = []const u8;

fn createModule(b: *std.Build, src: string, target: anytype, optimize: anytype) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(src), .target = target, .optimize = optimize });
}

fn trimNewline(input: []const u8) []const u8 {
    return std.mem.trimRight(u8, input, "\r\n");
}

fn nixBuildPath(allocator: std.mem.Allocator, attr: []const u8) ?[]const u8 {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "nix", "build", "--no-link", "--print-out-paths", attr },
    }) catch return null;
    if (result.term != .Exited or result.term.Exited != 0) return null;
    return allocator.dupe(u8, trimNewline(result.stdout)) catch null;
}

pub fn build(b: *std.Build) !void {
    const start = try std.time.Instant.now();
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const modules = .{
        .server = createModule(b, "server.zig", target, optimize),
        .dev_server = createModule(b, "dev_server.zig", target, optimize),
        .wasm = createModule(b, "wasm.zig", b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }), optimize),
    };
    const onnx_include_path = b.option([]const u8, "onnx-include", "Path to directory containing onnxruntime_c_api.h");
    const onnx_lib_path = b.option([]const u8, "onnx-lib", "Path to directory containing libonnxruntime");
    const model_url = b.option([]const u8, "model-url", "URL to download ONNX model from") orelse "https://raw.githubusercontent.com/Hyuto/yolov8-onnxruntime-web/master/public/model/yolov8n.onnx";
    const onnx_dev_root = onnx_include_path orelse nixBuildPath(b.allocator, "nixpkgs#onnxruntime.dev");
    const onnx_out_root = onnx_lib_path orelse nixBuildPath(b.allocator, "nixpkgs#onnxruntime");
    if (onnx_dev_root == null) return error.MissingOnnxRuntimeDev;
    if (onnx_out_root == null) return error.MissingOnnxRuntime;

    const sqlite3 = b.dependency("sqlite3", .{
        .target = target,
        .optimize = optimize,
    });
    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    const sqlite3_lib = sqlite3.artifact("sqlite3");
    const sqlite3_shell = sqlite3.artifact("shell");

    modules.server.addIncludePath(sqlite3.path("."));
    modules.dev_server.addIncludePath(sqlite3.path("."));
    modules.server.addImport("zigimg", zigimg.module("zigimg"));
    if (onnx_dev_root) |path| modules.server.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{path}) });

    inline for (&.{"httpz"}) |dependency_name| {
        modules.server.addImport(dependency_name, b.dependency(dependency_name, .{}).module(dependency_name));
    }

    const server = b.addExecutable(.{ .name = server_name, .root_module = modules.server });
    server.linkLibrary(sqlite3_lib);
    if (onnx_out_root) |path| server.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{path}) });
    server.linkSystemLibrary("onnxruntime");

    const dev_server = b.addExecutable(.{ .name = "dev_server", .root_module = modules.dev_server });
    dev_server.linkLibrary(sqlite3_lib);

    const wasm_app = b.addExecutable(.{ .name = wasm_app_name, .root_module = modules.wasm });
    wasm_app.entry = .disabled;
    wasm_app.rdynamic = true;

    b.installArtifact(server);
    b.installArtifact(wasm_app);
    b.installArtifact(dev_server);

    const run_server = b.addRunArtifact(dev_server);
    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_server.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_server.addArgs(args);
    }

    const run_db = b.addRunArtifact(sqlite3_shell);
    run_db.addArg("app.db");
    if (b.args) |args| {
        run_db.addArgs(args);
    }

    const fetch_model = b.addSystemCommand(&.{ "curl", "-fL", "--retry", "3", "-o" });
    const model_output = fetch_model.addOutputFileArg("model.onnx");
    fetch_model.addArg(model_url);
    const install_model = b.addInstallFileWithDir(model_output, .bin, "model.onnx");

    const server_test = b.addRunArtifact(b.addTest(.{
        .root_module = modules.server,
    }));
    const yolo_test = b.addRunArtifact(b.addTest(.{
        .root_module = createModule(b, "yolo.zig", target, optimize),
    }));
    const preprocess_test = b.addRunArtifact(b.addTest(.{
        .root_module = createModule(b, "image_preprocess.zig", target, optimize),
    }));
    const decode_test_module = createModule(b, "image_decode.zig", target, optimize);
    decode_test_module.addImport("zigimg", zigimg.module("zigimg"));
    const decode_test = b.addRunArtifact(b.addTest(.{
        .root_module = decode_test_module,
    }));

    b.step("db", "Run sqlite3 shell").dependOn(&run_db.step);
    b.step("model", "Download model.onnx into zig-out/bin").dependOn(&install_model.step);
    run_server.step.dependOn(&install_model.step);
    b.step("run", "Run dev server").dependOn(&run_server.step);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&server_test.step);
    test_step.dependOn(&yolo_test.step);
    test_step.dependOn(&preprocess_test.step);
    test_step.dependOn(&decode_test.step);

    const finish = try std.time.Instant.now();
    const duration: f64 = @floatFromInt(finish.since(start));
    std.debug.print("graph build duration {d:.3}ms \n", .{duration / std.time.ns_per_ms});
}
