const std = @import("std");
const onnx_build = @import("onnx");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_backend: onnx_build.Backend = if (target.result.os.tag.isDarwin()) .metal else .opencl;
    const backend = b.option(onnx_build.Backend, "backend", "ONNX execution backend: OpenCL, Metal, or CUDA") orelse default_backend;
    const cuda_arch = b.option([]const u8, "sm", "CUDA compute capability") orelse onnx_build.default_cuda_arch;
    const half = b.option(bool, "half", "Store float tensors on the device as halves") orelse (backend != .cuda);

    const sam3 = b.dependency("sam3", .{
        .target = target,
        .optimize = optimize,
        .backend = backend,
        .sm = cuda_arch,
        .half = half,
    });
    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    const native_main = b.createModule(.{
        .root_source_file = b.path("native_main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "sam3", .module = sam3.module("sam3") }},
    });

    if (target.result.os.tag.isDarwin()) {
        const macos_mod = b.createModule(.{
            .root_source_file = b.path("macos/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = sam3.module("sam3") },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
                .{ .name = "native_main", .module = native_main },
            },
        });
        macos_mod.addIncludePath(b.path("macos"));
        macos_mod.addCSourceFile(.{ .file = b.path("macos/bridge.m"), .flags = &.{"-fobjc-arc"} });
        macos_mod.link_libc = true;
        macos_mod.linkSystemLibrary("objc", .{});
        macos_mod.linkFramework("Foundation", .{});
        macos_mod.linkFramework("AppKit", .{});
        macos_mod.linkFramework("QuartzCore", .{});
        macos_mod.linkFramework("UniformTypeIdentifiers", .{});

        const exe = b.addExecutable(.{ .name = "sam3-macos", .root_module = macos_mod });
        b.installArtifact(exe);
        const run = b.step("run-macos", "Run the native macOS UI");
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.setCwd(b.path("."));
        run.dependOn(&run_cmd.step);
    }

    if (target.result.os.tag == .linux) {
        const linux_mod = b.createModule(.{
            .root_source_file = b.path("linux/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = if (optimize == .ReleaseFast) true else null,
            .imports = &.{
                .{ .name = "sam3", .module = sam3.module("sam3") },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
                .{ .name = "native_main", .module = native_main },
            },
        });
        linux_mod.link_libc = true;

        const exe = b.addExecutable(.{ .name = "sam3-linux", .root_module = linux_mod });
        b.installArtifact(exe);
        const run = b.step("run-linux", "Run the native Linux Wayland UI");
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.setCwd(b.path("."));
        run.dependOn(&run_cmd.step);
        b.step("run-wayland", "Alias for run-linux").dependOn(&run_cmd.step);
    }

    const test_step = b.step("test", "Run app tests");
    if (target.result.os.tag == .linux) {
        const font_test = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("linux/font.zig"),
            .target = target,
            .optimize = optimize,
        }) });
        test_step.dependOn(&b.addRunArtifact(font_test).step);
    }
}
