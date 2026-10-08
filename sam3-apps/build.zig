const std = @import("std");
const Backend = enum { cuda, opencl, metal };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_backend: Backend = blk: {
        if (target.result.os.tag.isDarwin()) break :blk .metal;
        if (target.result.os.tag == .linux) {
            if (std.Io.Dir.accessAbsolute(b.graph.io, "/dev/nvidia0", .{})) |_| {
                break :blk .cuda;
            } else |_| {}
        }
        break :blk .opencl;
    };

    const backend = b.option(Backend, "backend", "Inference backend: OpenCL, Metal, or CUDA") orelse default_backend;
    const cuda_arch = b.option([]const u8, "sm", "CUDA compute capability");
    const half = b.option(bool, "half", "Store float tensors on the device as halves");

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
    const zimo = b.dependency("zimo", .{});
    const log = b.createModule(.{
        .root_source_file = b.path("log.zig"),
        .target = target,
        .optimize = optimize,
    });

    const native_main = b.createModule(.{
        .root_source_file = b.path("native_main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sam3", .module = sam3.module("sam3") },
            .{ .name = "log", .module = log },
        },
    });

    const vdb = b.createModule(.{
        .root_source_file = b.path("src/vdb/vdb.zig"),
        .target = target,
        .optimize = optimize,
    });

    const core = b.createModule(.{
        .root_source_file = b.path("src/core/core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sam3", .module = sam3.module("sam3") },
            .{ .name = "zigimg", .module = zigimg.module("zigimg") },
            .{ .name = "zimo", .module = zimo.module("zimo") },
            .{ .name = "vdb", .module = vdb },
            .{ .name = "log", .module = log },
        },
    });
    core.link_libc = true;

    const run_step = b.step("run", "Run the native app");

    if (target.result.os.tag.isDarwin()) {
        const macos_mod = b.createModule(.{
            .root_source_file = b.path("macos/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sam3", .module = sam3.module("sam3") },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
                .{ .name = "zimo", .module = zimo.module("zimo") },
                .{ .name = "native_main", .module = native_main },
                .{ .name = "log", .module = log },
                .{ .name = "vdb", .module = vdb },
                .{ .name = "core", .module = core },
            },
        });
        macos_mod.addIncludePath(b.path("macos"));
        const bridge_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
        });
        bridge_mod.addIncludePath(b.path("macos"));
        bridge_mod.addCSourceFile(.{ .file = b.path("macos/bridge.m"), .flags = &.{"-fobjc-arc"} });
        bridge_mod.link_libc = true;
        const bridge_obj = b.addObject(.{
            .name = "macos_bridge",
            .root_module = bridge_mod,
        });
        macos_mod.addObject(bridge_obj);
        macos_mod.link_libc = true;
        macos_mod.linkSystemLibrary("objc", .{});
        macos_mod.linkFramework("Foundation", .{});
        macos_mod.linkFramework("AppKit", .{});
        macos_mod.linkFramework("QuartzCore", .{});
        macos_mod.linkFramework("UniformTypeIdentifiers", .{});
        macos_mod.linkFramework("AVFoundation", .{});
        macos_mod.linkFramework("CoreMedia", .{});
        macos_mod.linkFramework("CoreVideo", .{});

        const exe = b.addExecutable(.{ .name = "sam3-macos", .root_module = macos_mod });
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.setCwd(b.path("."));
        run_cmd.addPassthruArgs();
        run_step.dependOn(&run_cmd.step);
        b.step("run-macos", "Run the native macOS UI").dependOn(&run_cmd.step);
        b.step("run-darwin", "Alias for run-macos").dependOn(&run_cmd.step);
    }

    if (target.result.os.tag == .linux) {
        const linux_mod = b.createModule(.{
            .root_source_file = b.path("linux/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sam3", .module = sam3.module("sam3") },
                .{ .name = "zigimg", .module = zigimg.module("zigimg") },
                .{ .name = "zimo", .module = zimo.module("zimo") },
                .{ .name = "native_main", .module = native_main },
                .{ .name = "log", .module = log },
                .{ .name = "vdb", .module = vdb },
                .{ .name = "core", .module = core },
            },
        });
        linux_mod.link_libc = true;
        linux_mod.addCSourceFile(.{ .file = b.path("linux/video.c"), .flags = &.{} });

        const exe = b.addExecutable(.{ .name = "sam3-linux", .root_module = linux_mod });
        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.setCwd(b.path("."));
        run_cmd.addPassthruArgs();
        run_step.dependOn(&run_cmd.step);
        b.step("run-linux", "Run the native Linux Wayland UI").dependOn(&run_cmd.step);
        b.step("run-wayland", "Alias for run-linux").dependOn(&run_cmd.step);
    }

    const test_step = b.step("test", "Run app tests");
    if (target.result.os.tag.isDarwin()) {
        const bridge_test_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        bridge_test_mod.addIncludePath(b.path("macos"));
        bridge_test_mod.addCSourceFile(.{ .file = b.path("macos/bridge_test.m"), .flags = &.{"-fobjc-arc"} });
        bridge_test_mod.linkSystemLibrary("objc", .{});
        for ([_][]const u8{ "Foundation", "AppKit", "QuartzCore", "UniformTypeIdentifiers", "AVFoundation", "CoreMedia", "CoreVideo" }) |framework| {
            bridge_test_mod.linkFramework(framework, .{});
        }
        const bridge_test = b.addExecutable(.{ .name = "macos-bridge-test", .root_module = bridge_test_mod });
        test_step.dependOn(&b.addRunArtifact(bridge_test).step);
    }
    const vdb_test = b.addTest(.{
        .root_module = vdb,
    });
    vdb_test.root_module.link_libc = true;
    test_step.dependOn(&b.addRunArtifact(vdb_test).step);

    const core_test = b.addTest(.{
        .root_module = core,
    });
    test_step.dependOn(&b.addRunArtifact(core_test).step);

    if (target.result.os.tag == .linux) {
        const wayland_test = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("linux/wayland.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }) });
        test_step.dependOn(&b.addRunArtifact(wayland_test).step);
        const font_test = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("linux/font.zig"),
            .target = target,
            .optimize = optimize,
        }) });
        test_step.dependOn(&b.addRunArtifact(font_test).step);
    }
}
