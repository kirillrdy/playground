const std = @import("std");
const wasm_app_name = @import("server.zig").wasm_app_name;
const server_name = @import("server.zig").server_name;

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Nix dependencies
    const onnx_dev = nixPkg(b, "onnx_dev", "(pkgs.onnxruntime.override { cudaSupport = true; }).dev");
    const onnx_lib = nixPkg(b, "onnx_lib", "(pkgs.onnxruntime.override { cudaSupport = true; })");
    const ffmpeg_dev = nixPkg(b, "ffmpeg_dev", "pkgs.ffmpeg.dev");
    const ffmpeg_lib = nixPkg(b, "ffmpeg_lib", "pkgs.ffmpeg.lib");
    const cuda = nixPkg(b, "cuda", "pkgs.cudaPackages.cudatoolkit");
    const cudart = nixPkg(b, "cudart", "pkgs.cudaPackages.cuda_cudart");
    const nvcc_pkg = nixPkg(b, "nvcc_pkg", "pkgs.cudaPackages.cuda_nvcc");

    const zigimg = b.dependency("zigimg", .{ .target = target, .optimize = optimize });
    const vy_opts = b.addOptions();

    const server_mod = b.createModule(.{
        .root_source_file = b.path("server.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_mod.addOptions("config", vy_opts);
    server_mod.addImport("zigimg", zigimg.module("zigimg"));
    server_mod.addImport("httpz", b.dependency("httpz", .{}).module("httpz"));
    server_mod.addIncludePath(onnx_dev.path(b, "include"));
    server_mod.addIncludePath(ffmpeg_dev.path(b, "include"));
    server_mod.addIncludePath(cuda.path(b, "include"));

    const preprocess_obj = b.addSystemCommand(&.{ "sh", "-c", "$1 -arch=sm_89 -c $2 -o $3 -I$4", "--" });
    preprocess_obj.addFileArg(nvcc_pkg.path(b, "bin/nvcc"));
    preprocess_obj.addFileArg(b.path("preprocess.cu"));
    const preprocess_o = preprocess_obj.addOutputFileArg("preprocess.o");
    preprocess_obj.addDirectoryArg(cuda.path(b, "include"));

    const server = b.addExecutable(.{ .name = server_name, .root_module = server_mod });
    server.linkLibC();
    server.addObjectFile(preprocess_o);
    server.addLibraryPath(onnx_lib.path(b, "lib"));
    server.addLibraryPath(ffmpeg_lib.path(b, "lib"));
    server.addLibraryPath(cudart.path(b, "lib"));
    server.linkSystemLibrary("onnxruntime");
    server.linkSystemLibrary("cudart");
    server.linkSystemLibrary("avformat");
    server.linkSystemLibrary("avcodec");
    server.linkSystemLibrary("avutil");
    server.linkSystemLibrary("avfilter");
    b.installArtifact(server);

    const video_yolo_mod = b.createModule(.{
        .root_source_file = b.path("video_yolo.zig"),
        .target = target,
        .optimize = optimize,
    });
    video_yolo_mod.addOptions("config", vy_opts);
    video_yolo_mod.addIncludePath(onnx_dev.path(b, "include"));
    video_yolo_mod.addIncludePath(ffmpeg_dev.path(b, "include"));
    video_yolo_mod.addIncludePath(cuda.path(b, "include"));

    const video_yolo = b.addExecutable(.{ .name = "video_yolo", .root_module = video_yolo_mod });
    video_yolo.linkLibC();
    video_yolo.addObjectFile(preprocess_o);
    video_yolo.addLibraryPath(onnx_lib.path(b, "lib"));
    video_yolo.addLibraryPath(ffmpeg_lib.path(b, "lib"));
    video_yolo.addLibraryPath(cudart.path(b, "lib"));
    video_yolo.linkSystemLibrary("onnxruntime");
    video_yolo.linkSystemLibrary("cudart");
    video_yolo.linkSystemLibrary("avformat");
    video_yolo.linkSystemLibrary("avcodec");
    video_yolo.linkSystemLibrary("avutil");
    video_yolo.linkSystemLibrary("avfilter");
    b.installArtifact(video_yolo);

    const wasm_app = b.addExecutable(.{
        .name = wasm_app_name,
        .root_module = b.createModule(.{
            .root_source_file = b.path("wasm.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = optimize,
        }),
    });
    wasm_app.entry = .disabled;
    wasm_app.rdynamic = true;
    b.installArtifact(wasm_app);

    const dev_server = b.addExecutable(.{
        .name = "dev_server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("dev_server.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(dev_server);

    // Model presets
    const model_preset = b.option([]const u8, "model", "YOLO model preset (n, s, m)") orelse "n";
    const model_url = b.option([]const u8, "model-url", "Override model URL") orelse switch (model_preset[0]) {
        's' => "https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/main/onnx/model.onnx",
        'm' => "https://huggingface.co/onnx-community/yolo26m-ONNX/resolve/main/onnx/model.onnx",
        else => "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx",
    };
    const fetch_model = b.addSystemCommand(&.{ "curl", "-fL", "--retry", "3", "-o" });
    const model_file = fetch_model.addOutputFileArg("model.onnx");
    fetch_model.addArg(model_url);
    vy_opts.addOptionPath("model_path", model_file);
    b.getInstallStep().dependOn(&b.addInstallFileWithDir(model_file, .bin, "model.onnx").step);

    // Run steps
    const run_dev = b.addRunArtifact(dev_server);
    run_dev.step.dependOn(b.getInstallStep());
    b.step("run", "Run dev server").dependOn(&run_dev.step);

    const run_vy = b.addRunArtifact(video_yolo);
    run_vy.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_vy.addArgs(args);
    b.step("run-video", "Run video_yolo").dependOn(&run_vy.step);

    const test_step = b.step("test", "Run tests");
    const tests = [_][]const u8{ "yolo.zig", "image_preprocess.zig", "image_decode.zig" };
    for (tests) |t| {
        const t_mod = b.createModule(.{ .root_source_file = b.path(t), .target = target, .optimize = optimize });
        if (std.mem.eql(u8, t, "image_decode.zig")) t_mod.addImport("zigimg", zigimg.module("zigimg"));
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = t_mod })).step);
    }
}

fn nixPkg(b: *std.Build, name: []const u8, pkg_expr: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ "sh", "-c",
        \\set -euo pipefail
        \\out="$1"
        \\expr="$2"
        \\full_expr="let pkgs = import (builtins.getFlake \"nixpkgs\").outPath { system = \"$(nix eval --raw --impure --expr builtins.currentSystem)\"; config.allowUnfree = true; }; in $expr"
        \\path="$(nix build --impure --no-link --print-out-paths --expr "$full_expr")"
        \\ln -s "$path" "$out"
        , "--" });
    const out = run.addOutputFileArg(name);
    run.addArg(pkg_expr);
    return out;
}
