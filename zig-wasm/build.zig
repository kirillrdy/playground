const std = @import("std");
const wasm_app_name = @import("server.zig").wasm_app_name;
const server_name = @import("server.zig").server_name;
const string = []const u8;

fn createModule(b: *std.Build, src: string, target: anytype, optimize: anytype) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(src), .target = target, .optimize = optimize });
}

fn optionPath(path: ?[]const u8) ?std.Build.LazyPath {
    if (path) |p| return .{ .cwd_relative = p };
    return null;
}

fn modelPresetUrl(preset: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, preset, "n")) return "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx";
    if (std.mem.eql(u8, preset, "s")) return "https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/main/onnx/model.onnx";
    if (std.mem.eql(u8, preset, "m")) return "https://huggingface.co/onnx-community/yolo26m-ONNX/resolve/main/onnx/model.onnx";
    return null;
}

fn nixBuildAttrLink(b: *std.Build, out_name: []const u8, attr: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{
        "sh", "-c",
        \\set -euo pipefail
        \\out="$1"
        \\attr="$2"
        \\path="$(nix build --no-link --print-out-paths "$attr")"
        \\ln -s "$path" "$out"
        ,
        "--",
    });
    const out = run.addOutputFileArg(out_name);
    run.addArg(attr);
    return out;
}

fn nixBuildExprLink(b: *std.Build, out_name: []const u8, expr: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{
        "sh", "-c",
        \\set -euo pipefail
        \\out="$1"
        \\expr="$2"
        \\path="$(nix build --impure --no-link --print-out-paths --expr "$expr")"
        \\ln -s "$path" "$out"
        ,
        "--",
    });
    const out = run.addOutputFileArg(out_name);
    run.addArg(expr);
    return out;
}

pub fn build(b: *std.Build) !void {
    const start = try std.time.Instant.now();
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const modules = .{
        .server = createModule(b, "server.zig", target, optimize),
        .dev_server = createModule(b, "dev_server.zig", target, optimize),
        .video_yolo = createModule(b, "video_yolo.zig", target, optimize),
        .wasm = createModule(b, "wasm.zig", b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }), optimize),
    };
    const onnx_include_path = b.option([]const u8, "onnx-include", "Path to directory containing onnxruntime_c_api.h");
    const onnx_lib_path = b.option([]const u8, "onnx-lib", "Path to directory containing libonnxruntime");
    const ffmpeg_include_path = b.option([]const u8, "ffmpeg-include", "Path to directory containing FFmpeg headers");
    const ffmpeg_lib_path = b.option([]const u8, "ffmpeg-lib", "Path to directory containing FFmpeg libraries");
    const cuda_include_path = b.option([]const u8, "cuda-include", "Path to directory containing CUDA headers");
    const cuda_lib_path = b.option([]const u8, "cuda-lib", "Path to directory containing CUDA runtime libraries");
    const nvcc_path = b.option([]const u8, "nvcc", "Path to nvcc compiler binary");
    const model_preset = b.option([]const u8, "model", "YOLO model preset: n, s, or m") orelse "s";
    const preset_model_url = modelPresetUrl(model_preset) orelse {
        std.log.err("invalid -Dmodel={s}; expected one of: n, s, m", .{model_preset});
        return error.InvalidModelPreset;
    };
    const model_url = b.option([]const u8, "model-url", "URL to download ONNX model from (overrides -Dmodel)") orelse preset_model_url;
    const onnx_gpu_dev_expr =
        \\let
        \\  flake = builtins.getFlake "nixpkgs";
        \\  pkgs = import flake.outPath {
        \\    system = builtins.currentSystem;
        \\    config.allowUnfree = true;
        \\  };
        \\  ort = pkgs.onnxruntime.override { cudaSupport = true; };
        \\in ort.dev
    ;
    const onnx_gpu_out_expr =
        \\let
        \\  flake = builtins.getFlake "nixpkgs";
        \\  pkgs = import flake.outPath {
        \\    system = builtins.currentSystem;
        \\    config.allowUnfree = true;
        \\  };
        \\  ort = pkgs.onnxruntime.override { cudaSupport = true; };
        \\in ort
    ;
    const cuda_toolkit_expr =
        \\let
        \\  flake = builtins.getFlake "nixpkgs";
        \\  pkgs = import flake.outPath {
        \\    system = builtins.currentSystem;
        \\    config.allowUnfree = true;
        \\  };
        \\in pkgs.cudaPackages.cudatoolkit
    ;
    const cuda_cudart_expr =
        \\let
        \\  flake = builtins.getFlake "nixpkgs";
        \\  pkgs = import flake.outPath {
        \\    system = builtins.currentSystem;
        \\    config.allowUnfree = true;
        \\  };
        \\in pkgs.cudaPackages.cuda_cudart
    ;
    const cuda_nvcc_expr =
        \\let
        \\  flake = builtins.getFlake "nixpkgs";
        \\  pkgs = import flake.outPath {
        \\    system = builtins.currentSystem;
        \\    config.allowUnfree = true;
        \\  };
        \\in pkgs.cudaPackages.cuda_nvcc
    ;
    const onnx_dev_root = optionPath(onnx_include_path) orelse nixBuildExprLink(b, "onnx_dev_root", onnx_gpu_dev_expr);
    const onnx_out_root = optionPath(onnx_lib_path) orelse nixBuildExprLink(b, "onnx_out_root", onnx_gpu_out_expr);
    const ffmpeg_dev_root = optionPath(ffmpeg_include_path) orelse nixBuildAttrLink(b, "ffmpeg_dev_root", "nixpkgs#ffmpeg.dev");
    const ffmpeg_out_root = optionPath(ffmpeg_lib_path) orelse nixBuildAttrLink(b, "ffmpeg_out_root", "nixpkgs#ffmpeg.lib");
    const cuda_dev_root = optionPath(cuda_include_path) orelse nixBuildExprLink(b, "cuda_dev_root", cuda_toolkit_expr);
    const cuda_out_root = optionPath(cuda_lib_path) orelse nixBuildExprLink(b, "cuda_out_root", cuda_cudart_expr);
    const cuda_nvcc_root = nixBuildExprLink(b, "cuda_nvcc_root", cuda_nvcc_expr);
    const resolved_nvcc = optionPath(nvcc_path) orelse cuda_nvcc_root.path(b, "bin/nvcc");

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
    modules.server.addIncludePath(onnx_dev_root.path(b, "include"));
    modules.video_yolo.addIncludePath(onnx_dev_root.path(b, "include"));
    modules.video_yolo.addIncludePath(ffmpeg_dev_root.path(b, "include"));
    modules.video_yolo.addIncludePath(cuda_dev_root.path(b, "include"));
    const video_yolo_options = b.addOptions();
    modules.video_yolo.addOptions("config", video_yolo_options);

    inline for (&.{"httpz"}) |dependency_name| {
        modules.server.addImport(dependency_name, b.dependency(dependency_name, .{}).module(dependency_name));
    }

    const server = b.addExecutable(.{ .name = server_name, .root_module = modules.server });
    server.linkLibrary(sqlite3_lib);
    server.addLibraryPath(onnx_out_root.path(b, "lib"));
    server.linkSystemLibrary("onnxruntime");

    const preprocess_obj = b.addSystemCommand(&.{
        "sh", "-c",
        \\set -euo pipefail
        \\nvcc="$1"
        \\src="$2"
        \\out="$3"
        \\cuda_include="$4"
        \\"$nvcc" -arch=sm_89 -c "$src" -o "$out" -I"$cuda_include"
        ,
        "--",
    });
    preprocess_obj.addFileArg(resolved_nvcc);
    preprocess_obj.addFileArg(b.path("preprocess.cu"));
    const preprocess_o = preprocess_obj.addOutputFileArg("preprocess.o");
    preprocess_obj.addDirectoryArg(cuda_dev_root.path(b, "include"));

    const video_yolo = b.addExecutable(.{ .name = "video_yolo", .root_module = modules.video_yolo });
    video_yolo.linkLibC();
    video_yolo.addObjectFile(preprocess_o);
    video_yolo.linkSystemLibrary("cudart");
    video_yolo.addIncludePath(onnx_dev_root.path(b, "include"));
    video_yolo.addIncludePath(ffmpeg_dev_root.path(b, "include"));
    video_yolo.addIncludePath(cuda_dev_root.path(b, "include"));
    video_yolo.addLibraryPath(onnx_out_root.path(b, "lib"));
    video_yolo.addLibraryPath(ffmpeg_out_root.path(b, "lib"));
    video_yolo.addLibraryPath(cuda_out_root.path(b, "lib"));
    video_yolo.linkSystemLibrary("onnxruntime");
    video_yolo.linkSystemLibrary("avformat");
    video_yolo.linkSystemLibrary("avcodec");
    video_yolo.linkSystemLibrary("avutil");
    video_yolo.linkSystemLibrary("avfilter");

    const dev_server = b.addExecutable(.{ .name = "dev_server", .root_module = modules.dev_server });
    dev_server.linkLibrary(sqlite3_lib);

    const wasm_app = b.addExecutable(.{ .name = wasm_app_name, .root_module = modules.wasm });
    wasm_app.entry = .disabled;
    wasm_app.rdynamic = true;

    b.installArtifact(server);
    b.installArtifact(video_yolo);
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
    video_yolo_options.addOptionPath("model_path", model_output);
    const install_model = b.addInstallFileWithDir(model_output, .bin, "model.onnx");
    b.getInstallStep().dependOn(&install_model.step);

    const run_video_yolo = b.addRunArtifact(video_yolo);
    run_video_yolo.step.dependOn(b.getInstallStep());
    run_video_yolo.step.dependOn(&install_model.step);
    if (b.args) |args| {
        run_video_yolo.addArgs(args);
    }

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
    b.step("run-video", "Run video_yolo detector").dependOn(&run_video_yolo.step);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&server_test.step);
    test_step.dependOn(&yolo_test.step);
    test_step.dependOn(&preprocess_test.step);
    test_step.dependOn(&decode_test.step);

    const finish = try std.time.Instant.now();
    const duration: f64 = @floatFromInt(finish.since(start));
    std.debug.print("graph build duration {d:.3}ms \n", .{duration / std.time.ns_per_ms});
}
