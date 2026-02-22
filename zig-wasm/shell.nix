let
  pkgs = import (builtins.getFlake "nixpkgs").outPath {
    system = builtins.currentSystem;
    config.allowUnfree = true;
  };
  ortCuda = pkgs.onnxruntime.override { cudaSupport = true; };
  py = pkgs.python3.withPackages (ps: [
    ps.numpy
    ps.av
    (ps.onnxruntime.override { onnxruntime = ortCuda; })
  ]);
in
pkgs.mkShell {
  packages = [py];
}
