let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/archive/01f116e4df6a15f4ccdffb1bcd41096869fb385c.tar.gz";
  pkgs = import nixpkgs { };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    zig_0_15
    zls_0_15
    openssl
    tailwindcss_4
    postgresql
  ];
}
