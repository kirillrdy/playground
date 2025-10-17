let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/archive/d07d03d0bbe235eb05a30ecbe95ce8ca994cb1aa.tar.gz";
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
