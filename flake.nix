{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem(system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; allowBroken = true; };
        };
        tensorrt = (pkgs.cudaPackages_11.tensorrt_8_6.overrideAttrs (prev: next: {
          dontCheckForBrokenSymlinks = true;
        }));
      in
      with pkgs;
      {
        devShells.default = mkShell rec {
          nativeBuildInputs = [
            (python310.withPackages (p: with p; [
              virtualenv
              pip
            ]))
          ];

          buildInputs = [
            stdenv.cc.cc.lib
            zlib
            libGL
            glib
            tensorrt
          ] ++ (with cudaPackages_11; [
            cudatoolkit
            libcufft
            libcurand
            libcusparse
            libcusolver
          ]);

          shellHook = ''
            if [[ ! -d .hack ]]; then
              mkdir ./.hack
              ln -s ${tensorrt}/lib/libnvinfer.so.8.6.1 ./.hack/libnvinfer.so.7
              ln -s ${tensorrt}/lib/libnvinfer_plugin.so.8.6.1 ./.hack/libnvinfer_plugin.so.7
            fi
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$PWD/.hack:$LD_LIBRARY_PATH"
          '';

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
        };
      }
    );
}
