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
          config = { allowUnfree = true; };
        };
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
          ] ++ (with cudaPackages; [
            cudatoolkit
            tensorrt_8_6
            libcufft
            libcurand
            libcusparse
            libcusolver
          ]);

          shellHook = ''
            if [[ ! -d .hack ]]; then
              mkdir ./.hack
              ln -s ${cudaPackages.tensorrt_8_6}/lib/libnvinfer.so.8.6.1 ./.hack/libnvinfer.so.7
              ln -s ${cudaPackages.tensorrt_8_6}/lib/libnvinfer_plugin.so.8.6.1 ./.hack/libnvinfer_plugin.so.7
            fi
            export LD_LIBRARY_PATH="$PWD/.hack:$LD_LIBRARY_PATH"
          '';

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
        };
      }
    );
}
