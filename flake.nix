{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };

  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        gcc
        gnumake
        cmake
        ninja
        pkg-config

        libpng
        zlib
        tbb
        gbenchmark

        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
        gst_all_1.gst-plugins-good
        gst_all_1.gst-plugins-bad
        gst_all_1.gst-plugins-ugly

        cudaPackages.cuda_nvcc
        cudaPackages.cuda_cudart
        cudaPackages.cuda_cudart.static
        cudaPackages.cudatoolkit

        sysprof
      ];

      shellHook = ''
        export CC=${pkgs.gcc}/bin/gcc
        export CXX=${pkgs.gcc}/bin/g++

        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
        export CUDAHOSTCXX=$CXX

        export PATH=$CUDA_PATH/bin:$PATH
        export LD_LIBRARY_PATH=/run/opengl-driver/lib:$CUDA_PATH/lib:$LD_LIBRARY_PATH
      '';
    };
  };
}
