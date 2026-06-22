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
        cudaPackages.cuda_cudart
        cudaPackages.cuda_nvcc
        cudaPackages.cudnn
        gcc
        gnumake
        cmake
        pkg-config
      ];
      shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cuda_cudart}
        export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
        export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH
      '';
    };
  };
}
