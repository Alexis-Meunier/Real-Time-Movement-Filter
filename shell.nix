{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
let
  tbbDev = "/nix/store/2clcgn9d7rlyhdkpc1gx9frg14vm2k07-tbb-2021.11.0-dev";
  tbbLib = builtins.head (
    builtins.filter (p: builtins.pathExists "${p}/lib/libtbb.so.12")
      (map (p: "/nix/store/${p}") (builtins.attrNames (builtins.readDir /nix/store)))
  );
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    libpng
    zlib
    tbb
    gbenchmark
    cmake
    cudaPackages.cudatoolkit
  ];

  shellHook = ''
    TBB_LIB=$(dirname $(find /nix/store -name "libtbb.so.12" 2>/dev/null | head -1))
    export TBB_DIR="${tbbDev}/lib/cmake/TBB"
    export CMAKE_PREFIX_PATH="${pkgs.gbenchmark}:$CMAKE_PREFIX_PATH"
    export LIBRARY_PATH="$TBB_LIB:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="$TBB_LIB:$LD_LIBRARY_PATH"
    echo "TBB lib: $TBB_LIB"
  '';
}
