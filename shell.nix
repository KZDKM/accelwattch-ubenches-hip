{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  rocm = pkgs.rocmPackages;

  rocmToolkit = pkgs.symlinkJoin {
    name = "rocm-toolkit";
    paths = with rocm; [
      clr
      hipcc
      rocm-runtime
      rocm-device-libs
      rocm-comgr
      hip-common
      rocminfo
      rocm-smi
    ];
  };
  rocmLlvm = pkgs.rocmPackages.llvm.clang;
  gcc = pkgs.gcc.cc;
  gccVer = gcc.version;
  triple = pkgs.stdenv.hostPlatform.config;
  glibcDev = pkgs.glibc.dev;
in
pkgs.mkShell {
  packages = [
    rocmToolkit
    rocm.hipify
    pkgs.cudatoolkit
    rocmLlvm
    pkgs.gcc
  ];

  CUDA_PATH = pkgs.cudatoolkit;

  shellHook = ''
        export HIPIFY_CLANG_RESOURCE_DIR="$(${rocmLlvm}/bin/clang -print-resource-dir)"

        # C++ standard library include paths for the host-compilation pass
        export CXX_STDLIB_INCLUDES="\
    -isystem ${gcc}/include/c++/${gccVer} \
    -isystem ${gcc}/include/c++/${gccVer}/${triple} \
    -isystem ${gcc}/lib/gcc/${triple}/${gccVer}/include \
    -isystem ${glibcDev}/include"

        echo "resource dir: $HIPIFY_CLANG_RESOURCE_DIR"
        echo "stdlib includes: $CXX_STDLIB_INCLUDES"
  '';
}
