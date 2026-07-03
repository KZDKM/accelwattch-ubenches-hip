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
      rocrand hiprand
      rocwmma
      rocblas hipblas
      hipblas-common
    ];
  };
  rocmLlvm = pkgs.rocmPackages.llvm.clang;
  gcc = pkgs.gcc.cc;
  gccVer = gcc.version;
  triple = pkgs.stdenv.hostPlatform.config;
  glibcDev = pkgs.glibc.dev;
  oldPkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/nixos-23.11.tar.gz";
  }) { config.allowUnfree = true; config.cudaSupport = true; };
  cuda11 = oldPkgs.cudaPackages_11_8.cudatoolkit;
in
pkgs.mkShell {
  packages = [
    rocmToolkit
    rocm.hipify
    cuda11
    rocmLlvm
    pkgs.gcc
  ];

  CUDA_PATH = cuda11;

  shellHook = ''
        export HIPIFY_CLANG_RESOURCE_DIR="$(${rocmLlvm}/bin/clang -print-resource-dir)"

        # C++ standard library include paths for the host-compilation pass
        export CXX_STDLIB_INCLUDES="\
          -isystem ${gcc}/include/c++/${gccVer} \
          -isystem ${gcc}/include/c++/${gccVer}/${triple} \
          -isystem ${gcc}/lib/gcc/${triple}/${gccVer}/include \
          -isystem ${glibcDev}/include"
        export HIPCC_COMPILE_FLAGS_APPEND="-I${rocmToolkit}/include -I${rocm.hipblas}/include -I${rocm.hipblas}/include/hipblas"
        export HIPCC_LINK_FLAGS_APPEND="-L${rocmToolkit}/lib"
        export ROCM_PATH="${rocmToolkit}"

        echo "resource dir: $HIPIFY_CLANG_RESOURCE_DIR"
        echo "stdlib includes: $CXX_STDLIB_INCLUDES"
  '';
}
