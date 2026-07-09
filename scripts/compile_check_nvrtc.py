"""Compile every CosmoFuse CUDA kernel template to PTX with NVRTC.

Needs NO GPU and no CUDA driver: NVRTC is a pure compiler library. On a
machine without the CUDA toolkit, the pip wheels are enough:

    pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
    python scripts/compile_check_nvrtc.py

(`nvidia-cuda-nvrtc` provides libnvrtc; `nvidia-cuda-runtime` provides the
cuComplex.h header.)  Every production template instantiation of every
kernel family is added as a name expression, so template errors surface
too, not just parse errors.  Exit code 0 = all kernels compile.

This script is intentionally standalone (no CosmoFuse import) so it runs
in a minimal environment.
"""

import ctypes
import ctypes.util
import sys
from pathlib import Path

CUDA_DIR = Path(__file__).resolve().parent.parent / "src" / "CosmoFuse" / "cuda"

# Representative production template instantiations per kernel source.
NAME_EXPRESSIONS = {
    "tomo_vectorized_xipm.cu": [
        "gpu_fused_tomo_reduce_xipm<double, cuFloatComplex, 1, int>",
        "gpu_fused_tomo_reduce_xipm<double, cuFloatComplex, 5, int>",
        "gpu_fused_tomo_reduce_xipm<double, cuDoubleComplex, 5, long long>",
        "gpu_fused_tomo_reduce_xipm<float, cuFloatComplex, 2, int>",
    ],
    "density_density_tomo_vectorized.cu": [
        "gpu_fused_tomo_reduce_dd<double, 1, int>",
        "gpu_fused_tomo_reduce_dd<double, 5, long long>",
        "gpu_fused_tomo_reduce_dd<float, 2, int>",
    ],
    "density_shear_tomo_vectorized.cu": [
        "gpu_fused_tomo_reduce_ds<double, cuFloatComplex, 5, 5, int>",
        "gpu_fused_tomo_reduce_ds<double, cuDoubleComplex, 1, 1, long long>",
        "gpu_fused_tomo_reduce_ds<float, cuFloatComplex, 2, 3, int>",
    ],
    "aperture_tomo.cu": [
        "gpu_aperture_shear_tomo<double, float>",
        "gpu_aperture_shear_tomo<double, double>",
        "gpu_aperture_shear_tomo<float, float>",
        "gpu_aperture_density_tomo<double, float>",
        "gpu_aperture_density_tomo<double, double>",
        "gpu_aperture_density_tomo<float, float>",
    ],
    "tomo_fused_3x2pt.cu": [
        "gpu_3x2pt_tomo_fused<double, cuFloatComplex, int, float, 5, 5>",
        "gpu_3x2pt_tomo_fused<double, cuDoubleComplex, long long, double, 2, 2>",
        "gpu_3x2pt_tomo_fused<float, cuFloatComplex, int, float, 1, 1>",
    ],
}


def _candidate_paths(package, subdir, pattern):
    try:
        import importlib.util

        spec = importlib.util.find_spec(package)
        if spec is not None and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                yield from sorted(Path(location).glob(f"{subdir}/{pattern}"))
    except (ImportError, ValueError):
        return


def find_libnvrtc():
    found = ctypes.util.find_library("nvrtc")
    if found:
        return found
    for path in _candidate_paths("nvidia.cuda_nvrtc", "lib", "libnvrtc.so*"):
        if "builtins" not in path.name and "alt" not in path.name:
            return str(path)
    for path in (Path("/opt/cuda/lib64"), Path("/usr/local/cuda/lib64")):
        for candidate in sorted(path.glob("libnvrtc.so*")):
            return str(candidate)
    return None


def find_include_dir():
    for path in _candidate_paths("nvidia.cuda_runtime", "include", "cuComplex.h"):
        return str(path.parent)
    for path in (Path("/opt/cuda/include"), Path("/usr/local/cuda/include")):
        if (path / "cuComplex.h").exists():
            return str(path)
    return None


def prepare_source(filename):
    common = (CUDA_DIR / "common.cuh").read_text(encoding="utf-8")
    source = (CUDA_DIR / filename).read_text(encoding="utf-8")
    return source.replace("__COMMON_CUDA_SOURCE__", common)


def main():
    lib_path = find_libnvrtc()
    include_dir = find_include_dir()
    if lib_path is None or include_dir is None:
        print(
            "NVRTC or CUDA headers not found. Install them (no GPU needed):\n"
            "    pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12"
        )
        return 2

    nvrtc = ctypes.CDLL(lib_path)
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p

    def check(result, context):
        if result != 0:
            message = nvrtc.nvrtcGetErrorString(result).decode()
            raise RuntimeError(f"{context}: {message}")

    options = [
        f"--include-path={include_dir}".encode(),
        b"--std=c++14",
        b"--use_fast_math",
        b"--gpu-architecture=compute_75",
    ]
    options_array = (ctypes.c_char_p * len(options))(*options)

    failures = 0
    for filename, name_expressions in NAME_EXPRESSIONS.items():
        source = prepare_source(filename)
        prog = ctypes.c_void_p()
        check(
            nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog),
                source.encode(),
                filename.encode(),
                0,
                None,
                None,
            ),
            f"nvrtcCreateProgram({filename})",
        )
        try:
            for expr in name_expressions:
                check(
                    nvrtc.nvrtcAddNameExpression(prog, expr.encode()),
                    f"nvrtcAddNameExpression({expr})",
                )

            result = nvrtc.nvrtcCompileProgram(prog, len(options), options_array)
            log_size = ctypes.c_size_t()
            check(
                nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size)),
                "nvrtcGetProgramLogSize",
            )
            log = b""
            if log_size.value > 1:
                buffer = ctypes.create_string_buffer(log_size.value)
                check(nvrtc.nvrtcGetProgramLog(prog, buffer), "nvrtcGetProgramLog")
                log = buffer.value

            if result != 0:
                failures += 1
                print(f"FAIL {filename}")
                print(log.decode(errors="replace"))
                continue

            # Confirm each template instantiation was actually generated.
            for expr in name_expressions:
                lowered = ctypes.c_char_p()
                check(
                    nvrtc.nvrtcGetLoweredName(
                        prog, expr.encode(), ctypes.byref(lowered)
                    ),
                    f"nvrtcGetLoweredName({expr})",
                )
            print(f"ok   {filename}  ({len(name_expressions)} instantiations)")
            if log:
                print(log.decode(errors="replace"))
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    if failures:
        print(f"{failures} kernel source(s) failed to compile.")
        return 1
    print("All CUDA kernel templates compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
