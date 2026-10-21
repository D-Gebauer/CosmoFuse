"""
Backend abstraction layer for CPU (NumPy/Numba) and GPU (CuPy/CUDA) execution.

All correlation measurement kernels exist in two forms:
  - **CPU**: Numba @njit functions with parallel=True for multi-core execution
  - **GPU**: CuPy ElementwiseKernels for simple per-pair operations, and
    compiled CUDA kernels (.cu files in cuda/) for fused/vectorised
    tomographic operations, launched through :class:`raw.TypedKernel`
    (every launch checked against the signature parsed from the source)

One module per kernel family, CPU kernel and GPU wrapper side by side:
:mod:`.aperture` (M_ap, M_g), :mod:`.density` (ξ_g, ξ_t), :mod:`.shear`
(ξ±), :mod:`.three_by_two` (the fused 3x2pt path), :mod:`.degrade` (treecode
rows), :mod:`.threepoint` (step 1 of Global3PCF); :mod:`.combinations` holds
the tomographic layouts of the tiled kernels, :mod:`.raw` compiling and
launching, :mod:`.core` the :class:`Backend` that holds the active kernel set
and provides to_device()/to_numpy() for data movement.
"""

from .core import Backend, get_backend, safe_divide

__all__ = ["Backend", "get_backend", "safe_divide"]
