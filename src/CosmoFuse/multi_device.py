"""Run one measurement across several GPUs by splitting the patch set.

Pair geometry is per patch and the estimators are per patch, so patches are
the natural unit of work: device ``d`` owns a contiguous range of patches and
a :class:`~CosmoFuse.correlations.Integrated3PCF` of its own, measurement calls
run in one thread per device (CuPy's current device is thread-local), and the
per-patch outputs are concatenated back in patch order.  The result is
identical to a single-device run -- the patch axis is just assembled from
several pieces.

Device memory scales as ``n_patches / n_devices`` per GPU, which is the
other reason to use this: it is how a geometry that does not fit one card
is measured without coarsening or dropping patches.

Use it through ``Integrated3PCF(..., device=[0, 1])``::

    corr = Integrated3PCF(nside, phi, theta, device=[0, 1], mask=mask,
                          nbins=8, theta_min=15, theta_max=250)
    corr.preprocess()                      # or corr.load_pairs(path)
    res = corr.gg(shear, w)                # res.M_a, res.xi_p, res.xi_m
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Methods measured per device and assembled on the patch axis.  Aperture
# outputs are (..., n_patches), pair outputs (..., n_patches, nbins).
_ARRAY_METHODS: Dict[str, int] = {"m_a": -1, "m_g": -1}
_RESULT_METHODS = frozenset({"gg", "kk", "kg", "three_by_two_pt"})
_FIELD_PATCH_AXIS: Dict[str, int] = {
    "M_a": -1, "M_g": -1, "xi_p": -2, "xi_m": -2, "xi_g": -2, "xi_t": -2,
}
# kept for callers that ask which methods a group dispatches
_PATCH_AXIS = tuple(_ARRAY_METHODS) + tuple(sorted(_RESULT_METHODS))


class MultiDeviceIntegrated3PCF:
    """A patch-parallel group of :class:`Integrated3PCF` objects, one per GPU."""

    def __init__(
        self,
        nside: int,
        phi_center: np.ndarray,
        theta_center: np.ndarray,
        *args: Any,
        devices: Sequence[int],
        **kwargs: Any,
    ) -> None:
        from .correlations import Integrated3PCF

        # Usually CUDA ids; anything Integrated3PCF accepts works, so that the
        # splitting can be exercised with ("cpu", "cpu").
        ids = [int(d) if isinstance(d, (int, np.integer)) else d for d in devices]
        if len(ids) == 0:
            raise ValueError("devices must name at least one device")
        int_ids = [d for d in ids if isinstance(d, int)]
        if len(set(int_ids)) != len(int_ids):
            raise ValueError(f"duplicate device ids in {ids}")
        phi = np.asarray(phi_center)
        theta = np.asarray(theta_center)
        if phi.size < len(ids):
            raise ValueError(
                f"{phi.size} patches cannot be split across {len(ids)} devices"
            )
        kwargs.pop("device", None)

        self.devices = ids
        self.n_patches = int(phi.size)
        bounds = np.linspace(0, self.n_patches, len(ids) + 1).astype(int)
        self.patch_ranges: List[Tuple[int, int]] = [
            (int(bounds[i]), int(bounds[i + 1])) for i in range(len(ids))
        ]
        self.parts = [
            Integrated3PCF(
                nside,
                phi[a:b],
                theta[a:b],
                *args,
                device=dev,
                **kwargs,
            )
            for dev, (a, b) in zip(ids, self.patch_ranges)
        ]

    # ---- attributes that are the same on every part -----------------------

    # Forwarded to ``parts[0]`` because they describe the geometry as a
    # whole, not one device's share of it: every part is built with the same
    # binning, the same mask and the same row space.  Anything *per patch*
    # is deliberately absent -- ``phi_center``, ``theta_center``,
    # ``n_patches``, ``patch_ranges`` -- because the first device's value is
    # only its own slice, and returning that silently is how a 917-patch run
    # comes back with 459 patches and no error.
    _SHARED_ATTRIBUTES = frozenset(
        {
            "nside",
            "npix",
            "nbins",
            "binedges",
            "bincenters",
            "theta_min",
            "theta_max",
            "theta_Q",
            "patch_size",
            "map_dtype",
            "acc_dtype",
            "rotation_dtype",
            "index_dtype",
            "map_precision",
            "rotation_precision",
            "accumulation_precision",
            "pair_search_precision",
            "resolution_factor",
            "aperture_nside",
            "level_table",
            "row_pix",
            "row_pix_hash",
            "n_active",
            "n_rows",
            "n_appended",
            "pack_pairs",
            "pack_host_pairs",
            "mask",
        }
    )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not self.__dict__.get("parts"):
            raise AttributeError(name)
        if name in _ARRAY_METHODS or name in _RESULT_METHODS:
            return self._dispatch_method(name)
        if name in self._SHARED_ATTRIBUTES:
            return getattr(self.parts[0], name)
        raise NotImplementedError(
            f"{name!r} is not defined for a multi-device group: each part holds "
            f"its own patch range, so the first device's value would be a "
            f"fraction of the answer. Reach for the part you mean "
            f"(group.parts[i].{name}), or use a single-device Integrated3PCF. "
            f"The attributes that are the same on every part are: "
            f"{', '.join(sorted(self._SHARED_ATTRIBUTES))}."
        )

    @property
    def nbins(self) -> int:
        return self.parts[0].nbins

    @property
    def ntotpairs(self) -> int:
        return int(sum(int(p.ntotpairs) for p in self.parts))

    # ---- fan-out ----------------------------------------------------------

    def _run(self, fn: Any) -> List[Any]:
        """Call ``fn(part, index)`` once per device, in parallel threads."""
        import threading

        out: List[Any] = [None] * len(self.parts)
        errors: List[BaseException] = []

        def worker(i: int) -> None:
            try:
                part = self.parts[i]
                module = getattr(part.backend, "module", None)
                cuda = getattr(module, "cuda", None)
                if cuda is not None and part.backend.name == "cupy":
                    with cuda.Device(part.backend.device_id):
                        out[i] = fn(part, i)
                        cuda.runtime.deviceSynchronize()
                else:
                    out[i] = fn(part, i)
            except BaseException as exc:  # surface the first failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(self.parts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return out

    def _dispatch_method(self, name: str) -> Any:
        from .results import IntegratedResult

        def call(*args: Any, **kwargs: Any) -> Any:
            # every device needs the maps on its own card, so device-resident
            # inputs are not forwarded: host arrays only
            for a in tuple(args) + tuple(kwargs.values()):
                if a is not None and not isinstance(a, (np.ndarray, int, float, bool, str)) \
                        and hasattr(a, "__cuda_array_interface__"):
                    raise TypeError(
                        f"{name} on several devices takes host arrays; a device "
                        "array belongs to one GPU only."
                    )
            kwargs.setdefault("return_device", False)
            results = self._run(lambda part, _i: getattr(part, name)(*args, **kwargs))
            if name in _ARRAY_METHODS:
                return self._concat(results, _ARRAY_METHODS[name])
            first = results[0]
            assembled = {
                k: self._concat([getattr(r, k) for r in results], axis)
                for k, axis in _FIELD_PATCH_AXIS.items()
                if getattr(first, k) is not None
            }
            return IntegratedResult(
                binedges=first.binedges,
                gg_combinations=first.gg_combinations,
                kk_combinations=first.kk_combinations,
                kg_combinations=first.kg_combinations,
                **assembled,
            )

        call.__name__ = name
        return call

    @staticmethod
    def _concat(parts: Sequence[Any], axis: int) -> np.ndarray:
        arrays = [np.asarray(p) for p in parts]
        return np.concatenate(arrays, axis=axis)

    # ---- geometry ---------------------------------------------------------

    def preprocess(self, *args: Any, **kwargs: Any) -> None:
        """Find the pairs of every patch, one thread per device."""
        self._run(lambda part, _i: part.preprocess(*args, **kwargs))

    def prepare(self, *args: Any, **kwargs: Any) -> None:
        self._run(lambda part, _i: part.prepare(*args, **kwargs))

    def warmup(self) -> None:
        self._run(lambda part, _i: part.warmup())

    def load_pairs(
        self, filepath: str, start_ind: int = 0, stop_ind: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Load one pair file, giving each device its own patch range."""
        stop = self.n_patches if stop_ind is None else int(stop_ind)
        offset = int(start_ind)
        if stop - offset != self.n_patches:
            raise ValueError(
                f"the file slice [{offset}, {stop}) holds {stop - offset} patches, "
                f"but this group was built for {self.n_patches}"
            )
        self._run(
            lambda part, i: part.load_pairs(
                filepath,
                start_ind=offset + self.patch_ranges[i][0],
                stop_ind=offset + self.patch_ranges[i][1],
                **kwargs,
            )
        )

    def save_pairs(self, filepath: str, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "a multi-device group holds one patch range per device; build the "
            "geometry once on a single device (or CPU) to write the file, then "
            "load it here -- load_pairs() slices it per device."
        )

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        ranges = ", ".join(f"{d}:{a}-{b}" for d, (a, b) in zip(self.devices, self.patch_ranges))
        return f"<MultiDeviceIntegrated3PCF {self.n_patches} patches over {ranges}>"
