import logging
import warnings
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

class Backend:
    def __init__(self, name: str, module: Any, device_id: Optional[int] = None) -> None:
        self.name = name
        self.module = module
        self.device_id = device_id

        self.asarray = module.asarray
        self.zeros = module.zeros
        self.ones = module.ones
        self.sum = module.sum
        self.mean = module.mean
        self.conjugate = module.conjugate
        self.add = module.add
        self.float32 = module.float32
        self.float64 = module.float64
        self.complex64 = module.complex64
        self.complex128 = module.complex128
        self.uint32 = module.uint32
        self.int32 = module.int32

    def to_device(self, array: Any) -> Any:
        """Move a numpy array to the backend device."""
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
            with self.module.cuda.Device(self.device_id):
                return self.module.asarray(array)
        return array

    def to_numpy(self, array: Any) -> np.ndarray:
        """Move an array from the backend device to numpy."""
        if self.name == 'numpy':
            return np.asarray(array)
        elif self.name == 'cupy':
            return self.module.asnumpy(array)
        return np.asarray(array)

    def get_memory_pool(self) -> Optional[Any]:
        if self.name == 'cupy':
            return self.module.get_default_memory_pool()
        return None

def get_backend(device: Union[str, int] = 'auto') -> "Backend":
    """
    Get the appropriate backend (numpy or cupy).

    Args:
        device (str or int): 'cpu', 'gpu', 'auto', or a GPU ID (int).
                             If 'gpu' is specified, it uses the first available GPU (ID 0).
                             If an int is provided, it uses that specific GPU ID.

    Returns:
        Backend: An object wrapping the numpy/cupy module with helper methods.
    """
    if isinstance(device, int):
        device_id = device
        device_type = 'gpu'
    elif device.lower() == 'gpu':
        device_id = 0
        device_type = 'gpu'
    elif device.lower() == 'cpu':
        device_id = None
        device_type = 'cpu'
    elif device.lower() == 'auto':
        try:
            import cupy
            device_id = 0
            device_type = 'gpu'
        except ImportError:
            warnings.warn("Cupy not installed, falling back to CPU (numpy).")
            device_id = None
            device_type = 'cpu'
    else:
        raise ValueError(f"Unknown device: {device}")

    if device_type == 'cpu':
        return Backend('numpy', np)

    elif device_type == 'gpu':
        try:
            import cupy
            # Check if the requested device is available
            if device_id >= cupy.cuda.runtime.getDeviceCount():
                raise ValueError(f"GPU ID {device_id} not found.")

            return Backend('cupy', cupy, device_id)
        except ImportError:
            if device == 'auto':
                warnings.warn("Cupy not installed, falling back to CPU (numpy).")
                return Backend('numpy', np)
            else:
                raise ImportError("Cupy not installed but GPU requested.")
