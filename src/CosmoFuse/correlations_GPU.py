import logging
import warnings
from .correlations import Correlation

logger = logging.getLogger(__name__)

# Check for cupy just to warn if missing, though Correlation handles it.
try:
    import cupy as cp
except ImportError:
    pass

class Correlation_GPU(Correlation):
    def __init__(
        self,
        nside,
        phi_center,
        theta_center,
        nbins=10,
        theta_min=10,
        theta_max=170,
        patch_size=90,
        theta_Q=90,
        mask=None,
        fastmath=True,
        device=0,
    ):
        logger.warning("Correlation_GPU is deprecated. Use Correlation(device='gpu') or Correlation(device=ID) instead.")
        super().__init__(
            nside,
            phi_center,
            theta_center,
            nbins,
            theta_min,
            theta_max,
            patch_size,
            theta_Q,
            mask,
            fastmath,
            device=device
        )

    def prepare_gpu(self):
        logger.warning("prepare_gpu is deprecated. Use prepare() instead. calling prepare() now.")
        self.prepare()

    # Aliases for backward compatibility if attributes were accessed directly
    @property
    def inds_gpu(self):
        return self.inds_dev

    @property
    def exp2phi_gpu(self):
        return self.exp2phi_dev

    @property
    def bins_gpu(self):
        return self.bins_dev

    @property
    def tot_bins_gpu(self):
        return self.tot_bins_dev


class Correlation_GPU_lowmem(Correlation_GPU):
    def __init__(self, *args, **kwargs):
        logger.warning("Correlation_GPU_lowmem is deprecated. Using standard Correlation class (may use more memory).")
        super().__init__(*args, **kwargs)

    # If strictly needed, we could override xipm here to use a loop.
    # For now, we assume the unified class is sufficient or users can contribute a lowmem mode to Correlation.
