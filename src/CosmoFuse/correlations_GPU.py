import logging

import numpy as np

from .correlations import Correlation

logger = logging.getLogger(__name__)

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
        multiprocessing_start_method="spawn",
        map_precision: str | np.dtype | type = "float64",
        rotation_precision: str | np.dtype | type = "float64",
        index_precision: str | np.dtype | type = "uint32",
    ):
        logger.warning("Correlation_GPU is deprecated. Use Correlation(device='gpu') or Correlation(device=ID) instead.")
        super().__init__(
            nside=nside,
            phi_center=phi_center,
            theta_center=theta_center,
            nbins=nbins,
            theta_min=theta_min,
            theta_max=theta_max,
            patch_size=patch_size,
            theta_Q=theta_Q,
            mask=mask,
            fastmath=fastmath,
            device=device,
            multiprocessing_start_method=multiprocessing_start_method,
            map_precision=map_precision,
            rotation_precision=rotation_precision,
            index_precision=index_precision,
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
