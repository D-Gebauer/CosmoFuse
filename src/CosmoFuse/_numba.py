"""Numba entry points with types mypy can check.

``numba.prange`` is typed as a class without ``__iter__``; to a type checker
it is ``range`` (which is also what it is with ``NUMBA_DISABLE_JIT=1``).  At
run time the jitted functions see the real ``numba.prange``.
"""

from typing import TYPE_CHECKING, Any, Callable

import numba
from numba import njit

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

#: ``numba.get_num_threads``, typed (numba < 0.68 leaves it untyped; going
#: through ``Any`` is correct for both).
_numba: Any = numba
get_num_threads: Callable[[], int] = _numba.get_num_threads

__all__ = ["get_num_threads", "njit", "prange"]
