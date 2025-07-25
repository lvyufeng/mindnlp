from datetime import timedelta
from typing import Optional

# from core._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
_DEFAULT_PG_TIMEOUT = timedelta(seconds=5000)

__all__ = ["default_pg_timeout", "default_pg_nccl_timeout"]

# Default process group wide timeout, if applicable.
# This only applies to the non-nccl backends
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT
# Separate timeout for PGNCCL mainly becuase it's always been that way in the C++ layer, but until recently
# there was one default that applied across all backends in the python layer.
# Later, we could consider merging them back together at the c++ layer if we can align on a same value.
# (only if TORCH_NCCL_BLOCKING_WAIT or TORCH_NCCL_ASYNC_ERROR_HANDLING is set to 1).

default_pg_nccl_timeout: Optional[timedelta] = _DEFAULT_PG_TIMEOUT
