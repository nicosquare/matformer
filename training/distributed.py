"""Compatibility wrapper for the training.distributed module."""

from __future__ import annotations

import sys as _sys

from src.training import distributed as _impl

_sys.modules[__name__] = _impl
