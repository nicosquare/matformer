"""Compatibility wrapper for the training.data module."""

from __future__ import annotations

import sys as _sys

from src.training import data as _impl

_sys.modules[__name__] = _impl
