"""Compatibility wrapper for the training orchestration module."""

from __future__ import annotations

import sys as _sys

from src.training import run as _impl

_sys.modules[__name__] = _impl
