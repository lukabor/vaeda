"""Pluggable compute backends for vaeda's VAE and PU classifier.

Each backend takes numpy arrays in and returns numpy arrays out, hiding the
framework-specific training loop (torch's hand-rolled loop, TensorFlow's
``model.fit``) behind the :class:`~vaeda.backends.base.Backend` protocol.

:func:`get_backend` resolves which backend to use: an explicit
``VAEDA_BACKEND`` environment variable wins, otherwise the single installed
backend is auto-detected, with torch the default (and a warning) when both are
installed. Framework imports are deferred until a backend is actually loaded,
so importing vaeda never forces torch *or* tensorflow into the process.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .base import Backend

# Supported backend name -> pip extra that installs it.
_SUPPORTED: dict[str, str] = {"torch": "torch", "tensorflow": "tensorflow"}
# torch is the default when the choice is ambiguous.
_DEFAULT = "torch"


def _available_backends() -> list[str]:
    """Return the installed backend names, without importing the frameworks."""
    return [
        name
        for name in _SUPPORTED
        if importlib.util.find_spec(name) is not None
    ]


def _resolve_backend_name(requested: str | None, available: list[str]) -> str:
    """Decide which backend to use from an explicit request and what is installed.

    Pure function (no environment or imports) so the policy is testable in
    isolation. Raises ``ValueError`` for an unknown name and ``ImportError``
    when the chosen/only-sensible backend is not installed.
    """
    if requested is not None:
        requested = requested.lower()
        if requested not in _SUPPORTED:
            choices = ", ".join(_SUPPORTED)
            msg = f"Unknown VAEDA_BACKEND={requested!r}; choose from: {choices}"
            raise ValueError(msg)
        if requested not in available:
            extra = _SUPPORTED[requested]
            msg = (
                f"VAEDA_BACKEND={requested!r} was requested but is not installed. "
                f"Install it with: pip install vaeda[{extra}]"
            )
            raise ImportError(msg)
        return requested

    if not available:
        msg = (
            "No vaeda compute backend is installed. Install one with: "
            f"pip install vaeda[{_SUPPORTED[_DEFAULT]}] (default) "
            "or pip install vaeda[tensorflow]"
        )
        raise ImportError(msg)

    if _DEFAULT in available:
        if len(available) > 1:
            installed = ", ".join(available)
            logger.warning(
                f"Multiple vaeda backends installed ({installed}); "
                f"defaulting to {_DEFAULT!r}. Set VAEDA_BACKEND to choose explicitly."
            )
        return _DEFAULT
    return available[0]


@functools.cache
def get_backend() -> Backend:
    """Return the resolved compute backend (cached for the process)."""
    name = _resolve_backend_name(os.environ.get("VAEDA_BACKEND"), _available_backends())
    if name == "torch":
        from ._torch import BACKEND

        return BACKEND
    from ._tf import BACKEND

    return BACKEND
