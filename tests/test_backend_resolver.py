"""Tests for the backend resolver (Phase 2 of docs/Roadmap.md).

The resolution *decision* is a pure function of the requested name and the set
of installed backends, so it is tested directly without touching the
environment or importing any framework.
"""

import pytest


class TestResolveBackendName:
    """backends._resolve_backend_name: pick a backend from request + availability."""

    def test_explicit_request_is_honoured(self):
        """
        Given VAEDA_BACKEND names an installed backend
        When the backend name is resolved
        Then that backend is chosen
        """
        from vaeda.backends import _resolve_backend_name

        assert _resolve_backend_name("torch", ["torch"]) == "torch"

    def test_explicit_request_is_case_insensitive(self):
        """
        Given VAEDA_BACKEND is set with odd casing
        When the backend name is resolved
        Then it matches the canonical lower-case backend name
        """
        from vaeda.backends import _resolve_backend_name

        assert _resolve_backend_name("TensorFlow", ["tensorflow"]) == "tensorflow"

    def test_autodetect_single_backend(self):
        """
        Given no explicit request and exactly one installed backend
        When the backend name is resolved
        Then the installed backend is chosen
        """
        from vaeda.backends import _resolve_backend_name

        assert _resolve_backend_name(None, ["tensorflow"]) == "tensorflow"

    def test_both_installed_defaults_to_torch_with_warning(self):
        """
        Given no explicit request and both backends installed
        When the backend name is resolved
        Then torch is chosen and a warning is emitted about the default
        """
        from loguru import logger

        from vaeda.backends import _resolve_backend_name

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")
        try:
            name = _resolve_backend_name(None, ["torch", "tensorflow"])
        finally:
            logger.remove(sink_id)

        assert name == "torch"
        assert any("VAEDA_BACKEND" in m for m in messages)

    def test_requested_but_not_installed_raises_with_hint(self):
        """
        Given VAEDA_BACKEND requests a backend that is not installed
        When the backend name is resolved
        Then an ImportError names the matching pip extra to install
        """
        from vaeda.backends import _resolve_backend_name

        with pytest.raises(ImportError, match=r"vaeda\[tensorflow\]"):
            _resolve_backend_name("tensorflow", ["torch"])

    def test_unknown_backend_name_raises(self):
        """
        Given VAEDA_BACKEND names a backend vaeda does not support
        When the backend name is resolved
        Then a ValueError is raised
        """
        from vaeda.backends import _resolve_backend_name

        with pytest.raises(ValueError, match="bogus"):
            _resolve_backend_name("bogus", ["torch"])

    def test_no_backend_installed_raises_with_hint(self):
        """
        Given no explicit request and no backend installed
        When the backend name is resolved
        Then an ImportError points the user at the install extras
        """
        from vaeda.backends import _resolve_backend_name

        with pytest.raises(ImportError, match=r"vaeda\[torch\]"):
            _resolve_backend_name(None, [])


class TestGetBackend:
    """backends.get_backend: load the resolved backend object."""

    def test_returns_torch_backend_exposing_the_seam(self):
        """
        Given torch is installed (the test environment default)
        When get_backend is called
        Then it returns the torch backend exposing the numpy-in/out seam
        """
        from vaeda.backends import get_backend

        backend = get_backend()

        assert backend.name == "torch"
        assert callable(backend.train_clust_vae)
        assert callable(backend.train_pu_fold)
