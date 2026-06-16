"""Pluggable compute backends for vaeda's VAE and PU classifier.

Each backend takes numpy arrays in and returns numpy arrays out, hiding the
framework-specific training loop (torch's hand-rolled loop, TensorFlow's
``model.fit``) behind a single interface. Phase 2 of docs/Roadmap.md adds the
``Backend`` protocol and ``get_backend()`` resolver here; for now the torch
backend lives under ``backends._torch`` and is imported directly.
"""
