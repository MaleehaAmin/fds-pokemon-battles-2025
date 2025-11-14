# src/fds_pokemon/__init__.py
"""
FDS Pokemon Battles 2025 – reusable package.

Modules:
- data: loading train/test JSONL
- features: feature engineering from raw battles
- utils: generic ML helpers
- models: training recipes (advstack, calibrated ensembles, etc.)
"""
__all__ = ["data", "features", "utils", "models"]