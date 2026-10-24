"""Augmented Bayesian Policy Search (ICLR 2024): local Bayesian optimization of linear policies whose Gaussian
process prior mean comes from an ensemble of learned critics. Everything runs in JAX's native float32."""

import os
from pathlib import Path

import jax

if jax.config.jax_compilation_cache_dir is None:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "abps" / "jax"
    jax.config.update("jax_compilation_cache_dir", str(cache))
if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
