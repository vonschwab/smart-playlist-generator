"""
Configuration Resolution for Playlist Pipeline
==============================================

Extracted from pipeline.py (Phase 5.1).

This module provides structured configuration resolution with explicit
precedence rules for playlist generation parameters.

Resolution Order (highest to lowest priority):
1. Runtime overrides (passed as function parameters)
2. Mode-specific config (e.g., dynamic.weight)
3. Direct config (e.g., weight)
4. Default values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """Single source of configuration values with priority."""

    name: str
    """Human-readable name of this config source."""

    get_value: Callable[[str], Any | None]
    """Function to retrieve value for a key (returns None if not found)."""

    priority: int
    """Priority level (lower = higher priority)."""


class ConfigResolver:
    """Resolves configuration values with explicit precedence rules.

    Precedence (highest to lowest):
    1. Runtime overrides (priority=0)
    2. Mode-specific config (priority=1)
    3. Direct config (priority=2)
    4. Defaults (priority=3)

    Usage:
        resolver = ConfigResolver()
        resolver.add_source("runtime", lambda k: overrides.get(k), priority=0)
        resolver.add_source("defaults", lambda k: defaults.get(k), priority=3)

        value = resolver.resolve("sonic_weight", default=0.7)
    """

    def __init__(self):
        """Initialize empty configuration resolver."""
        self.sources: List[ConfigSource] = []
        self.logger = logging.getLogger(__name__)

    def add_source(
        self,
        name: str,
        get_value: Callable[[str], Any | None],
        priority: int,
    ) -> ConfigResolver:
        """Add a configuration source.

        Args:
            name: Human-readable name
            get_value: Function to retrieve value for a key
            priority: Priority level (lower = higher priority)

        Returns:
            Self for chaining
        """
        source = ConfigSource(name=name, get_value=get_value, priority=priority)
        self.sources.append(source)
        # Keep sources sorted by priority
        self.sources.sort(key=lambda s: s.priority)
        return self

    def resolve(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        log_source: bool = False,
    ) -> Any:
        """Resolve configuration value through source chain.

        Args:
            key: Configuration key to resolve
            default: Default value if not found
            required: If True, raise ValueError if not found and no default
            log_source: If True, log which source provided the value

        Returns:
            Resolved configuration value

        Raises:
            ValueError: If required=True and value not found
        """
        for source in self.sources:
            value = source.get_value(key)
            if value is not None:
                if log_source:
                    self.logger.debug(
                        f"Config '{key}': {value} (from {source.name})"
                    )
                return value

        # Value not found in any source
        if required and default is None:
            raise ValueError(
                f"Required config key '{key}' not found in any source"
            )

        if log_source and default is not None:
            self.logger.debug(f"Config '{key}': {default} (from default)")

        return default

    def resolve_many(
        self,
        keys: List[str],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve multiple configuration keys.

        Args:
            keys: List of configuration keys
            defaults: Optional dict of default values

        Returns:
            Dictionary of resolved values
        """
        defaults = defaults or {}
        result = {}

        for key in keys:
            default = defaults.get(key)
            result[key] = self.resolve(key, default=default)

        return result

    def clear(self) -> None:
        """Remove all configuration sources."""
        self.sources.clear()


def build_pipeline_config_resolver(
    *,
    mode: str,
    overrides: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> ConfigResolver:
    """Build a configuration resolver for DS pipeline.

    Args:
        mode: Playlist mode (dynamic, narrow, etc.)
        overrides: Runtime configuration overrides
        base_config: Base configuration dictionary

    Returns:
        Configured ConfigResolver instance

    Example:
        resolver = build_pipeline_config_resolver(
            mode="dynamic",
            overrides={"sonic_weight": 0.8},
            base_config=config_dict,
        )

        sonic_weight = resolver.resolve("sonic_weight", default=0.7)
    """
    resolver = ConfigResolver()

    # Priority 0: Runtime overrides (highest priority)
    if overrides:
        resolver.add_source(
            name="runtime_overrides",
            get_value=lambda k: overrides.get(k),
            priority=0,
        )

    # Priority 1: Mode-specific config
    if base_config:
        mode_config = base_config.get(mode, {})
        if isinstance(mode_config, dict):
            resolver.add_source(
                name=f"mode_{mode}",
                get_value=lambda k: mode_config.get(k),
                priority=1,
            )

    # Priority 2: Direct config
    if base_config:
        resolver.add_source(
            name="base_config",
            get_value=lambda k: base_config.get(k),
            priority=2,
        )

    return resolver


def resolve_hybrid_weights(
    *,
    sonic_weight: Optional[float] = None,
    genre_weight: Optional[float] = None,
    mode: str = "dynamic",
    overrides: Optional[Dict[str, Any]] = None,
) -> tuple[float, float]:
    """Resolve sonic and genre weights with fallback logic.

    Args:
        sonic_weight: Explicit sonic weight (highest priority)
        genre_weight: Explicit genre weight (highest priority)
        mode: Playlist mode for defaults
        overrides: Configuration overrides

    Returns:
        Tuple of (sonic_weight, genre_weight)

    Example:
        sonic, genre = resolve_hybrid_weights(
            sonic_weight=0.8,
            mode="dynamic",
        )
        # Returns: (0.8, 0.2)  # genre = 1.0 - sonic
    """
    # Default weights by mode
    defaults = {
        "dynamic": (0.7, 0.3),
        "narrow": (0.8, 0.2),
        "sonic_only": (1.0, 0.0),
    }

    default_sonic, default_genre = defaults.get(mode, (0.7, 0.3))

    # Resolve sonic weight
    resolved_sonic = sonic_weight
    if resolved_sonic is None and overrides:
        resolved_sonic = overrides.get("sonic_weight")
    if resolved_sonic is None:
        resolved_sonic = default_sonic

    # Resolve genre weight
    resolved_genre = genre_weight
    if resolved_genre is None and overrides:
        resolved_genre = overrides.get("genre_weight")
    if resolved_genre is None:
        # Auto-complement if only sonic provided
        if sonic_weight is not None or (overrides and "sonic_weight" in overrides):
            resolved_genre = 1.0 - resolved_sonic
        else:
            resolved_genre = default_genre

    # Normalize to sum to 1.0
    total = resolved_sonic + resolved_genre
    if total > 0 and abs(total - 1.0) > 0.01:
        logger.warning(
            f"Hybrid weights don't sum to 1.0 (sonic={resolved_sonic}, "
            f"genre={resolved_genre}, total={total}); normalizing"
        )
        resolved_sonic = resolved_sonic / total
        resolved_genre = resolved_genre / total

    return resolved_sonic, resolved_genre
