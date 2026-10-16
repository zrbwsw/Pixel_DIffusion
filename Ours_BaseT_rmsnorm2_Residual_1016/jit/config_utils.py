"""Strict grouped-YAML support for the class-to-image argparse entrypoints.

YAML mappings may be nested to organize parameters by module. Group names are
documentation-only; every leaf key must be the destination of exactly one
argparse option. YAML values become parser defaults, so an explicitly supplied
command-line option always wins. This keeps one experiment config portable
across machines whose paths and per-rank batch sizes differ.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml


def _flatten_grouped_config(
    node: Mapping[str, Any],
    *,
    prefix: str = "",
    values: dict[str, Any] | None = None,
    origins: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten nested documentation groups while retaining source paths."""

    if values is None:
        values = {}
    if origins is None:
        origins = {}

    for raw_key, value in node.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"YAML config keys must be non-empty strings, got {raw_key!r}")
        key_path = f"{prefix}.{raw_key}" if prefix else raw_key
        if isinstance(value, Mapping):
            _flatten_grouped_config(
                value,
                prefix=key_path,
                values=values,
                origins=origins,
            )
            continue

        if raw_key in values:
            raise ValueError(
                f"Duplicate argparse key {raw_key!r} in YAML config: "
                f"{origins[raw_key]!r} and {key_path!r}"
            )
        values[raw_key] = value
        origins[raw_key] = key_path

    return values, origins


def _coerce_config_value(
    action: argparse.Action,
    value: Any,
    *,
    source: str,
) -> Any:
    """Apply argparse's type and choice checks to a YAML-provided default."""

    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(
            "${PRETRAIN_MODELS_ROOT}",
            os.environ.get("PRETRAIN_MODELS_ROOT", "/data0/pretrain_models"),
        )
        value = os.path.expandvars(value)

    boolean_actions = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse.BooleanOptionalAction,
    )
    if isinstance(action, boolean_actions):
        if not isinstance(value, bool):
            raise ValueError(f"{source} must be a YAML boolean, got {value!r}")
        converted = value
    elif action.nargs not in (None, "?"):
        if not isinstance(value, list):
            raise ValueError(f"{source} must be a YAML list, got {value!r}")
        converted = [action.type(item) if action.type else item for item in value]
    elif action.type is not None:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for {source}: {value!r} ({exc})"
            ) from exc
    else:
        converted = value

    if action.choices is not None and converted not in action.choices:
        raise ValueError(
            f"Invalid value for {source}: {converted!r}; "
            f"expected one of {list(action.choices)!r}"
        )
    return converted


def load_grouped_yaml_defaults(
    parser: argparse.ArgumentParser,
    config_path: str | Path,
) -> dict[str, Any]:
    """Load, flatten, and validate one grouped experiment configuration."""

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise ValueError(f"YAML config root must be a mapping: {path}")

    values, origins = _flatten_grouped_config(document)
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    unknown = sorted(key for key in values if key not in actions)
    if unknown:
        locations = ", ".join(f"{key} ({origins[key]})" for key in unknown)
        raise ValueError(f"Unknown argparse key(s) in {path}: {locations}")

    return {
        key: _coerce_config_value(actions[key], value, source=origins[key])
        for key, value in values.items()
    }


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse grouped YAML defaults first, then explicit command-line overrides."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=str, default=None)
    config_args, _ = probe.parse_known_args(arguments)
    if config_args.config:
        parser.set_defaults(**load_grouped_yaml_defaults(parser, config_args.config))
    return parser.parse_args(arguments)
