"""Loads the price table (a data file) and prices usage records."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Optional

from .paths import user_prices_path

STALE_AFTER_DAYS = 60


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    max_output: Optional[int] = None


@dataclass
class PriceTable:
    verified_on: Optional[date]
    source: str
    path: str
    models: dict[str, ModelPrice]
    write_5m: float = 1.25
    write_1h: float = 2.0

    def lookup(self, model: str) -> Optional[ModelPrice]:
        best = None
        for key in self.models:
            if model == key or model.startswith(key + "-") or model.startswith(key + "@"):
                if best is None or len(key) > len(best):
                    best = key
        return self.models[best] if best else None

    def cost(
        self,
        model: str,
        *,
        new: int,
        cache_read: int,
        cache_write: int,
        cache_write_1h: int = 0,
        output: int,
    ) -> Optional[float]:
        p = self.lookup(model)
        if p is None:
            return None
        w5 = cache_write - cache_write_1h
        return (
            new * p.input
            + cache_read * p.cache_read
            + w5 * p.input * self.write_5m
            + cache_write_1h * p.input * self.write_1h
            + output * p.output
        ) / 1_000_000

    def age_days(self, today: Optional[date] = None) -> Optional[int]:
        if self.verified_on is None:
            return None
        return ((today or date.today()) - self.verified_on).days

    def is_stale(self, today: Optional[date] = None) -> bool:
        age = self.age_days(today)
        return age is None or age > STALE_AFTER_DAYS


def _validate(data: dict) -> None:
    for name, m in data.get("models", {}).items():
        for key in ("input", "output", "cache_read"):
            float(m[key])  # KeyError / ValueError name the problem


def packaged_prices_text() -> str:
    return resources.files("tokentrail").joinpath("data/prices.toml").read_text(encoding="utf-8")


def load(path: Optional[Path] = None) -> PriceTable:
    """The user's copy if present, else the packaged file."""
    if path is None:
        up = user_prices_path()
        path = up if up.is_file() else None
    if path is not None:
        text, origin = path.read_text(encoding="utf-8"), str(path)
    else:
        text, origin = packaged_prices_text(), "(packaged default)"
    try:
        data = tomllib.loads(text)
        _validate(data)
        verified = data.get("verified_on")
        if isinstance(verified, str):
            verified = date.fromisoformat(verified)
    except (tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as e:
        from .errors import TokentrailError

        raise TokentrailError(
            f"The price file {origin} can't be read ({e}).",
            "Fix it, or delete it to go back to the packaged prices "
            "(`tokentrail prices --init` makes a fresh copy).",
        ) from e
    mult = data.get("cache_write_multiplier", {})
    models = {
        name: ModelPrice(
            input=float(m["input"]),
            output=float(m["output"]),
            cache_read=float(m["cache_read"]),
            max_output=int(m["max_output"]) if "max_output" in m else None,
        )
        for name, m in data.get("models", {}).items()
    }
    return PriceTable(
        verified_on=verified if isinstance(verified, date) else None,
        source=str(data.get("source", "")),
        path=origin,
        models=models,
        write_5m=float(mult.get("5m", 1.25)),
        write_1h=float(mult.get("1h", 2.0)),
    )
