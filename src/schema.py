"""Pydantic v2 models for the extraction output and the expense summary.

See Prompt.md - Phase 1. These types are the contract between ``extract`` /
``confidence`` / ``pipeline`` / ``summary`` and the JSON on disk.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import load_config

try:
    #: Fields scoring below this are listed in ``low_confidence_fields``.
    #: Sourced from ``config.yaml`` -> ``confidence.low_conf_threshold``.
    LOW_CONF_THRESHOLD: float = load_config().confidence.low_conf_threshold
except Exception:  # noqa: BLE001  # pragma: no cover - broken config, keep schema importable
    LOW_CONF_THRESHOLD = 0.7


class FieldConf(BaseModel):
    value: str | None = None
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, round(float(v), 3)))


class Item(BaseModel):
    name: FieldConf = Field(default_factory=FieldConf)
    price: FieldConf = Field(default_factory=FieldConf)
    meta: dict[str, Any] = Field(default_factory=dict)


class ReceiptExtraction(BaseModel):
    store_name: FieldConf = Field(default_factory=FieldConf)
    date: FieldConf = Field(default_factory=FieldConf)
    items: list[Item] = Field(default_factory=list)
    total_amount: FieldConf = Field(default_factory=FieldConf)
    low_confidence_fields: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_plain_dict(self) -> dict[str, Any]:
        """The flat schema exactly as shown in the assignment PDF (section 4)."""
        return {
            "store_name": self.store_name.value,
            "date": self.date.value,
            "items": [
                {"name": it.name.value, "price": it.price.value} for it in self.items
            ],
            "total_amount": self.total_amount.value,
        }

    def to_output_dict(self) -> dict[str, Any]:
        """Nested {value, confidence} schema (PDF section 6c) + plain form."""
        return {
            "store_name": self.store_name.model_dump(),
            "date": self.date.model_dump(),
            "items": [
                {"name": it.name.model_dump(), "price": it.price.model_dump(),
                 "qty": it.meta.get("qty")}
                for it in self.items
            ],
            "total_amount": self.total_amount.model_dump(),
            "low_confidence_fields": self.low_confidence_fields,
            "flags": self.flags,
            "meta": self.meta,
            "plain": self.to_plain_dict(),
        }


class StoreSpend(BaseModel):
    count: int = 0
    spend: float = 0.0
    mean_confidence: float = 0.0


class ExpenseSummary(BaseModel):
    currency: str = "RM"
    total_spend: float = 0.0
    num_transactions: int = 0
    num_transactions_with_total: int = 0
    spend_per_store: dict[str, StoreSpend] = Field(default_factory=dict)
    date_range: dict[str, str | None] = Field(default_factory=dict)
    excluded: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
