"""Validated application data. Missing source fields stay None (JSON null)."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Price(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")


class Rating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(ge=0, le=100, allow_inf_nan=False)
    best: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    worst: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_scale(self):
        if self.best is not None and self.value > self.best:
            raise ValueError("Rating exceeds its scale")
        if self.worst is not None and self.value < self.worst:
            raise ValueError("Rating is below its scale")
        if self.best is not None and self.worst is not None and self.best < self.worst:
            raise ValueError("Invalid rating scale")
        return self


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_url: HttpUrl
    name: str | None = Field(default=None, min_length=1, max_length=500)
    price: Price | None = None
    description: str | None = Field(default=None, min_length=1, max_length=6000)
    rating: Rating | None = None
    extraction_method: str = Field(max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=30)


class Rewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=4000)
