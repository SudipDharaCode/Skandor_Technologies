"""Pydantic models: the LLM contract, the internal spec, and the API response."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProductType = Literal["ring", "pendant", "earrings", "bracelet", "necklace"]
Material = Literal["yellow_gold", "white_gold", "rose_gold", "silver", "platinum"]
StoneType = Literal["diamond", "sapphire", "ruby", "emerald", "cubic_zirconia", "none"]
SettingStyle = Literal["solitaire", "halo", "pave", "channel", "cluster", "plain"]
SizeUnit = Literal["mm", "cm", "inch", "us_ring_size"]
Source = Literal["stated", "inferred", "missing"]
Confidence = Literal["high", "medium", "low"]

SPEC_FIELDS = ("product_type", "material", "stone_type", "stone_count", "size_mm", "setting_style")


# LLM contract
class FieldSources(BaseModel):
    """Where the model got each value: written in the brief, guessed, or absent."""

    model_config = ConfigDict(extra="forbid")
    product_type: Source = "missing"
    material: Source = "missing"
    stone_type: Source = "missing"
    stone_count: Source = "missing"
    size: Source = "missing"
    setting_style: Source = "missing"


class LLMExtraction(BaseModel):
    """Exactly what we ask Gemini to return. Validated strictly before anything else sees it.
    Unknown keys are rejected so a renamed key ("metal") goes through repair instead of silently becoming null."""

    model_config = ConfigDict(extra="forbid")
    product_type: ProductType | None = None
    material: Material | None = None
    stone_type: StoneType | None = None
    stone_count: int | None = Field(default=None, ge=0)
    size_value: float | None = Field(default=None, gt=0)
    size_unit: SizeUnit | None = None
    setting_style: SettingStyle | None = None
    sources: FieldSources = Field(default_factory=FieldSources)
    image_stone_count: int | None = Field(default=None, ge=0)


# internal spec
class ProductSpec(BaseModel):
    """Normalized spec used by the matcher. size is always in millimeters."""

    product_type: ProductType | None = None
    material: Material | None = None
    stone_type: StoneType | None = None
    stone_count: int | None = None
    size_mm: float | None = None
    setting_style: SettingStyle | None = None


class InventoryItem(BaseModel):
    id: str
    name: str
    product_type: ProductType
    material: Material
    stone_type: StoneType
    stone_count: int
    size_mm: float
    setting_style: SettingStyle
    base_price_usd: float
    stone_unit_price_usd: float


# API response
class FieldResult(BaseModel):
    value: str | int | float | None
    confidence: Confidence
    reason: str


class ReviewFlag(BaseModel):
    code: str
    field: str | None = None
    message: str


class MatchCandidate(BaseModel):
    id: str
    name: str
    score: float
    breakdown: dict[str, float]


class MatchResult(BaseModel):
    selected_id: str | None
    auto_selected: bool
    coverage: float
    candidates: list[MatchCandidate]
    confidence: Confidence
    reason: str


class ImageEstimate(BaseModel):
    cv_count: int | None
    cv_confidence: Confidence | None
    cv_details: dict | None
    vlm_count: int | None
    final_count: int | None
    confidence: Confidence
    reason: str


class ExtractionInfo(BaseModel):
    mode: Literal["clean", "repaired", "retried", "degraded"]
    steps: list[str]
    model_used: str | None
    repaired_fields: list[str]


class PriceEstimate(BaseModel):
    value_usd: float | None
    confidence: Confidence
    reason: str


class AnalyzeResponse(BaseModel):
    request_id: str
    spec: dict[str, FieldResult]
    extraction: ExtractionInfo
    image_estimate: ImageEstimate | None
    match: MatchResult
    price: PriceEstimate
    review_flags: list[ReviewFlag]
    needs_review: bool
    meta: dict
