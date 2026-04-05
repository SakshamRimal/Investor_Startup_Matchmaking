from pydantic import BaseModel
from typing import List


class CompatibilityRequest(BaseModel):
    investor_id: str
    startup_id: str


class TractionRequest(BaseModel):
    startup_id: str


class SectorRequest(BaseModel):
    sector_a: str
    sector_b: str


class NextSectorRequest(BaseModel):
    history: List[str]
    top_k: int = 3


class SuggestRequest(BaseModel):
    investor_id: str
    top_k: int = 5
    novelty_weight: float = 0.3
