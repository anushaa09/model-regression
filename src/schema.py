from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class RegressionResult(BaseModel):
    module: str

    baseline_metric: float
    candidate_metric: float

    absolute_change: float
    relative_change: float

    p_value: Optional[float] = None

    significance_level: float = 0.05
    threshold: float = 0.02

    statistically_significant: bool
    practically_significant: bool

    status: str

    details: Dict[str, Any] = Field(default_factory=dict)