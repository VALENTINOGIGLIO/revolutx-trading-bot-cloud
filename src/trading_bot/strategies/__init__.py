"""Trading Bot Strategies Package."""
from .amv_trend_break import (
    AMVTrendBreakStrategy,
    PROFILE_4H_PLATEAU_WINNER,
    PROFILE_2H_PLATEAU_WINNER,
    PROFILE_V2_BASELINE,
)

__all__ = [
    "AMVTrendBreakStrategy",
    "PROFILE_4H_PLATEAU_WINNER",
    "PROFILE_2H_PLATEAU_WINNER",
    "PROFILE_V2_BASELINE",
]
