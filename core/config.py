"""
Core Configuration for Quant Multi-Agent System
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QuantConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000

    # Trading parameters
    default_risk_free_rate: float = 0.04
    max_portfolio_volatility: float = 0.20
    min_trade_size: float = 100.0
    max_leverage: float = 2.0

    # Data sources
    yfinance_enabled: bool = True
    polygon_enabled: bool = False


class AssetClass(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SCALPING = "scalping"
    SWING = "swing"
    POSITIONAL = "positional"
    ARBITRAGE = "arbitrage"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeFrame(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


config = QuantConfig()
