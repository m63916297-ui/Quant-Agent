"""
Data Models for Quant System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


@dataclass
class Portfolio:
    id: str
    name: str
    positions: List["Position"] = field(default_factory=list)
    cash: float = 100000.0
    total_value: float = 100000.0
    created_at: datetime = field(default_factory=datetime.now)

    def get_weights(self) -> Dict[str, float]:
        if self.total_value == 0:
            return {}
        return {p.symbol: p.market_value / self.total_value for p in self.positions}

    def get_return(self, benchmark: float = 0.0) -> float:
        if not self.positions:
            return 0.0
        total_return = (
            self.total_value - self.cash - sum(p.cost_basis for p in self.positions)
        ) / self.cash
        return total_return - benchmark


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_return(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class Signal:
    symbol: str
    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    strategy: str
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1d"

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    @property
    def cagr(self) -> float:
        years = (self.end_date - self.start_date).days / 365.25
        if years == 0:
            return 0.0
        return (self.final_capital / self.initial_capital) ** (1 / years) - 1


@dataclass
class RiskMetrics:
    portfolio_value: float
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: float = 0.0
    alpha: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    constraints_satisfied: bool
    optimization_method: str
