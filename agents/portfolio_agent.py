"""
Portfolio Management Agent using LangChain
"""

from langchain.agents import Agent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.tools import Tool
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from models.data_models import Portfolio, Position, RiskMetrics
from core.config import config


class PortfolioAgent:
    def __init__(self, llm=None):
        self.name = "Portfolio Manager Agent"
        self.description = "Manages investment portfolios, optimizes allocation, and rebalances positions"
        self.llm = llm
        self.portfolios: Dict[str, Portfolio] = {}

        self.system_prompt = """You are an expert Portfolio Manager with 20 years of experience in quantitative finance.
        
        Your responsibilities include:
        1. Portfolio construction and optimization using Modern Portfolio Theory
        2. Asset allocation across multiple asset classes
        3. Rebalancing decisions based on drift thresholds
        4. Risk-adjusted return optimization
        5. Factor-based portfolio construction
        
        Always consider:
        - Diversification to reduce unsystematic risk
        - Correlation between assets
        - Transaction costs and tax implications
        - Client risk tolerance and investment horizon
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content="{input}"),
            ]
        )

    def optimize_portfolio(
        self, returns_df: pd.DataFrame, risk_free_rate: float = 0.04
    ) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization"""
        if returns_df.empty or len(returns_df.columns) < 2:
            return {"error": "Insufficient data for optimization"}

        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        num_assets = len(returns_df.columns)
        num_portfolios = 10000

        results = {
            "weights": [],
            "returns": [],
            "volatilities": [],
            "sharpe_ratios": [],
        }

        for _ in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix * 252, weights))
            )
            sharpe = (
                (portfolio_return - risk_free_rate) / portfolio_std
                if portfolio_std > 0
                else 0
            )

            results["weights"].append(weights)
            results["returns"].append(portfolio_return)
            results["volatilities"].append(portfolio_std)
            results["sharpe_ratios"].append(sharpe)

        max_sharpe_idx = np.argmax(results["sharpe_ratios"])
        optimal_weights = {
            returns_df.columns[i]: round(results["weights"][max_sharpe_idx][i], 4)
            for i in range(num_assets)
        }

        return {
            "optimal_weights": optimal_weights,
            "expected_return": round(results["returns"][max_sharpe_idx], 4),
            "expected_volatility": round(results["volatilities"][max_sharpe_idx], 4),
            "sharpe_ratio": round(results["sharpe_ratios"][max_sharpe_idx], 4),
            "efficient_frontier": {
                "returns": results["returns"],
                "volatilities": results["volatilities"],
                "sharpe_ratios": results["sharpe_ratios"],
            },
        }

    def calculate_risk_metrics(
        self, portfolio: Portfolio, returns: pd.Series
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        portfolio_values = [
            portfolio.cash + sum(p.market_value for p in portfolio.positions)
        ]

        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5) * portfolio.total_value
            cvar_95 = (
                returns[returns <= np.percentile(returns, 5)].mean()
                * portfolio.total_value
            )

            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            excess_returns = returns - risk_free_rate / 252
            sharpe = (
                np.sqrt(252) * excess_returns.mean() / returns.std()
                if returns.std() > 0
                else 0
            )

            downside_returns = returns[returns < 0]
            sortino = (
                np.sqrt(252) * excess_returns.mean() / downside_returns.std()
                if len(downside_returns) > 0 and downside_returns.std() > 0
                else 0
            )
        else:
            volatility = 0
            var_95 = 0
            cvar_95 = 0
            max_drawdown = 0
            sharpe = 0
            sortino = 0

        return RiskMetrics(
            portfolio_value=portfolio.total_value,
            volatility=round(volatility, 4),
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
        )

    def generate_rebalancing_signal(
        self, portfolio: Portfolio, drift_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Generate rebalancing signals based on portfolio drift"""
        current_weights = portfolio.get_weights()
        target_weights = (
            {k: 1.0 / len(current_weights) for k in current_weights.keys()}
            if current_weights
            else {}
        )

        if not target_weights:
            return {"action": "hold", "reason": "No positions to rebalance"}

        drifts = {
            symbol: abs(current_weights.get(symbol, 0) - target_weights.get(symbol, 0))
            for symbol in set(
                list(current_weights.keys()) + list(target_weights.keys())
            )
        }

        max_drift = max(drifts.values()) if drifts else 0
        drift_symbols = [s for s, d in drifts.items() if d > drift_threshold]

        if max_drift > drift_threshold:
            trades = []
            for symbol in drift_symbols:
                current_w = current_weights.get(symbol, 0)
                target_w = target_weights.get(symbol, 0)
                diff = target_w - current_w
                trade_value = abs(diff) * portfolio.total_value

                trades.append(
                    {
                        "symbol": symbol,
                        "action": "buy" if diff > 0 else "sell",
                        "value": round(trade_value, 2),
                        "current_weight": round(current_w, 4),
                        "target_weight": round(target_w, 4),
                        "drift": round(drifts[symbol], 4),
                    }
                )

            return {
                "action": "rebalance",
                "reason": f"Max drift {max_drift:.2%} exceeds threshold {drift_threshold:.2%}",
                "trades": trades,
                "estimated_cost": round(
                    sum(abs(t["value"]) for t in trades) * 0.001, 2
                ),
            }

        return {
            "action": "hold",
            "reason": f"Max drift {max_drift:.2%} within threshold",
        }

    def create_portfolio(
        self, portfolio_id: str, name: str, initial_capital: float = 100000
    ) -> Portfolio:
        """Create a new portfolio"""
        portfolio = Portfolio(
            id=portfolio_id,
            name=name,
            cash=initial_capital,
            total_value=initial_capital,
        )
        self.portfolios[portfolio_id] = portfolio
        return portfolio

    def add_position(
        self, portfolio_id: str, symbol: str, quantity: float, price: float
    ) -> Dict:
        """Add a position to portfolio"""
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}

        portfolio = self.portfolios[portfolio_id]
        cost = quantity * price

        if cost > portfolio.cash:
            return {"error": "Insufficient cash"}

        existing = next((p for p in portfolio.positions if p.symbol == symbol), None)
        if existing:
            total_cost = existing.cost_basis + cost
            total_qty = existing.quantity + quantity
            existing.entry_price = total_cost / total_qty
            existing.quantity = total_qty
        else:
            portfolio.positions.append(
                Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                )
            )

        portfolio.cash -= cost
        portfolio.total_value = portfolio.cash + sum(
            p.market_value for p in portfolio.positions
        )

        return {"success": True, "portfolio": portfolio}

    def execute(self, action: str, context: Dict) -> Dict:
        """Execute portfolio management action"""
        if action == "optimize":
            return self.optimize_portfolio(
                context.get("returns_df"), context.get("risk_free_rate", 0.04)
            )
        elif action == "rebalance":
            portfolio = self.portfolios.get(context.get("portfolio_id"))
            if portfolio:
                return self.generate_rebalancing_signal(
                    portfolio, context.get("drift_threshold", 0.05)
                )
        elif action == "create":
            return self.create_portfolio(
                context.get("id"), context.get("name"), context.get("capital", 100000)
            )
        elif action == "add_position":
            return self.add_position(
                context.get("portfolio_id"),
                context.get("symbol"),
                context.get("quantity"),
                context.get("price"),
            )

        return {"error": f"Unknown action: {action}"}
