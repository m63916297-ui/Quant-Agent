"""
Risk Management Agent using LangChain
"""

from langchain.prompts import ChatPromptTemplate
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from models.data_models import RiskMetrics


class RiskAgent:
    def __init__(self, llm=None):
        self.name = "Risk Management Agent"
        self.description = (
            "Manages portfolio risk, calculates VaR, and monitors risk limits"
        )
        self.llm = llm
        self.risk_limits = {
            "max_var_daily": 0.02,
            "max_volatility": 0.20,
            "max_drawdown": 0.15,
            "max_position_size": 0.15,
            "max_sector_exposure": 0.30,
        }

        self.system_prompt = """You are an expert Risk Management Agent with 20 years of experience in quantitative risk management.
        
        Your expertise includes:
        1. Value at Risk (VaR) calculation methods (Historical, Parametric, Monte Carlo)
        2. Expected Shortfall (CVaR/ES) analysis
        3. Stress testing and scenario analysis
        4. Risk factor decomposition
        5. Correlation matrix modeling
        
        Always consider:
        - Tail risk and extreme events
        - Diversification benefits
        - Correlation instability during crises
        - Liquidity risk
        """

    def calculate_var_historical(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Calculate historical VaR"""
        if len(returns) < 30:
            return 0.0
        var = np.percentile(returns, (1 - confidence) * 100)
        return var

    def calculate_var_parametric(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        if len(returns) < 30:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        z_score = 1.645 if confidence == 0.95 else 2.326
        var = mean - z_score * std
        return var

    def calculate_var_monte_carlo(
        self, returns: pd.Series, confidence: float = 0.95, simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR"""
        if len(returns) < 30:
            return 0.0

        mean = returns.mean()
        std = returns.std()

        simulated_returns = np.random.normal(mean, std, simulations)
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        return var

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) < 30:
            return 0.0

        var = self.calculate_var_historical(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var * 1.5

    def calculate_var_portfolio(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        returns_df: pd.DataFrame,
        confidence: float = 0.95,
        portfolio_value: float = 100000,
    ) -> Dict[str, Any]:
        """Calculate portfolio VaR with position-level decomposition"""
        if returns_df.empty or len(returns_df) < 30:
            return {"error": "Insufficient return data"}

        portfolio_returns = []
        for i in range(len(returns_df)):
            daily_return = 0
            for symbol in positions.keys():
                if symbol in returns_df.columns:
                    weight = (
                        positions[symbol] * prices.get(symbol, 0)
                    ) / portfolio_value
                    daily_return += weight * returns_df[symbol].iloc[i]
            portfolio_returns.append(daily_return)

        portfolio_returns = pd.Series(portfolio_returns)

        var_hist = self.calculate_var_historical(portfolio_returns, confidence)
        var_param = self.calculate_var_parametric(portfolio_returns, confidence)
        var_mc = self.calculate_var_monte_carlo(portfolio_returns, confidence)

        cvar = self.calculate_cvar(portfolio_returns, confidence)

        var_dollar = abs(var_hist) * portfolio_value
        cvar_dollar = abs(cvar) * portfolio_value

        component_vars = {}
        for symbol in positions.keys():
            if symbol in returns_df.columns:
                weight = (positions[symbol] * prices.get(symbol, 0)) / portfolio_value
                marginal_var = weight * returns_df[symbol].std() * 1.645
                component_vars[symbol] = {
                    "weight": round(weight, 4),
                    "marginal_var": round(marginal_var, 6),
                    "contribution": round(marginal_var / abs(var_param) * 100, 2)
                    if var_param != 0
                    else 0,
                }

        return {
            "var_95_historical": round(var_hist, 6),
            "var_95_parametric": round(var_param, 6),
            "var_95_monte_carlo": round(var_mc, 6),
            "cvar_95": round(cvar, 6),
            "var_dollar": round(var_dollar, 2),
            "cvar_dollar": round(cvar_dollar, 2),
            "portfolio_volatility": round(portfolio_returns.std() * np.sqrt(252), 4),
            "component_var": component_vars,
        }

    def stress_test(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        scenarios: List[Dict[str, float]],
        portfolio_value: float = 100000,
    ) -> Dict[str, Any]:
        """Perform stress testing with custom scenarios"""
        results = []

        for scenario in scenarios:
            scenario_pnl = 0
            scenario_details = {}

            for symbol in positions.keys():
                weight = (positions[symbol] * prices.get(symbol, 0)) / portfolio_value
                price_change = scenario.get(symbol, scenario.get("default", 0))
                pnl = weight * price_change * portfolio_value
                scenario_pnl += pnl
                scenario_details[symbol] = {
                    "weight": round(weight, 4),
                    "shock": price_change,
                    "pnl": round(pnl, 2),
                }

            results.append(
                {
                    "scenario_name": scenario.get("name", "Unnamed"),
                    "total_pnl": round(scenario_pnl, 2),
                    "pnl_percentage": round(scenario_pnl / portfolio_value * 100, 2),
                    "details": scenario_details,
                }
            )

        sorted_results = sorted(results, key=lambda x: x["total_pnl"])

        return {
            "worst_case": sorted_results[0] if sorted_results else {},
            "best_case": sorted_results[-1] if sorted_results else {},
            "all_scenarios": results,
            "average_impact": round(
                sum(r["total_pnl"] for r in results) / len(results), 2
            )
            if results
            else 0,
        }

    def calculate_max_drawdown(self, equity_curve: List[float]) -> Dict[str, Any]:
        """Calculate maximum drawdown metrics"""
        if not equity_curve or len(equity_curve) < 2:
            return {"error": "Insufficient data"}

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        peak_idx = equity_series[:max_dd_idx].idxmax()
        trough_idx = max_dd_idx

        recovery_idx = None
        for i in range(max_dd_idx + 1, len(equity_series)):
            if equity_series.iloc[i] >= equity_series.iloc[peak_idx]:
                recovery_idx = i
                break

        return {
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "peak_value": round(equity_series.iloc[peak_idx], 2),
            "trough_value": round(equity_series.iloc[trough_idx], 2),
            "peak_date_index": int(peak_idx),
            "trough_date_index": int(trough_idx),
            "recovery_date_index": int(recovery_idx) if recovery_idx else None,
            "recovery_time_days": int(recovery_idx - trough_idx)
            if recovery_idx
            else None,
        }

    def check_risk_limits(
        self,
        risk_metrics: RiskMetrics,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Check if current positions violate risk limits"""
        violations = []
        warnings = []

        for symbol, position_value in positions.items():
            weight = (
                (position_value * prices.get(symbol, 0)) / portfolio_value
                if portfolio_value > 0
                else 0
            )

            if weight > self.risk_limits["max_position_size"]:
                violations.append(
                    {
                        "limit": "max_position_size",
                        "symbol": symbol,
                        "current": round(weight, 4),
                        "limit_value": self.risk_limits["max_position_size"],
                        "severity": "critical",
                    }
                )
            elif weight > self.risk_limits["max_position_size"] * 0.9:
                warnings.append(
                    {
                        "limit": "max_position_size",
                        "symbol": symbol,
                        "current": round(weight, 4),
                        "limit_value": self.risk_limits["max_position_size"],
                        "severity": "warning",
                    }
                )

        if risk_metrics.var_95 / portfolio_value > self.risk_limits["max_var_daily"]:
            violations.append(
                {
                    "limit": "max_var_daily",
                    "current": round(risk_metrics.var_95 / portfolio_value, 4),
                    "limit_value": self.risk_limits["max_var_daily"],
                    "severity": "critical",
                }
            )

        if risk_metrics.volatility > self.risk_limits["max_volatility"]:
            violations.append(
                {
                    "limit": "max_volatility",
                    "current": round(risk_metrics.volatility, 4),
                    "limit_value": self.risk_limits["max_volatility"],
                    "severity": "critical",
                }
            )

        if abs(risk_metrics.max_drawdown) > self.risk_limits["max_drawdown"]:
            violations.append(
                {
                    "limit": "max_drawdown",
                    "current": round(risk_metrics.max_drawdown, 4),
                    "limit_value": -self.risk_limits["max_drawdown"],
                    "severity": "critical",
                }
            )

        return {
            "status": "VIOLATION" if violations else "WARNING" if warnings else "OK",
            "violations": violations,
            "warnings": warnings,
            "recommendations": self._generate_risk_recommendations(
                violations, warnings
            ),
        }

    def _generate_risk_recommendations(
        self, violations: List, warnings: List
    ) -> List[str]:
        recommendations = []

        if any(v["limit"] == "max_var_daily" for v in violations):
            recommendations.append(
                "Reduce overall portfolio exposure to meet VaR limits"
            )

        if any(v["limit"] == "max_volatility" for v in violations):
            recommendations.append(
                "Diversify portfolio or reduce high-volatility positions"
            )

        if any(v["limit"] == "max_position_size" for v in violations):
            recommendations.append(
                "Reduce oversized position to comply with single-name limits"
            )

        if any(v["limit"] == "max_drawdown" for v in violations):
            recommendations.append("Consider implementing stop-losses or reducing risk")

        if not recommendations:
            recommendations.append("Risk metrics within acceptable limits")

        return recommendations

    def calculate_risk_adjusted_metrics(
        self, returns: pd.Series, risk_free_rate: float = 0.04
    ) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted performance metrics"""
        if len(returns) < 30:
            return {"error": "Insufficient data"}

        excess_returns = returns - risk_free_rate / 252

        mean_excess = excess_returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)

        sharpe = mean_excess / volatility if volatility > 0 else 0

        downside_returns = returns[returns < 0]
        downside_vol = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino = mean_excess / downside_vol if downside_vol > 0 else 0

        treynor = mean_excess / 1.0 if "beta" in returns else 0

        capm_return = risk_free_rate + 0.5 * (returns.mean() * 252 - risk_free_rate)
        alpha = mean_excess - capm_return

        information_ratio = mean_excess / volatility if volatility > 0 else 0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "treynor_ratio": round(treynor, 4),
            "alpha_annualized": round(alpha, 4),
            "information_ratio": round(information_ratio, 4),
            "annual_return": round(returns.mean() * 252, 4),
            "annual_volatility": round(volatility, 4),
            "calmar_ratio": round(
                returns.mean()
                * 252
                / abs(
                    self.calculate_max_drawdown(returns.cumsum().tolist()).get(
                        "max_drawdown", 0.01
                    )
                ),
                2,
            ),
        }

    def execute(self, action: str, context: Dict) -> Dict:
        """Execute risk management action"""
        if action == "calculate_var":
            return self.calculate_var_portfolio(
                context.get("positions", {}),
                context.get("prices", {}),
                context.get("returns_df"),
                context.get("confidence", 0.95),
                context.get("portfolio_value", 100000),
            )
        elif action == "stress_test":
            return self.stress_test(
                context.get("positions", {}),
                context.get("prices", {}),
                context.get("scenarios", []),
                context.get("portfolio_value", 100000),
            )
        elif action == "check_limits":
            return self.check_risk_limits(
                context.get("risk_metrics"),
                context.get("positions", {}),
                context.get("prices", {}),
                context.get("portfolio_value", 100000),
            )
        elif action == "risk_metrics":
            return self.calculate_risk_adjusted_metrics(
                context.get("returns"), context.get("risk_free_rate", 0.04)
            )
        elif action == "drawdown":
            return self.calculate_max_drawdown(context.get("equity_curve", []))

        return {"error": f"Unknown action: {action}"}
