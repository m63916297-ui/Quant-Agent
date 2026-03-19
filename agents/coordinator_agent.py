"""
Coordinator Agent - Orchestrates all Quant Agents
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from agents.portfolio_agent import PortfolioAgent
from agents.trading_agent import TradingAgent
from agents.market_agent import MarketAnalysisAgent
from agents.risk_agent import RiskAgent
from models.data_models import Portfolio, Signal, Order, RiskMetrics


class QuantCoordinator:
    def __init__(self, llm=None):
        self.name = "Quant Multi-Agent Coordinator"
        self.description = (
            "Orchestrates all quantitative agents to provide unified trading decisions"
        )
        self.llm = llm

        self.agents = {
            "portfolio": PortfolioAgent(llm),
            "trading": TradingAgent(llm),
            "market": MarketAnalysisAgent(llm),
            "risk": RiskAgent(llm),
        }

        self.decision_history = []

    def analyze_and_recommend(
        self, symbol: str, portfolio: Portfolio, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze market and generate trading recommendation"""
        signals = {}
        market_analysis = self.agents["market"].generate_market_report(symbol)

        momentum_signal = self.agents["trading"].generate_momentum_signal(
            market_data, symbol
        )
        mean_reversion_signal = self.agents["trading"].generate_mean_reversion_signal(
            market_data, symbol
        )

        signals["momentum"] = {
            "direction": momentum_signal.direction,
            "strength": momentum_signal.strength,
            "indicators": momentum_signal.indicators,
        }
        signals["mean_reversion"] = {
            "direction": mean_reversion_signal.direction,
            "strength": mean_reversion_signal.strength,
            "indicators": mean_reversion_signal.indicators,
        }

        consensus = self._calculate_consensus(signals, market_analysis)

        position_size = self.agents["trading"].calculate_position_size(
            momentum_signal, portfolio.total_value
        )

        decision = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_analysis": market_analysis,
            "signals": signals,
            "consensus": consensus,
            "recommendation": self._generate_recommendation(consensus, position_size),
            "position_sizing": position_size,
            "risk_assessment": self._assess_trade_risk(symbol, market_data, consensus),
        }

        self.decision_history.append(decision)
        return decision

    def _calculate_consensus(
        self, signals: Dict, market_analysis: Dict
    ) -> Dict[str, Any]:
        """Calculate consensus from multiple signals"""
        directions = {"buy": 1, "sell": -1, "hold": 0}
        weighted_score = 0
        total_weight = 0

        for strategy, signal_data in signals.items():
            direction = signal_data["direction"]
            strength = signal_data["strength"]
            weight = 1.0 if strategy == "momentum" else 0.8

            weighted_score += directions.get(direction, 0) * strength * weight
            total_weight += weight

        consensus_score = weighted_score / total_weight if total_weight > 0 else 0

        trend = (
            market_analysis.get("trend_analysis", {})
            .get("medium", {})
            .get("trend", "unknown")
        )
        regime = market_analysis.get("regime", {}).get("regime", "unknown")

        if trend == "bullish":
            weighted_score += 0.3
        elif trend == "bearish":
            weighted_score -= 0.3

        consensus_score = weighted_score / (total_weight + 1)

        if consensus_score > 0.3:
            final_direction = "buy"
        elif consensus_score < -0.3:
            final_direction = "sell"
        else:
            final_direction = "hold"

        confidence = min(abs(consensus_score), 1.0)

        return {
            "direction": final_direction,
            "confidence": round(confidence, 4),
            "score": round(consensus_score, 4),
            "regime": regime,
            "trend": trend,
        }

    def _generate_recommendation(
        self, consensus: Dict, position_size: Dict
    ) -> Dict[str, Any]:
        """Generate final trading recommendation"""
        direction = consensus["direction"]
        confidence = consensus["confidence"]

        if direction == "hold" or confidence < 0.4:
            action = "no_action"
            reason = "Insufficient conviction or conflicting signals"
        elif confidence >= 0.7:
            action = f"execute_{direction}"
            reason = f"High confidence {direction} signal"
        else:
            action = f"consider_{direction}"
            reason = f"Moderate confidence {direction} signal"

        return {
            "action": action,
            "direction": direction,
            "confidence": confidence,
            "reason": reason,
            "position_size": position_size.get("quantity", 0),
            "risk_amount": position_size.get("risk_amount", 0),
        }

    def _assess_trade_risk(
        self, symbol: str, market_data: pd.DataFrame, consensus: Dict
    ) -> Dict[str, Any]:
        """Assess risk for potential trade"""
        if market_data.empty:
            return {"error": "No market data"}

        returns = market_data["Close"].pct_change().dropna()

        volatility = returns.std() * np.sqrt(252)
        var_95 = np.percentile(returns, 0.05)

        regime = consensus.get("regime", "unknown")

        risk_level = "low"
        risk_factors = []

        if volatility > 0.30:
            risk_level = "high"
            risk_factors.append("High market volatility")

        if "Volatile" in regime:
            risk_level = "high"
            risk_factors.append("Volatile market regime")
        elif regime == "Range-bound":
            risk_factors.append("Range-bound market - mean reversion preferred")

        if abs(var_95) > 0.03:
            risk_factors.append(f"High daily VaR: {var_95:.2%}")

        return {
            "risk_level": risk_level,
            "annual_volatility": round(volatility, 4),
            "var_95_daily": round(var_95, 4),
            "risk_factors": risk_factors,
            "adjustments": self._get_risk_adjustments(risk_level),
        }

    def _get_risk_adjustments(self, risk_level: str) -> List[str]:
        """Get position size adjustments based on risk level"""
        adjustments = {
            "low": ["Standard position sizing acceptable"],
            "medium": [
                "Consider reducing position size by 20-30%",
                "Use tighter stops",
            ],
            "high": [
                "Reduce position size by 50%",
                "Use protective stops mandatory",
                "Consider hedging with options",
            ],
        }
        return adjustments.get(risk_level, adjustments["low"])

    def get_portfolio_analysis(
        self, portfolio: Portfolio, returns_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis"""
        risk_metrics = self.agents["risk"].calculate_risk_adjusted_metrics(
            returns_df.sum(axis=1).pct_change().dropna()
        )

        rebalance_signal = self.agents["portfolio"].generate_rebalancing_signal(
            portfolio
        )

        risk_limits_check = self.agents["risk"].check_risk_limits(
            RiskMetrics(
                portfolio_value=portfolio.total_value,
                volatility=risk_metrics.get("annual_volatility", 0),
                var_95=0,
                cvar_95=0,
                max_drawdown=0,
            ),
            {p.symbol: p.quantity for p in portfolio.positions},
            {p.symbol: p.current_price for p in portfolio.positions},
            portfolio.total_value,
        )

        return {
            "portfolio_id": portfolio.id,
            "total_value": round(portfolio.total_value, 2),
            "cash": round(portfolio.cash, 2),
            "positions_count": len(portfolio.positions),
            "weights": portfolio.get_weights(),
            "risk_metrics": risk_metrics,
            "rebalance_signal": rebalance_signal,
            "risk_limits_status": risk_limits_check,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents in the system"""
        return {
            "coordinator": {
                "name": self.name,
                "decisions_made": len(self.decision_history),
            },
            "agents": {
                name: {
                    "name": agent.name,
                    "description": agent.description,
                    "status": "active",
                }
                for name, agent in self.agents.items()
            },
            "system_time": datetime.now().isoformat(),
        }

    def execute(self, action: str, context: Dict) -> Dict:
        """Execute coordinated action across agents"""
        if action == "analyze":
            return self.analyze_and_recommend(
                context.get("symbol"),
                context.get("portfolio"),
                context.get("market_data"),
            )
        elif action == "portfolio_analysis":
            return self.get_portfolio_analysis(
                context.get("portfolio"), context.get("returns_df")
            )
        elif action == "system_status":
            return self.get_system_status()
        elif action in self.agents:
            return self.agents[action].execute(
                context.get("agent_action"), context.get("agent_context", {})
            )

        return {"error": f"Unknown action: {action}"}
