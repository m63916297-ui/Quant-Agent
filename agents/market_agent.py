"""
Market Analysis Agent using LangChain
"""

from langchain.prompts import ChatPromptTemplate
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from models.data_models import MarketData


class MarketAnalysisAgent:
    def __init__(self, llm=None):
        self.name = "Market Analysis Agent"
        self.description = "Analyzes market conditions, identifies trends, and provides market insights"
        self.llm = llm
        self.cached_data: Dict[str, pd.DataFrame] = {}

        self.system_prompt = """You are an expert Market Analysis Agent with 20 years of experience in financial markets.
        
        Your expertise includes:
        1. Technical analysis (chart patterns, indicators, trends)
        2. Market regime detection
        3. Volatility analysis
        4. Sector rotation analysis
        5. Correlation and cross-asset analysis
        
        Always consider:
        - Multiple timeframes for confirmation
        - Market context and macro conditions
        - Volume and price action
        - Risk sentiment indicators
        """

    def fetch_market_data(
        self, symbol: str, start: str, end: str = None, interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch market data from yfinance"""
        cache_key = f"{symbol}_{start}_{end}_{interval}"

        if cache_key in self.cached_data:
            return self.cached_data[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if not df.empty:
                self.cached_data[cache_key] = df
            return df
        except Exception as e:
            return pd.DataFrame()

    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend across multiple timeframes"""
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data"}

        results = {}

        for period, days in [("short", 20), ("medium", 50), ("long", 200)]:
            if len(df) >= days:
                sma = df["Close"].rolling(days).mean().iloc[-1]
                current = df["Close"].iloc[-1]
                prev_sma = df["Close"].rolling(days).mean().iloc[-2]

                trend = "bullish" if current > sma else "bearish"
                momentum = "accelerating" if current > prev_sma else "decelerating"

                results[period] = {
                    "sma": round(sma, 2),
                    "current": round(current, 2),
                    "trend": trend,
                    "momentum": momentum,
                    "distance_pct": round((current - sma) / sma * 100, 2),
                }

        return results

    def detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using volatility and trend analysis"""
        if df.empty or len(df) < 60:
            return {"error": "Insufficient data"}

        returns = df["Close"].pct_change().dropna()

        volatility_20 = returns.tail(20).std() * np.sqrt(252)
        volatility_60 = returns.tail(60).std() * np.sqrt(252)

        trend_50 = df["Close"].tail(50).iloc[0]
        trend_current = df["Close"].iloc[-1]
        trend_change = (trend_current - trend_50) / trend_50

        sma_20 = df["Close"].rolling(20).mean().iloc[-1]
        sma_50 = df["Close"].rolling(50).mean().iloc[-1]

        if volatility_20 > volatility_60 * 1.2:
            volatility_regime = "high"
        elif volatility_20 < volatility_60 * 0.8:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"

        if sma_20 > sma_50 and trend_change > 0:
            trend_regime = "bullish"
        elif sma_20 < sma_50 and trend_change < 0:
            trend_regime = "bearish"
        else:
            trend_regime = "mixed"

        if volatility_regime == "high" and trend_regime in ["bullish", "bearish"]:
            regime = f"Trending (Volatile {trend_regime})"
        elif volatility_regime == "low":
            regime = "Range-bound / Consolidating"
        elif trend_regime == "mixed":
            regime = "Uncertain / Transition"
        else:
            regime = f"{trend_regime.capitalize()} Trend"

        return {
            "regime": regime,
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
            "volatility_20d": round(volatility_20, 4),
            "volatility_60d": round(volatility_60, 4),
            "trend_strength": round(abs(trend_change) * 100, 2),
            "recommendation": self._get_regime_recommendation(regime),
        }

    def _get_regime_recommendation(self, regime: str) -> str:
        recommendations = {
            "Trending (Volatile Bullish)": "Use trend-following strategies with wider stops",
            "Trending (Volatile Bearish)": "Reduce exposure, use protective strategies",
            "Range-bound / Consolidating": "Mean reversion strategies preferred",
            "Uncertain / Transition)": "Reduce position sizes, wait for clarity",
            "Bullish Trend": "Favor momentum strategies, trail stops",
            "Bearish Trend": "Hedge positions, reduce long exposure",
        }
        return recommendations.get(regime, "Monitor and adjust")

    def analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive volatility analysis"""
        if df.empty or len(df) < 30:
            return {"error": "Insufficient data"}

        returns = df["Close"].pct_change().dropna()

        volatility_annual = returns.std() * np.sqrt(252)

        var_95 = np.percentile(returns, 0.05)
        var_99 = np.percentile(returns, 0.01)

        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        avg_vol = rolling_vol.mean()
        vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol)

        returns_sorted = returns.sort_values()
        skewness = (
            3 * (returns.mean() - returns.median()) / returns.std()
            if returns.std() > 0
            else 0
        )

        kurtosis = (
            ((returns - returns.mean()) ** 4).mean() / (returns.std() ** 4)
            if returns.std() > 0
            else 0
        )

        volatility_spike = current_vol > avg_vol * 1.5

        return {
            "annual_volatility": round(volatility_annual, 4),
            "var_95": round(var_95, 4),
            "var_99": round(var_99, 4),
            "current_vol_percentile": round(vol_percentile, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "volatility_spike": volatility_spike,
            "current_vol": round(current_vol, 4),
            "average_vol": round(avg_vol, 4),
            "interpretation": self._interpret_volatility(
                volatility_spike, vol_percentile
            ),
        }

    def _interpret_volatility(self, spike: bool, percentile: float) -> str:
        if spike:
            return "Volatility spike detected - Consider defensive positioning"
        elif percentile > 0.8:
            return "High volatility environment - Reduce position sizes"
        elif percentile < 0.2:
            return "Low volatility - Potential for breakout moves"
        else:
            return "Normal volatility levels"

    def calculate_support_resistance(
        self, df: pd.DataFrame, window: int = 20
    ) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        if df.empty or len(df) < window * 2:
            return {"error": "Insufficient data"}

        highs = df["High"].rolling(window).max()
        lows = df["Low"].rolling(window).min()

        current_price = df["Close"].iloc[-1]

        resistance_levels = []
        for i in range(window, len(df)):
            if df["High"].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df["High"].iloc[i])

        support_levels = []
        for i in range(window, len(df)):
            if df["Low"].iloc[i] == lows.iloc[i]:
                support_levels.append(df["Low"].iloc[i])

        resistance = sorted(set(resistance_levels))[-3:] if resistance_levels else []
        support = sorted(set(support_levels))[:3] if support_levels else []

        return {
            "current_price": round(current_price, 2),
            "resistance_levels": [round(r, 2) for r in resistance],
            "support_levels": [round(s, 2) for s in support],
            "distance_to_resistance": round(
                (max(resistance) - current_price) / current_price * 100, 2
            )
            if resistance
            else 0,
            "distance_to_support": round(
                (current_price - min(support)) / current_price * 100, 2
            )
            if support
            else 0,
        }

    def generate_market_report(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Generate comprehensive market analysis report"""
        end_date = datetime.now()
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        else:
            start_date = end_date - timedelta(days=365)

        df = self.fetch_market_data(symbol, start_date.strftime("%Y-%m-%d"))

        if df.empty:
            return {"error": f"Could not fetch data for {symbol}"}

        df.columns = [c.lower() for c in df.columns]

        return {
            "symbol": symbol,
            "period": period,
            "trend_analysis": self.analyze_trend(df),
            "regime": self.detect_market_regime(df),
            "volatility": self.analyze_volatility(df),
            "support_resistance": self.calculate_support_resistance(df),
            "summary": self._generate_summary(symbol, df),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_summary(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate executive summary"""
        if df.empty:
            return {}

        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        period_return = (end_price - start_price) / start_price * 100

        volume_avg = df["Volume"].mean()
        volume_current = df["Volume"].iloc[-1]
        volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1

        return {
            "period_return": round(period_return, 2),
            "current_price": round(end_price, 2),
            "volume_ratio": round(volume_ratio, 2),
            "volume_status": "above_average"
            if volume_ratio > 1.2
            else "below_average"
            if volume_ratio < 0.8
            else "average",
        }

    def compare_assets(self, symbols: List[str], period: str = "1m") -> pd.DataFrame:
        """Compare multiple assets"""
        results = []

        for symbol in symbols:
            report = self.generate_market_report(symbol, period)
            if "error" not in report:
                results.append(
                    {
                        "Symbol": symbol,
                        "Price": report["summary"].get("current_price", 0),
                        "Return": report["summary"].get("period_return", 0),
                        "Volatility": report["volatility"].get("annual_volatility", 0),
                        "Volume Status": report["summary"].get(
                            "volume_status", "unknown"
                        ),
                        "Trend": report["trend_analysis"]
                        .get("medium", {})
                        .get("trend", "unknown"),
                    }
                )

        return pd.DataFrame(results)

    def execute(self, action: str, context: Dict) -> Dict:
        """Execute market analysis action"""
        if action == "analyze_trend":
            return self.analyze_trend(context.get("data"))
        elif action == "market_regime":
            return self.detect_market_regime(context.get("data"))
        elif action == "volatility":
            return self.analyze_volatility(context.get("data"))
        elif action == "support_resistance":
            return self.calculate_support_resistance(context.get("data"))
        elif action == "report":
            return self.generate_market_report(
                context.get("symbol"), context.get("period", "1y")
            )
        elif action == "compare":
            return self.compare_assets(
                context.get("symbols", []), context.get("period", "1m")
            )

        return {"error": f"Unknown action: {action}"}
