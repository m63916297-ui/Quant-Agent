"""
Trading Algorithm Agent using LangChain
"""

from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.data_models import Signal, Order, BacktestResult


class TradingAgent:
    def __init__(self, llm=None):
        self.name = "Algorithmic Trading Agent"
        self.description = (
            "Executes trading strategies, manages orders, and performs backtesting"
        )
        self.llm = llm
        self.orders: List[Order] = []
        self.signals: List[Signal] = []

        self.system_prompt = """You are an expert Algorithmic Trading Agent with 20 years of experience in quantitative trading.
        
        Your expertise includes:
        1. Strategy development (momentum, mean reversion, statistical arbitrage)
        2. Order execution optimization (TWAP, VWAP, POV)
        3. Transaction cost analysis
        4. Slippage modeling
        5. Market microstructure understanding
        
        Always consider:
        - Market impact and liquidity
        - Execution speed vs. fill quality
        - Risk management rules
        - Regulatory constraints
        """

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trading signals"""
        df = df.copy()

        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_200"] = df["close"].rolling(window=200).mean()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

        df["BB_Middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        df["ATR"] = (
            df[["high", "low", "close"]]
            .apply(lambda x: max(x) - min(x), axis=1)
            .rolling(window=14)
            .mean()
        )

        return df

    def generate_momentum_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate momentum-based trading signal"""
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]

        score = 0
        indicators = {}

        if pd.notna(latest["SMA_20"]) and pd.notna(latest["SMA_50"]):
            if latest["SMA_20"] > latest["SMA_50"]:
                score += 1
                indicators["trend"] = "bullish"
            else:
                score -= 1
                indicators["trend"] = "bearish"

        if pd.notna(latest["RSI"]):
            indicators["rsi"] = round(latest["RSI"], 2)
            if latest["RSI"] < 30:
                score += 1
                indicators["rsi_signal"] = "oversold"
            elif latest["RSI"] > 70:
                score -= 1
                indicators["rsi_signal"] = "overbought"

        if pd.notna(latest["MACD"]) and pd.notna(latest["Signal_Line"]):
            if latest["MACD"] > latest["Signal_Line"]:
                score += 1
                indicators["macd_signal"] = "bullish_cross"
            else:
                score -= 1
                indicators["macd_signal"] = "bearish_cross"

        price_position = (
            (latest["close"] - latest["BB_Lower"])
            / (latest["BB_Upper"] - latest["BB_Lower"])
            if pd.notna(latest["BB_Upper"])
            else 0.5
        )
        indicators["bb_position"] = round(price_position, 4)

        if price_position < 0.2:
            score += 1
            indicators["bb_signal"] = "near_lower_band"
        elif price_position > 0.8:
            score -= 1
            indicators["bb_signal"] = "near_upper_band"

        if score >= 2:
            direction = "buy"
            strength = min(abs(score) / 4, 1.0)
        elif score <= -2:
            direction = "sell"
            strength = min(abs(score) / 4, 1.0)
        else:
            direction = "hold"
            strength = 0.5

        signal = Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy="momentum",
            indicators=indicators,
            expiry=datetime.now() + timedelta(hours=4),
        )
        self.signals.append(signal)
        return signal

    def generate_mean_reversion_signal(
        self, df: pd.DataFrame, symbol: str, lookback: int = 20
    ) -> Signal:
        """Generate mean reversion trading signal"""
        df = df.copy()
        latest = df.iloc[-1]

        df["zscore"] = (df["close"] - df["close"].rolling(lookback).mean()) / df[
            "close"
        ].rolling(lookback).std()
        latest_zscore = df["zscore"].iloc[-1] if pd.notna(df["zscore"].iloc[-1]) else 0

        indicators = {"zscore": round(latest_zscore, 4)}

        if latest_zscore < -2:
            direction = "buy"
            strength = min(abs(latest_zscore) / 3, 1.0)
            indicators["signal"] = "oversold_extreme"
        elif latest_zscore > 2:
            direction = "sell"
            strength = min(latest_zscore / 3, 1.0)
            indicators["signal"] = "overbought_extreme"
        elif latest_zscore < -1:
            direction = "buy"
            strength = 0.6
            indicators["signal"] = "oversold"
        elif latest_zscore > 1:
            direction = "sell"
            strength = 0.6
            indicators["signal"] = "overbought"
        else:
            direction = "hold"
            strength = 0.5
            indicators["signal"] = "neutral"

        signal = Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy="mean_reversion",
            indicators=indicators,
            expiry=datetime.now() + timedelta(hours=1),
        )
        self.signals.append(signal)
        return signal

    def calculate_position_size(
        self, signal: Signal, portfolio_value: float, risk_per_trade: float = 0.02
    ) -> Dict[str, Any]:
        """Calculate optimal position size based on risk management"""
        if signal.direction == "hold":
            return {"quantity": 0, "reason": "No signal"}

        risk_amount = portfolio_value * risk_per_trade

        atr_multiplier = 2
        stop_distance = signal.indicators.get(
            "atr", signal.indicators.get("BB_Upper", 0) * 0.02
        )

        if stop_distance == 0:
            price = 100
            stop_distance = price * 0.02
        else:
            price = signal.indicators.get("current_price", 100)

        risk_per_share = stop_distance
        quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = quantity * price
        actual_risk = quantity * risk_per_share

        return {
            "quantity": quantity,
            "position_value": round(position_value, 2),
            "risk_amount": round(actual_risk, 2),
            "risk_percentage": round(actual_risk / portfolio_value * 100, 2)
            if portfolio_value > 0
            else 0,
            "stop_loss_distance": round(stop_distance, 4),
            "signal_strength": signal.strength,
        }

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Execute a trading order"""
        order_id = f"ORD-{len(self.orders) + 1:06d}"

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status="pending",
        )

        self.orders.append(order)

        if order_type == "market":
            order.status = "filled"
            order.filled_quantity = quantity
            order.average_fill_price = price if price else 100
            order.filled_at = datetime.now()

        return order

    def backtest_strategy(
        self,
        df: pd.DataFrame,
        strategy: str = "momentum",
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """Backtest a trading strategy"""
        df = df.copy()

        if strategy == "momentum":
            df = self.calculate_indicators(df)
            df["signal"] = 0
            df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 70), "signal"] = 1
            df.loc[(df["SMA_20"] < df["SMA_50"]) | (df["RSI"] > 70), "signal"] = -1
        else:
            df["signal"] = 0
            zscore = (df["close"] - df["close"].rolling(20).mean()) / df[
                "close"
            ].rolling(20).std()
            df.loc[zscore < -1.5, "signal"] = 1
            df.loc[zscore > 1.5, "signal"] = -1

        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

        equity = [initial_capital]
        position = 0

        for i in range(1, len(df)):
            if df["signal"].iloc[i - 1] == 1:
                position = equity[-1] / df["close"].iloc[i - 1]
            elif df["signal"].iloc[i - 1] == -1:
                position = 0

            equity.append(equity[-1] * (1 + df["returns"].iloc[i] * position))

        df["equity"] = equity[1:]
        df["drawdown"] = df["equity"].cummax() - df["equity"]

        total_return = (equity[-1] - initial_capital) / initial_capital
        returns = df["strategy_returns"].dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - 0.04) / volatility if volatility > 0 else 0
        max_dd = df["drawdown"].max() / initial_capital

        trades = []
        in_position = False
        entry_price = 0
        entry_date = None

        for i in range(len(df)):
            if df["signal"].iloc[i] == 1 and not in_position:
                in_position = True
                entry_price = df["close"].iloc[i]
                entry_date = df.index[i]
            elif df["signal"].iloc[i] == -1 and in_position:
                in_position = False
                exit_price = df["close"].iloc[i]
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": df.index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": (exit_price - entry_price) / entry_price,
                    }
                )

        winning_trades = [t for t in trades if t["return"] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        return BacktestResult(
            strategy_name=strategy,
            start_date=df.index[0] if len(df) > 0 else datetime.now(),
            end_date=df.index[-1] if len(df) > 0 else datetime.now(),
            initial_capital=initial_capital,
            final_capital=equity[-1],
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            total_trades=len(trades),
            equity_curve=equity,
            trades=trades,
        )

    def execute(self, action: str, context: Dict) -> Dict:
        """Execute trading action"""
        if action == "signal_momentum":
            return self.generate_momentum_signal(
                context.get("data"), context.get("symbol")
            )
        elif action == "signal_mean_reversion":
            return self.generate_mean_reversion_signal(
                context.get("data"), context.get("symbol")
            )
        elif action == "position_size":
            signal = Signal(
                symbol=context.get("symbol", ""),
                direction=context.get("direction", "hold"),
                strength=context.get("strength", 0.5),
                strategy="calculated",
                indicators=context.get("indicators", {}),
            )
            return self.calculate_position_size(
                signal, context.get("portfolio_value", 100000)
            )
        elif action == "execute_order":
            return self.execute_order(
                context.get("symbol"),
                context.get("side"),
                context.get("quantity"),
                context.get("order_type", "market"),
                context.get("price"),
                context.get("stop_price"),
            )
        elif action == "backtest":
            return self.backtest_strategy(
                context.get("data"),
                context.get("strategy", "momentum"),
                context.get("capital", 100000),
            )

        return {"error": f"Unknown action: {action}"}
