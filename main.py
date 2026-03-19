"""
Quant Multi-Agent System
Quick start script
"""

from agents.coordinator_agent import QuantCoordinator
from agents.portfolio_agent import PortfolioAgent
from agents.trading_agent import TradingAgent
from agents.market_agent import MarketAnalysisAgent
from agents.risk_agent import RiskAgent
import yfinance as yf
import pandas as pd


def main():
    print("=" * 60)
    print("QUANT AI - Multi-Agent Trading System")
    print("=" * 60)

    coordinator = QuantCoordinator()
    portfolio_agent = PortfolioAgent()
    trading_agent = TradingAgent()
    market_agent = MarketAnalysisAgent()
    risk_agent = RiskAgent()

    print("\n[1] Initializing agents...")
    print(f"    - Portfolio Agent: {portfolio_agent.name}")
    print(f"    - Trading Agent: {trading_agent.name}")
    print(f"    - Market Agent: {market_agent.name}")
    print(f"    - Risk Agent: {risk_agent.name}")

    print("\n[2] Fetching market data for AAPL...")
    df = yf.download("AAPL", start="2023-01-01", progress=False)
    df.columns = [c.lower() for c in df.columns]
    print(f"    - Retrieved {len(df)} days of data")

    print("\n[3] Generating trading signals...")
    momentum_signal = trading_agent.generate_momentum_signal(df, "AAPL")
    print(f"    - Momentum Signal: {momentum_signal.direction.upper()}")
    print(f"    - Strength: {momentum_signal.strength:.1%}")

    mean_rev_signal = trading_agent.generate_mean_reversion_signal(df, "AAPL")
    print(f"    - Mean Reversion Signal: {mean_rev_signal.direction.upper()}")
    print(f"    - Strength: {mean_rev_signal.strength:.1%}")

    print("\n[4] Market analysis...")
    report = market_agent.generate_market_report("AAPL", "3mo")
    print(f"    - Regime: {report['regime']['regime']}")
    print(f"    - Trend: {report['regime']['trend_regime']}")

    print("\n[5] Portfolio optimization example...")
    test_returns = pd.DataFrame(
        {
            "AAPL": df["close"].pct_change().dropna(),
        }
    )

    print("\n[6] Risk metrics...")
    print(f"    - System ready for analysis")

    print("\n" + "=" * 60)
    print("To start the Streamlit UI, run:")
    print("    streamlit run frontend/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
