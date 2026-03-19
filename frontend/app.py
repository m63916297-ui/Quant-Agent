"""
Quant AI - Multi-Agent Trading System
Streamlit Frontend Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_agent import PortfolioAgent
from agents.trading_agent import TradingAgent
from agents.market_agent import MarketAnalysisAgent
from agents.risk_agent import RiskAgent
from agents.coordinator_agent import QuantCoordinator
from models.data_models import Portfolio, Position

st.set_page_config(
    page_title="Quant AI - Multi-Agent Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .signal-buy { color: #00c853; font-weight: bold; }
    .signal-sell { color: #ff1744; font-weight: bold; }
    .signal-hold { color: #ff9800; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_agents():
    if "coordinator" not in st.session_state:
        st.session_state.coordinator = QuantCoordinator()
        st.session_state.portfolio_agent = PortfolioAgent()
        st.session_state.trading_agent = TradingAgent()
        st.session_state.market_agent = MarketAnalysisAgent()
        st.session_state.risk_agent = RiskAgent()


@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        return pd.DataFrame()


def plot_price_chart(df: pd.DataFrame, symbol: str):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_20"],
                line=dict(color="orange", width=1),
                name="SMA 20",
            )
        )

    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_50"],
                line=dict(color="blue", width=1),
                name="SMA 50",
            )
        )

    fig.update_layout(
        title=f"{symbol} Price Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
        height=400,
        xaxis_rangeslider_visible=False,
    )

    return fig


def plot_equity_curve(equity: list, title: str = "Equity Curve"):
    df = pd.DataFrame({"Date": range(len(equity)), "Value": equity})
    fig = px.line(df, x="Date", y="Value", title=title)
    fig.update_layout(template="plotly_dark", height=300)
    fig.update_traces(line_color="#00c853")
    return fig


def plot_returns_distribution(returns: pd.Series):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name="Returns", marker_color="#667eea")
    )

    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
    )

    return fig


def display_signal_indicator(direction: str, strength: float):
    colors = {"buy": "#00c853", "sell": "#ff1744", "hold": "#ff9800"}
    emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}

    return f"""
    <div style="background: {colors[direction]}22; padding: 15px; border-radius: 10px; border-left: 4px solid {colors[direction]};">
        <h3 style="color: {colors[direction]}; margin: 0;">
            {emoji[direction]} {direction.upper()}
        </h3>
        <p style="margin: 5px 0 0 0;">Confidence: {strength:.1%}</p>
    </div>
    """


def main():
    initialize_agents()

    st.markdown('<h1 class="main-header">📈 Quant AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Agent Quantitative Trading System powered by AI</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Configuration")

        with st.expander("🔐 API Settings", expanded=False):
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Optional: For enhanced AI analysis",
            )

        st.divider()

        st.subheader("📊 Market Data")
        default_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "SPY",
        ]
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL",
            help="Enter stock symbols",
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",")]

        period_options = {
            "1 Week": "1w",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
        }
        period = st.selectbox("Time Period", list(period_options.keys()), index=4)
        period_value = period_options[period]

        st.divider()

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dashboard",
            "📈 Market Analysis",
            "💼 Portfolio",
            "🎯 Signals & Trading",
            "⚠️ Risk Management",
        ]
    )

    with tab1:
        st.header("System Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Agents", "4", "Portfolio, Trading, Market, Risk")

        with col2:
            st.metric("Symbols Tracked", len(symbols), "+2 from last session")

        with col3:
            st.metric("System Status", "🟢 Online", "All systems operational")

        with col4:
            st.metric("Last Analysis", datetime.now().strftime("%H:%M"), "")

        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            if symbols:
                symbol = symbols[0]
                df = fetch_stock_data(symbol, period_value)

                if not df.empty:
                    st.subheader(f"{symbol} Overview")
                    tab_chart1, tab_chart2 = st.tabs(
                        ["Price Chart", "Technical Analysis"]
                    )

                    with tab_chart1:
                        st.plotly_chart(
                            plot_price_chart(df, symbol), use_container_width=True
                        )

                    with tab_chart2:
                        returns = df["close"].pct_change().dropna()
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.plotly_chart(
                                plot_returns_distribution(returns),
                                use_container_width=True,
                            )

                        with col_b:
                            st.write("### Key Metrics")
                            st.write(f"**Daily Return:** {returns.iloc[-1]:.2%}")
                            st.write(
                                f"**Volatility ( annualized):** {returns.std() * np.sqrt(252):.2%}"
                            )
                            st.write(f"**Max Return:** {returns.max():.2%}")
                            st.write(f"**Min Return:** {returns.min():.2%}")
                            st.write(f"**Skewness:** {returns.skew():.2f}")
                            st.write(f"**Kurtosis:** {returns.kurtosis():.2f}")

        with col2:
            st.subheader("Quick Signals")
            for sym in symbols[:3]:
                data = fetch_stock_data(sym, "1mo")
                if not data.empty:
                    signal = st.session_state.trading_agent.generate_momentum_signal(
                        data, sym
                    )
                    st.markdown(
                        display_signal_indicator(signal.direction, signal.strength),
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{sym} - {datetime.now().strftime('%H:%M')}")
                    st.divider()

    with tab2:
        st.header("Market Analysis")

        selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)

        if st.button("🔍 Analyze Market", use_container_width=True):
            with st.spinner("Analyzing market conditions..."):
                report = st.session_state.market_agent.generate_market_report(
                    selected_symbol, period_value
                )

                if "error" not in report:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"{selected_symbol} Market Report")
                        st.write(f"**Period:** {report['period']}")
                        st.write(f"**Regime:** {report['regime']['regime']}")
                        st.write(f"**Trend:** {report['regime']['trend_regime']}")
                        st.write(
                            f"**Recommendation:** {report['regime']['recommendation']}"
                        )

                    with col2:
                        st.subheader("Trend Analysis")
                        trend = report.get("trend_analysis", {})
                        for term in ["short", "medium", "long"]:
                            if term in trend:
                                t = trend[term]
                                st.write(
                                    f"**{term.upper()}:** {t.get('trend', 'N/A').title()} ({t.get('distance_pct', 0):.1f}% from SMA)"
                                )

                    st.divider()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Volatility Analysis")
                        vol = report.get("volatility", {})
                        st.write(
                            f"**Annual Volatility:** {vol.get('annual_volatility', 0):.2%}"
                        )
                        st.write(f"**VaR 95%:** {vol.get('var_95', 0):.2%}")
                        st.write(
                            f"**Vol Percentile:** {vol.get('current_vol_percentile', 0):.1%}"
                        )
                        st.write(f"**Status:** {vol.get('interpretation', 'N/A')}")

                    with col2:
                        st.subheader("Support & Resistance")
                        sr = report.get("support_resistance", {})
                        st.write(
                            f"**Current Price:** ${sr.get('current_price', 0):.2f}"
                        )
                        st.write("**Resistance Levels:**")
                        for i, r in enumerate(sr.get("resistance_levels", [])[:3], 1):
                            st.write(f"  R{i}: ${r:.2f}")
                        st.write("**Support Levels:**")
                        for i, s in enumerate(sr.get("support_levels", [])[:3], 1):
                            st.write(f"  S{i}: ${s:.2f}")

                    with col3:
                        st.subheader("Summary")
                        summary = report.get("summary", {})
                        st.write(
                            f"**Period Return:** {summary.get('period_return', 0):.2f}%"
                        )
                        st.write(
                            f"**Volume Status:** {summary.get('volume_status', 'N/A').replace('_', ' ').title()}"
                        )

    with tab3:
        st.header("Portfolio Management")

        if "portfolio" not in st.session_state:
            st.session_state.portfolio = Portfolio(
                id="main", name="My Portfolio", cash=100000.0, total_value=100000.0
            )

        portfolio = st.session_state.portfolio

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${portfolio.total_value:,.2f}")
        with col2:
            st.metric(
                "Cash",
                f"${portfolio.cash:,.2f}",
                f"{(portfolio.cash / portfolio.total_value) * 100:.1f}%",
            )
        with col3:
            st.metric("Positions", len(portfolio.positions))
        with col4:
            returns = (portfolio.total_value - 100000) / 100000
            st.metric("Total Return", f"{returns:.2%}")

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Add Position")
            with st.form("add_position_form"):
                new_symbol = st.text_input("Symbol").upper()
                quantity = st.number_input("Quantity", min_value=1, value=10)
                price = st.number_input("Price per Share", min_value=0.01, value=100.0)

                if st.form_submit_button("Add Position"):
                    result = st.session_state.portfolio_agent.add_position(
                        "main", new_symbol, quantity, price
                    )
                    if result.get("success"):
                        st.success(f"Added {quantity} shares of {new_symbol}")
                        st.rerun()
                    else:
                        st.error(result.get("error", "Failed to add position"))

        with col2:
            st.subheader("Optimize Portfolio")
            if len(portfolio.positions) >= 2:
                if st.button("🎯 Optimize Weights", use_container_width=True):
                    st.info("Select at least 2 symbols to optimize")
            else:
                st.info("Add at least 2 positions to enable optimization")

        st.divider()

        if portfolio.positions:
            st.subheader("Current Positions")
            positions_df = pd.DataFrame(
                [
                    {
                        "Symbol": p.symbol,
                        "Quantity": p.quantity,
                        "Entry Price": f"${p.entry_price:.2f}",
                        "Current Price": f"${p.current_price:.2f}",
                        "Market Value": f"${p.market_value:.2f}",
                        "P&L": f"${p.unrealized_pnl:.2f}",
                        "Return": f"{p.unrealized_return:.2%}",
                    }
                    for p in portfolio.positions
                ]
            )
            st.dataframe(positions_df, use_container_width=True, hide_index=True)

            weights = portfolio.get_weights()
            if weights:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=list(weights.keys()),
                            values=list(weights.values()),
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Set3),
                        )
                    ]
                )
                fig.update_layout(title="Portfolio Allocation", height=400)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Trading Signals & Execution")

        col1, col2 = st.columns([1, 1])

        with col1:
            signal_symbol = st.selectbox("Symbol", symbols)
            signal_period = st.selectbox(
                "Signal Period", ["1mo", "3mo", "6mo", "1y"], index=2
            )

        with col2:
            strategy = st.selectbox(
                "Strategy", ["Momentum", "Mean Reversion", "Combined"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Signals", use_container_width=True):
                with st.spinner("Analyzing..."):
                    data = fetch_stock_data(signal_symbol, signal_period)

                    if not data.empty:
                        if strategy in ["Momentum", "Combined"]:
                            mom_signal = (
                                st.session_state.trading_agent.generate_momentum_signal(
                                    data, signal_symbol
                                )
                            )
                            st.markdown("### Momentum Signal")
                            st.markdown(
                                display_signal_indicator(
                                    mom_signal.direction, mom_signal.strength
                                ),
                                unsafe_allow_html=True,
                            )

                            st.write("**Indicators:**")
                            for k, v in mom_signal.indicators.items():
                                st.write(f"  - {k}: {v}")

                        if strategy in ["Mean Reversion", "Combined"]:
                            mr_signal = st.session_state.trading_agent.generate_mean_reversion_signal(
                                data, signal_symbol
                            )
                            st.markdown("### Mean Reversion Signal")
                            st.markdown(
                                display_signal_indicator(
                                    mr_signal.direction, mr_signal.strength
                                ),
                                unsafe_allow_html=True,
                            )

                            st.write("**Indicators:**")
                            for k, v in mr_signal.indicators.items():
                                st.write(f"  - {k}: {v}")

        with col2:
            st.subheader("Position Sizing")
            risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
            st.write(
                f"**Risk Amount:** ${portfolio.total_value * risk_percent / 100:,.2f}"
            )

            if st.button("💰 Calculate Position Size"):
                st.info("Run signal generation first")

        with col3:
            st.subheader("Backtest Strategy")
            bt_symbol = st.selectbox("Backtest Symbol", symbols)

            if st.button("🔬 Run Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    data = fetch_stock_data(bt_symbol, "2y")

                    if not data.empty:
                        result = st.session_state.trading_agent.backtest_strategy(
                            data, strategy="momentum", initial_capital=100000
                        )

                        st.write(f"**Total Return:** {result.total_return:.2%}")
                        st.write(f"**Sharpe Ratio:** {result.sharpe_ratio:.2f}")
                        st.write(f"**Max Drawdown:** {result.max_drawdown:.2%}")
                        st.write(f"**Win Rate:** {result.win_rate:.1%}")
                        st.write(f"**Total Trades:** {result.total_trades}")

                        if result.equity_curve:
                            st.plotly_chart(
                                plot_equity_curve(
                                    result.equity_curve,
                                    f"Backtest Equity - {result.strategy_name}",
                                ),
                                use_container_width=True,
                            )

    with tab5:
        st.header("Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("VaR Analysis")

            confidence = st.slider("Confidence Level (%)", 90, 99, 95, 1)

            if st.button("Calculate VaR", use_container_width=True):
                if portfolio.positions:
                    st.info("Select symbols with historical data for VaR calculation")
                else:
                    st.info("Add positions to calculate portfolio VaR")

        with col2:
            st.subheader("Stress Testing")

            scenarios = [
                {"name": "Market Crash (-20%)", "default": -0.20},
                {"name": "Moderate Drop (-10%)", "default": -0.10},
                {"name": "High Volatility (+50% vol)", "default": -0.05},
            ]

            if st.button("Run Stress Test", use_container_width=True):
                if portfolio.positions:
                    result = st.session_state.risk_agent.stress_test(
                        {p.symbol: p.quantity for p in portfolio.positions},
                        {p.symbol: p.current_price for p in portfolio.positions},
                        scenarios,
                        portfolio.total_value,
                    )

                    st.write(
                        "**Worst Case:**",
                        result.get("worst_case", {}).get("scenario_name", "N/A"),
                    )
                    st.write(
                        f"  P&L: ${result.get('worst_case', {}).get('total_pnl', 0):,.2f}"
                    )

                    st.write(
                        "**Best Case:**",
                        result.get("best_case", {}).get("scenario_name", "N/A"),
                    )
                    st.write(
                        f"  P&L: ${result.get('best_case', {}).get('total_pnl', 0):,.2f}"
                    )
                else:
                    st.info("Add positions to run stress tests")

        st.divider()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Risk Limits Status")

            limits = st.session_state.risk_agent.risk_limits

            for limit_name, limit_value in limits.items():
                st.write(
                    f"**{limit_name.replace('_', ' ').title()}:** {limit_value:.2%}"
                )

        with col2:
            st.subheader("Current Risk Metrics")
            st.write("**Portfolio Volatility:** -")
            st.write("**VaR (95%):** -")
            st.write("**Max Drawdown:** -")
            st.write("**Sharpe Ratio:** -")

    st.divider()

    with st.expander("📋 System Logs"):
        st.text(f"[{datetime.now().strftime('%H:%M:%S')}] Quant AI System initialized")
        st.text(
            f"[{datetime.now().strftime('%H:%M:%S')}] 4 agents loaded: Portfolio, Trading, Market, Risk"
        )
        st.text(
            f"[{datetime.now().strftime('%H:%M:%S')}] {len(symbols)} symbols configured"
        )


if __name__ == "__main__":
    main()
