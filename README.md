# Quant AI - Multi-Agent Quantitative Trading System

Sistema multiagente cuantitativo profesional para análisis de mercado, gestión de portafolios y trading algorítmico.

## Visión General

Quant AI implementa un enfoque sistemático al trading cuantitativo, combinando múltiples agentes especializados que trabajan de forma coordinada para generar señales de inversión basadas en datos, gestión de riesgos y optimización de portafolios.

El sistema está diseñado para actuar como un **Quantitative Analyst Virtual** que automatiza tareas de:
- Análisis técnico y detección de regímenes de mercado
- Generación de señales de trading (momentum, mean reversion)
- Optimización de portafolios usando Modern Portfolio Theory
- Gestión de riesgos (VaR, CVaR, stress testing)
- Backtesting y validación de estrategias

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Quant Coordinator                         │
│            (Orquestador Multi-Agente)                        │
├─────────────┬─────────────┬──────────────┬──────────────────┤
│   Market    │   Trading   │  Portfolio   │      Risk        │
│   Agent     │   Agent     │    Agent     │      Agent       │
├─────────────┼─────────────┼──────────────┼──────────────────┤
│ • Análisis  │ • Señales   │ • MPT        │ • VaR (Hist,     │
│   técnico   │   momentum  │ • Optimizac. │   Param, MC)     │
│ • Regímenes │ • Mean      │ • Rebalanceo  │ • CVaR/ES       │
│ • Volatil.  │   reversion │ • Allocation  │ • Stress test    │
│ • S/R levels│ • Position  │ • Tracking   │ • Límites        │
│             │   sizing    │   error      │ • Drawdown       │
└─────────────┴─────────────┴──────────────┴──────────────────┘
```

### Agente de Mercado (Market Analysis Agent)

**Responsabilidades:**
- Obtención y procesamiento de datos de mercado via yFinance
- Análisis técnico multi-timeframe (corto, medio, largo plazo)
- Detección de regímenes de mercado (trending, range-bound, volatile)
- Identificación de niveles de soporte y resistencia
- Análisis de volatilidad (GARCH implícito, rolling volatility)

**Metodologías:**
```python
# Tendencias: SMAs cruzadas (20, 50, 200 períodos)
# Momentum: RSI(14), MACD(12,26,9)
# Volatilidad: Bollinger Bands (±2σ, ventana 20)
# Regime Detection: Comparación de volatilidades 20d vs 60d
```

### Agente de Trading (Trading Algorithm Agent)

**Responsabilidades:**
- Generación de señales de trading cuantitativas
- Cálculo de tamaño de posición basado en risk management
- Ejecución simulada de órdenes
- Backtesting de estrategias con slippage implícito

**Estrategias Implementadas:**

| Estrategia | Lógica | Parámetros |
|-----------|--------|------------|
| Momentum | Seguimiento de tendencia | SMA crossover + RSI |
| Mean Reversion | Retorno a la media | Z-score > ±1.5/2.0 |

```python
# Señal Momentum
if SMA_20 > SMA_50 AND RSI < 70 → BUY
if SMA_20 < SMA_50 OR RSI > 70 → SELL

# Señal Mean Reversion  
if Z-score < -2.0 → BUY (sobrevendido extremo)
if Z-score > 2.0 → SELL (sobrecomprado extremo)
```

### Agente de Portafolio (Portfolio Management Agent)

**Responsabilidades:**
- Construcción de portafolio optimizado (Efficient Frontier)
- Asignación de activos usando Markowitz MPT
- Rebalanceo basado en drift thresholds
- Monitoreo de tracking error

**Metodología - Mean-Variance Optimization:**
```python
# Función Objetivo: Maximizar Sharpe Ratio
max w: (w'μ - r_f) / √(w'Σw)

# Restricciones:
# - Σw_i = 1 (presupuesto)
# - w_i ≥ 0 (long only por defecto)
# - w_i ≤ max_position (límites por posición)
```

**Algoritmo:**
- Simulación Monte Carlo (10,000 portafolios aleatorios)
- Cálculo de frontera eficiente
- Selección de portafolio óptimo por Sharpe Ratio

### Agente de Riesgo (Risk Management Agent)

**Responsabilidades:**
- Cálculo de Value at Risk (VaR) múltiples métodos
- Expected Shortfall (CVaR/ES) para tail risk
- Stress testing con escenarios históricos/sintéticos
- Monitoreo de límites de riesgo
- Métricas risk-adjusted (Sharpe, Sortino, Calmar)

**Métricas Implementadas:**

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| VaR 95% | Percentil 5% retornos | Límite de pérdida diaria |
| CVaR 95% | E[Loss \| Loss > VaR] | Tail risk |
| Sharpe Ratio | (Rp - Rf) / σp | Retorno ajustado por riesgo |
| Sortino Ratio | (Rp - Rf) / σ_downside | Mejor para retornos asimétricos |
| Max Drawdown | max(Peak - Trough) / Peak | Peor pérdida acumulada |
| Calmar Ratio | CAGR / \|Max DD\| | Reward/risk compuesto |

**Métodos VaR:**
```python
# Historical VaR
VaR_hist = percentile(returns, 5%)

# Parametric VaR (Normal distribution)
VaR_param = μ - z_α × σ

# Monte Carlo VaR
VaR_mc = percentile(simulated_returns, 5%)
# Simulación: 10,000 retornos con μ, σ estimados
```

---

## Instalación

### Requisitos

- Python 3.9+
- OpenAI API Key (opcional, para análisis LLM)

### Pasos

```bash
# Clonar o entrar al directorio
cd quant

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env y añadir OPENAI_API_KEY
```

### Dependencias

```
langchain>=0.1.0         # Framework de agentes
streamlit>=1.30.0        # Dashboard UI
pandas>=2.0.0            # Manipulación de datos
numpy>=1.24.0            # Computación numérica
yfinance>=0.2.0          # Datos de mercado
plotly>=5.18.0           # Visualizaciones
scikit-learn>=1.3.0      # Machine learning utilities
scipy>=1.11.0            # Optimización
arch>=6.0.0             # Modelos de volatilidad GARCH
```

---

## Uso

### Dashboard Streamlit (UI Interactiva)

```bash
streamlit run frontend/app.py
```

Accede a `http://localhost:8501` para ver el dashboard con:
- Dashboard principal con overview del sistema
- Análisis de mercado en tiempo real
- Gestión de portafolio con visualización
- Generación de señales y backtesting
- Panel de risk management

### Script de Consola

```bash
python main.py
```

Ejecuta una demostración del sistema mostrando:
- Inicialización de agentes
- Obtención de datos de ejemplo
- Generación de señales
- Análisis de mercado

### Uso Programático

```python
from agents.coordinator_agent import QuantCoordinator
from agents.market_agent import MarketAnalysisAgent
from agents.trading_agent import TradingAgent
import yfinance as yf

# Inicializar agentes
coordinator = QuantCoordinator()
market_agent = MarketAnalysisAgent()
trading_agent = TradingAgent()

# Obtener datos
df = yf.download("AAPL", start="2023-01-01")
df.columns = [c.lower() for c in df.columns]

# Generar señal
signal = trading_agent.generate_momentum_signal(df, "AAPL")
print(f"Signal: {signal.direction} | Strength: {signal.strength:.1%}")

# Análisis de mercado
report = market_agent.generate_market_report("AAPL", "3mo")
print(f"Regime: {report['regime']['regime']}")
```

---

## APIs de los Agentes

### MarketAnalysisAgent

```python
# Análisis de tendencia multi-timeframe
trend = agent.analyze_trend(df)
# Returns: {short: {sma, current, trend, momentum, distance_pct}, ...}

# Detección de régimen de mercado
regime = agent.detect_market_regime(df)
# Returns: {regime, volatility_regime, trend_regime, volatility_20d, ...}

# Análisis de volatilidad
vol = agent.analyze_volatility(df)
# Returns: {annual_volatility, var_95, var_99, skewness, kurtosis, ...}

# Soporte y resistencia
sr = agent.calculate_support_resistance(df)
# Returns: {current_price, resistance_levels[], support_levels[], ...}

# Reporte completo
report = agent.generate_market_report(symbol, period="3mo")
# Returns: {trend_analysis, regime, volatility, support_resistance, summary}
```

### TradingAgent

```python
# Señal de momentum
signal = agent.generate_momentum_signal(df, symbol)
# Returns: Signal(direction, strength, indicators, strategy, timestamp)

# Señal de mean reversion
signal = agent.generate_mean_reversion_signal(df, symbol)
# Returns: Signal con z-score y distancia a media

# Tamaño de posición
position = agent.calculate_position_size(signal, portfolio_value, risk_per_trade=0.02)
# Returns: {quantity, position_value, risk_amount, risk_percentage, stop_loss_distance}

# Backtesting
result = agent.backtest_strategy(df, strategy='momentum', initial_capital=100000)
# Returns: BacktestResult(total_return, sharpe_ratio, max_drawdown, win_rate, ...)
```

### PortfolioAgent

```python
# Crear portafolio
portfolio = agent.create_portfolio("main", "My Portfolio", initial_capital=100000)

# Agregar posición
agent.add_position(portfolio_id, symbol, quantity, price)

# Optimizar portafolio
optimization = agent.optimize_portfolio(returns_df, risk_free_rate=0.04)
# Returns: {optimal_weights, expected_return, expected_volatility, sharpe_ratio, efficient_frontier}

# Señal de rebalanceo
rebalance = agent.generate_rebalancing_signal(portfolio, drift_threshold=0.05)
# Returns: {action: 'rebalance'|'hold', trades[], estimated_cost}
```

### RiskAgent

```python
# Calcular VaR de portafolio
var = agent.calculate_var_portfolio(positions, prices, returns_df, confidence=0.95)
# Returns: {var_95_historical, var_95_parametric, var_95_monte_carlo, cvar_95, component_var}

# Stress testing
stress = agent.stress_test(positions, prices, scenarios, portfolio_value)
# Returns: {worst_case, best_case, all_scenarios[], average_impact}

# Verificar límites de riesgo
limits = agent.check_risk_limits(risk_metrics, positions, prices, portfolio_value)
# Returns: {status: 'OK'|'WARNING'|'VIOLATION', violations[], warnings[], recommendations[]}

# Métricas risk-adjusted
metrics = agent.calculate_risk_adjusted_metrics(returns, risk_free_rate=0.04)
# Returns: {sharpe_ratio, sortino_ratio, calmar_ratio, annual_return, ...}
```

---

## Consideraciones Cuantitativas

### Supuestos del Modelo

1. **Normalidad de Retornos**: Los métodos paramétricos asumen distribución normal. En mercados reales, los retornos exhiben leptokurtosis (fat tails) y skewness.

2. **Estacionaridad**: Los modelos asumen que las estadísticas históricas son predictivas del futuro. En regímenes cambiantes, esto puede fallar.

3. **Liquidez**: El backtesting ignora impacto de mercado. En práctica, órdenes grandes mueven precios.

4. **Costos de Transacción**: No se incluyen en señales generadas (fácil de añadir).

### Limitaciones

- **No es consejo financiero**: El sistema es una herramienta de análisis, no un asesor registrado.
- **Riesgo de overfitting**: Backtesting pasado no garantiza resultados futuros.
- **Dependencia de datos**: La calidad del análisis depende de la calidad/frescura de los datos.
- **Latencia**: Los precios en vivo difieren de los datos históricos usados.

### Mejoras Sugeridas para Producción

```python
# 1. Implementar risk management más sofisticado
from arch import arch_model
garch = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch.fit()

# 2. Añadir multi-factor models
# Fama-French: SMB, HML, MOM factors

# 3. Machine Learning para señales
# LSTM para predicción de precios
# Random Forest para clasificación de régimen

# 4. Execution algorithms reales
# TWAP, VWAP, Implementation Shortfall
```

---

## Estructura de Archivos

```
quant/
├── agents/                     # Agentes cuantitativos
│   ├── __init__.py
│   ├── coordinator_agent.py    # Orquestador multiagente
│   ├── market_agent.py         # Análisis de mercado
│   ├── trading_agent.py        # Señales y backtesting
│   ├── portfolio_agent.py       # Gestión de portafolios
│   └── risk_agent.py           # Gestión de riesgos
│
├── models/                     # Modelos de datos
│   └── data_models.py          # Portfolio, Position, Signal, Order, etc.
│
├── core/                      # Configuración
│   └── config.py              # Parámetros globales, enums
│
├── frontend/                  # Interface de usuario
│   └── app.py                 # Dashboard Streamlit
│
├── utils/                     # Utilidades (futuro)
│
├── main.py                    # Script de demostración
├── requirements.txt           # Dependencias Python
├── .env.example              # Template de configuración
└── README.md                 # Este archivo
```

---

## Disclaimer

**WARNING**: Este software se proporciona "tal cual" sin garantías de ningún tipo. El trading algorítmico conlleva riesgos sustanciales de pérdida financiera. Los resultados pasados (backtesting) no son indicativos de resultados futuros.

No constituye asesoramiento financiero. El usuario es el único responsable de sus decisiones de inversión. Consulta con un asesor financiero registrado antes de tomar cualquier decisión de trading.

---

## Licencia

MIT License - Libre para uso educativo y personal.

---

*Desarrollado como sistema de demostración cuantitativa. Para uso en producción, requiere validación, testing exhaustivo y cumplimiento regulatorio.*
