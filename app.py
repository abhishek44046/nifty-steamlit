import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time

# Set up page styling
st.set_page_config(page_title="Live Nifty Monitor", page_icon="📈", layout="centered")

def fetch_nifty_data():
    data = yf.download("^NSEI", period="1d", interval="1m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def build_algorithms_matrix(df):
    signals = {}
    close, high, low = df['Close'], df['High'], df['Low']
    
    signals['1. EMA Crossover (9/21)'] = np.where(ta.trend.ema_indicator(close, 9) > ta.trend.ema_indicator(close, 21), 1, -1)[-1]
    signals['2. SMA Golden/Death Cross'] = np.where(ta.trend.sma_indicator(close, 50) > ta.trend.sma_indicator(close, 200), 1, -1)[-1]
    
    rsi_val = ta.momentum.rsi(close, 14).iloc[-1]
    signals['3. RSI Momentum'] = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
    signals['4. MACD Crossover'] = np.where(ta.trend.macd(close) > ta.trend.macd_signal(close), 1, -1)[-1]
    signals['5. Bollinger Bands Position'] = 1 if close.iloc[-1] > ((ta.volatility.bollinger_hband(close) + ta.volatility.bollinger_lband(close)) / 2).iloc[-1] else -1
    signals['6. ADX Direction (+DI/-DI)'] = np.where(ta.trend.adx_pos(high, low, close) > ta.trend.adx_neg(high, low, close), 1, -1)[-1]
    signals['7. Stochastic Oscillator'] = np.where(ta.momentum.stoch(high, low, close) > ta.momentum.stoch_signal(high, low, close), 1, -1)[-1]
    signals['8. Rate of Change (ROC)'] = np.where(ta.momentum.roc(close, 12) > 0, 1, -1)[-1]
    signals['9. Commodity Channel Index'] = 1 if ta.trend.cci(high, low, close, 20).iloc[-1] > 0 else -1
    signals['10. Chaikin Money Flow'] = 1 if ta.volume.chaikin_money_flow(high, low, close, df['Volume']).iloc[-1] > 0 else -1
    signals['11. Parabolic SAR'] = 1 if np.isnan(ta.trend.psar_down(high, low, close).iloc[-1]) else -1
    
    return signals

# Create a permanent visual placeholder that overrides itself on every cycle
placeholder = st.empty()

while True:
    with placeholder.container():
        df = fetch_nifty_data()
        if not df.empty and len(df) >= 2:
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100
            
            st.title("📈 Nifty 50 Multi-Algo Monitor")
            st.metric(label="NIFTY 50 Close Index", value=f"{current_price:.2f}", delta=f"{change:+.2f} ({pct_change:+.2f}%)")
            st.markdown("---")
            
            algo_results = build_algorithms_matrix(df)
            score = sum(1 for v in algo_results.values() if v == 1) - sum(1 for v in algo_results.values() if v == -1)
            
            # Display signals in a neat, clean structured data table layout
            report_data = []
            for algo, sig in algo_results.items():
                status = "🟢 BULLISH" if sig == 1 else ("🔴 BEARISH" if sig == -1 else "⚪ NEUTRAL")
                report_data.append({"Algorithm Engine": algo, "Market Signal Direction": status})
            
            st.table(pd.DataFrame(report_data))
            
            # Dynamic Score Consensus
            normalized = (score / len(algo_results)) * 100
            if normalized >= 50: consensus = "🔥 STRONG BULLISH"
            elif 15 <= normalized < 50: consensus = "📈 BULLISH"
            elif -15 < normalized < 15: consensus = "⚖️ NEUTRAL / SIDEWAYS"
            elif -50 < normalized <= -15: consensus = "📉 BEARISH"
            else: consensus = "❄️ STRONG BEARISH"
            
            st.info(f"**Consensus Trend Evaluation Matrix Bias:** {consensus} (Net Score: {score:+})")
            
        else:
            st.warning("Fetching real-time streams from market data channels...")
            
    time.sleep(5)