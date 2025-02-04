import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Function to fetch stock data
def get_stock_data(ticker, period, interval):
    """
    Fetches historical stock data from Yahoo Finance.

    :param ticker: Stock symbol (e.g., 'AAPL')
    :param period: Data period (e.g., '1mo', '3mo', '6mo', '1y')
    :param interval: Timeframe interval (e.g., '1d', '1h', '5m')
    :return: DataFrame with stock data
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            st.error("⚠️ No data found for the given ticker and timeframe.")
        return df
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to compute technical indicators
def compute_technical_indicators(df):
    """
    Computes Bollinger Bands, EMA, MACD, and RSI.

    :param df: DataFrame containing stock price data
    :return: DataFrame with technical indicators
    """
    try:
        if 'Close' not in df.columns:
            st.error("⚠️ 'Close' column not found in data.")
            return df

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        if len(df) < 20:
            st.error("⚠️ Not enough data points for indicator calculation (Minimum: 20).")
            return df

        # Bollinger Bands
        bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()

        # Exponential Moving Averages (EMA)
        df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
        df['EMA_15'] = EMAIndicator(close=df['Close'], window=15).ema_indicator()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()

        # RSI (Relative Strength Index)
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

        return df

    except Exception as e:
        st.error(f"⚠️ Error computing indicators: {e}")
        return df

# Function to plot candlestick chart
def plot_candlestick(df, ticker):
    """
    Plots a candlestick chart with Bollinger Bands and EMAs.

    :param df: DataFrame containing stock price data
    :param ticker: Stock ticker symbol
    """
    if df.empty or 'Close' not in df.columns:
        st.error("⚠️ No data available to plot candlestick chart.")
        return

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='blue')))

    # EMA
    if 'EMA_9' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], mode='lines', name='EMA 9', line=dict(color='orange')))
    if 'EMA_15' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_15'], mode='lines', name='EMA 15', line=dict(color='red')))

    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# Streamlit App
def main():
    st.title("📈 Stock Price Prediction with Candlestick Chart & Indicators")

    # Sidebar Inputs
    st.sidebar.header("🔍 Stock Data Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()

    period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    period = st.sidebar.selectbox("Select Data Period", period_options, index=4)

    interval_options = ["1m", "5m", "15m", "1h", "1d", "1wk"]
    interval = st.sidebar.selectbox("Select Timeframe Interval", interval_options, index=4)

    if st.sidebar.button("📊 Fetch Data"):
        df = get_stock_data(ticker, period, interval)

        if not df.empty:
            st.subheader("📜 Raw Stock Data")
            st.write(df.tail())

            df = compute_technical_indicators(df)

            if not df.empty:
                st.subheader("📊 Candlestick Chart")
                plot_candlestick(df, ticker)

                st.subheader("📈 Technical Indicators")
                st.write(df.tail())

                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Processed Data", csv, f"{ticker}_data.csv", "text/csv")

if __name__ == "__main__":
    main()
