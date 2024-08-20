import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
import plotly.graph_objects as go
import requests

# Title and sidebar information
st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Harsh Dugad](https://www.linkedin.com/in/harsh-dugad-90067923b)")

# Function to get stock data
@st.cache
def get_stock_data(op, start_date, end_date, interval='1d'):
    try:
        df = yf.download(op, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data found for the selected time frame and date range.")
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# Function to get stock info
@st.cache
def get_stock_info(op):
    try:
        ticker = yf.Ticker(op)
        info = ticker.info
        return info
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

# Function to get market indices data
def get_market_indices():
    indices = {
        'Nifty 50': '^NSEI',
        'Bank Nifty': '^NSEBANK',
        'Sensex': '^BSESN',
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'Nikkei 225': '^N225',
        'Shanghai Composite': '000001.SS',
        'Hang Seng': '^HSI',
        'Straits Times': '^STI'
    }
    
    data = {}
    for name, symbol in indices.items():
        try:
            df = yf.download(symbol, period='1d', interval='1d', progress=False)
            data[name] = df['Close'].iloc[-1]  # Last closing price
        except Exception as e:
            data[name] = 'Error'
    
    return data

# Main function to handle app logic
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Candlestick', 'Financial Info'])
    
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Candlestick':
        candlestick_chart()
    elif option == 'Financial Info':
        financial_info()

# Sidebar input fields
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (in days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End Date', today)

# Dropdown for candlestick time frame
time_frame = st.sidebar.selectbox('Select Time Frame', ['1d', '1wk', '1mo', '1y'])

# Define mapping from time frame to Yahoo Finance interval
interval_map = {
    '1d': '1d',
    '1wk': '1wk',
    '1mo': '1mo',
    '1y': '1y'
}

# Download data and get stock info based on inputs
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
        interval = interval_map[time_frame]
        data = get_stock_data(option, start_date, end_date, interval=interval)
        info = get_stock_info(option)
        if data.empty:
            st.sidebar.error(f'No data found for symbol `{option}` with interval `{interval}`')
        else:
            st.sidebar.success(f'Data successfully retrieved for `{option}` with interval `{interval}`')
    else:
        st.sidebar.error('Error: End date must fall after start date')

# Display real-time market indices
def display_indices():
    st.header('Market Indices')
    indices_data = get_market_indices()
    for name, price in indices_data.items():
        st.write(f"**{name}:** {price if price != 'Error' else 'Error fetching data'}")

# Technical indicators visualization
def tech_indicators():
    st.header('Technical Indicators')
    if not data.empty:
        option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

        # Bollinger bands
        bb_indicator = BollingerBands(data['Close'])
        bb = data.copy()
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        bb = bb[['Close', 'bb_h', 'bb_l']]

        # MACD
        macd = MACD(data['Close']).macd()
        # RSI
        rsi = RSIIndicator(data['Close']).rsi()
        # SMA
        sma = SMAIndicator(data['Close'], window=14).sma_indicator()
        # EMA
        ema = EMAIndicator(data['Close']).ema_indicator()

        if option == 'Close':
            st.write('Close Price')
            st.line_chart(data['Close'])
        elif option == 'BB':
            st.write('Bollinger Bands')
            st.line_chart(bb)
        elif option == 'MACD':
            st.write('MACD')
            st.line_chart(macd)
        elif option == 'RSI':
            st.write('RSI')
            st.line_chart(rsi)
        elif option == 'SMA':
            st.write('SMA')
            st.line_chart(sma)
        else:
            st.write('EMA')
            st.line_chart(ema)
    else:
        st.write('No data available to visualize.')

# Display recent data
def dataframe():
    st.header('Recent Data')
    if not data.empty:
        st.dataframe(data.tail(10))
    else:
        st.write('No data available to display.')

# Candlestick chart visualization
def candlestick_chart():
    st.header('Candlestick Chart')
    if not data.empty:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f'Candlestick chart for {option} ({interval_map.get(time_frame, "1d")})',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
    else:
        st.write('No data available for candlestick chart.')

# Financial Information
def financial_info():
    st.header('Financial Information')
    if info:
        st.write(f"**Market Capitalization:** {format_market_cap(info.get('marketCap', 'N/A'))}")
        st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
        st.write(f"**Dividend Yield:** {format_dividend_yield(info.get('dividendYield', 'N/A'))}")
        st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**Enterprise Value:** {format_market_cap(info.get('enterpriseValue', 'N/A'))}")
    else:
        st.write('No financial information available.')

# Format market capitalization and dividend yield
def format_market_cap(value):
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value / 1e12:.2f} Trillion"
        elif value >= 1e9:
            return f"${value / 1e9:.2f} Billion"
        elif value >= 1e6:
            return f"${value / 1e6:.2f} Million"
        else:
            return f"${value:.2f}"
    return value

def format_dividend_yield(value):
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return value

if __name__ == '__main__':
    display_indices()  # Display market indices
    main()
