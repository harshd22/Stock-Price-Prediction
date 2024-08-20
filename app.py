import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Initialize News API (replace 'YOUR_NEWSAPI_KEY' with your actual API key)
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')

# Title and sidebar information
st.title('Stock Price Predictions with News and Sentiment Analysis')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Harsh Dugad](www.linkedin.com/in/harsh-dugad-90067923b)")

# Function to get stock data and financials
@st.cache_resource
def get_stock_data(op, start_date, end_date, interval='1d'):
    try:
        df = yf.download(op, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data found for the selected time frame and date range.")
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

@st.cache_resource
def get_stock_info(op):
    try:
        ticker = yf.Ticker(op)
        info = ticker.info
        return info
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

# Function to fetch news articles
@st.cache_resource
def fetch_news(stock_symbol):
    try:
        news = newsapi.get_everything(q=stock_symbol, language='en', sort_by='relevancy', page_size=5)
        return news['articles']
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Return compound sentiment score

# Main function to handle app logic
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Candlestick', 'Financial Info', 'Predict', 'News & Sentiment'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Candlestick':
        candlestick_chart()
    elif option == 'Financial Info':
        financial_info()
    elif option == 'News & Sentiment':
        news_sentiment_analysis()
    else:
        predict()

# Sidebar input fields
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (in days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

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

# Download the data globally for access in different functions
data = get_stock_data(option, start_date, end_date, interval=interval_map.get(time_frame, '1d'))
info = get_stock_info(option)
scaler = StandardScaler()

# Technical indicators visualization
def tech_indicators():
    st.header('Technical Indicators')
    if not data.empty:
        option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

        # Bollinger bands
        bb_indicator = BollingerBands(data.Close)
        bb = data.copy()
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        bb = bb[['Close', 'bb_h', 'bb_l']]

        # MACD
        macd = MACD(data.Close).macd()
        # RSI
        rsi = RSIIndicator(data.Close).rsi()
        # SMA
        sma = SMAIndicator(data.Close, window=14).sma_indicator()
        # EMA
        ema = EMAIndicator(data.Close).ema_indicator()

        if option == 'Close':
            st.write('Close Price')
            st.line_chart(data.Close)
        elif option == 'BB':
            st.write('BollingerBands')
            st.line_chart(bb)
        elif option == 'MACD':
            st.write('Moving Average Convergence Divergence')
            st.line_chart(macd)
        elif option == 'RSI':
            st.write('Relative Strength Indicator')
            st.line_chart(rsi)
        elif option == 'SMA':
            st.write('Simple Moving Average')
            st.line_chart(sma)
        else:
            st.write('Exponential Moving Average')
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
        market_cap = info.get('marketCap', None)
        if market_cap:
            if market_cap >= 1e12:
                market_cap_str = f'{market_cap / 1e12:.2f} Trillion'
            elif market_cap >= 1e9:
                market_cap_str = f'{market_cap / 1e9:.2f} Billion'
            elif market_cap >= 1e6:
                market_cap_str = f'{market_cap / 1e6:.2f} Million'
            else:
                market_cap_str = f'{market_cap:,}'
        else:
            market_cap_str = 'No information provided'

        st.write(f"**Market Capitalization:** {market_cap_str}")
        st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'No information provided')}")
        st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'No information provided')}")
        st.write(f"**Dividend Yield:** {info.get('dividendYield', 'No information provided')}")
        st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'No information provided')}")
        st.write(f"**Enterprise Value:** {info.get('enterpriseValue', 'No information provided')}")
    else:
        st.write('No financial information available.')

# News and Sentiment Analysis
def news_sentiment_analysis():
    st.header('News and Sentiment Analysis')
    news = fetch_news(option)
    if news:
        analyzer = SentimentIntensityAnalyzer()
        for article in news:
            title = article['title']
            description = article['description']
            url = article['url']
            st.subheader(title)
            st.write(description)
            st.write(f"[Read more]({url})")

            # Sentiment analysis
            sentiment_score = analyze_sentiment(description)
            if sentiment_score > 0.05:
                sentiment = 'Positive'
            elif sentiment_score < -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            st.write(f"Sentiment: **{sentiment}** (Score: {sentiment_score:.2f})")
            st.write("---")
    else:
        st.write("No news available.")

# Prediction function placeholder (You can add your own model predictions here)
def predict():
    st.header('Prediction')
    st.write('Prediction logic here.')

if __name__ == "__main__":
    main()
