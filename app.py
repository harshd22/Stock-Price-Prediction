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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from yahooquery import Ticker

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Harsh Dugad](www.linkedin.com/in/harsh-dugad-90067923b)")

# Global variable
data = pd.DataFrame()

def main():
    global data
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict', 'News'])
    
    # Get user inputs
    stock_symbol = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration', value=3000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End Date', today)
    
    if st.sidebar.button('Send'):
        if start_date < end_date:
            st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
            data = download_data(stock_symbol, start_date, end_date)
            if option == 'Visualize':
                tech_indicators()
            elif option == 'Recent Data':
                dataframe()
            elif option == 'Predict':
                predict()
            elif option == 'News':
                news()
        else:
            st.sidebar.error('Error: End date must fall after start date')

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def tech_indicators():
    global data
    st.header('Technical Indicators')
    if not data.empty:
        option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

        # Bollinger bands
        bb_indicator = BollingerBands(data.Close)
        bb = data
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        # Creating a new dataframe
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
        st.write("No data available. Please fetch data first.")

def dataframe():
    global data
    st.header('Recent Data')
    if not data.empty:
        st.dataframe(data.tail(10))
    else:
        st.write("No data available. Please fetch data first.")

def predict():
    global data
    if not data.empty:
        model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
        num = st.number_input('How many days forecast?', value=5)
        num = int(num)
        if st.button('Predict'):
            if model == 'LinearRegression':
                engine = LinearRegression()
                model_engine(engine, num)
            elif model == 'RandomForestRegressor':
                engine = RandomForestRegressor()
                model_engine(engine, num)
            elif model == 'ExtraTreesRegressor':
                engine = ExtraTreesRegressor()
                model_engine(engine, num)
            elif model == 'KNeighborsRegressor':
                engine = KNeighborsRegressor()
                model_engine(engine, num)
            else:
                engine = XGBRegressor()
                model_engine(engine, num)
    else:
        st.write("No data available. Please fetch data first.")

def model_engine(model, num):
    global data
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

def fetch_news(symbol, start_date, end_date):
    try:
        ticker = Ticker(symbol)
        news_items = ticker.news
        filtered_news = [item for item in news_items if start_date <= datetime.datetime.strptime(item['providerPublishTime'], "%Y-%m-%dT%H:%M:%S.%fZ").date() <= end_date]
        return filtered_news
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def news():
    global data
    st.header('Top News Articles')
    if not data.empty:
        articles = fetch_news(option, start_date, end_date)
        if articles:
            for article in articles:
                st.subheader(article.get('title', 'No Title'))
                st.write(article.get('summary', 'No Summary'))
                st.write(f"[Read more]({article.get('link', '#')})")
                st.write("---")
        else:
            st.write("No news articles found.")
    else:
        st.write('No data available to fetch news.')

if __name__ == '__main__':
    main()
