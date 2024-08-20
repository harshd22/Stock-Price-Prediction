import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

# App Title and Sidebar
st.title('Stock Price Predictions & Analysis')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Vikas Sharma](https://www.linkedin.com/in/vikas-sharma005/)")

# Main function for the app
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict', 'Educational Resources', 'Financial Data'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Educational Resources':
        educational_resources()
    elif option == 'Financial Data':
        financial_data()
    else:
        predict()

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

# Sidebar inputs
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\n\nEnd date: {end_date}')
        data = download_data(option, start_date, end_date)
        if data.empty:
            st.sidebar.error(f'No data found for symbol `{option}`')
        else:
            st.sidebar.success(f'Data successfully retrieved for `{option}`')
    else:
        st.sidebar.error('Error: End date must fall after start date')

scaler = StandardScaler()

# Technical Indicators Visualization
def tech_indicators():
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
        
        # Candlestick chart using Plotly
        st.sidebar.header("Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
        fig.update_layout(title=f'Candlestick Chart for {option}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    else:
        st.write('No data available to visualize.')

# Recent Data Display
def dataframe():
    st.header('Recent Data')
    if not data.empty:
        st.dataframe(data.tail(10))
    else:
        st.write('No data available to display.')

# Stock Price Prediction
def predict():
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
        st.write('No data available to make predictions.')

def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # splitting the data
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

# Educational Resources Section
def educational_resources():
    st.sidebar.header("Educational Resources")

    st.write("## Stock Market Basics")
    st.write("""
    - **Stocks**: Shares of a company that represent ownership.
    - **Bonds**: Debt instruments issued by companies/governments to raise capital.
    - **Technical Indicators**: Tools used by traders to analyze stock price movements.
    - **Fundamental Analysis**: Evaluating a stock by examining financial statements.
    """)

    st.write("### Useful Resources")
    st.markdown("[Investopedia](https://www.investopedia.com) - A comprehensive resource for financial education.")
    st.markdown("[Yahoo Finance](https://finance.yahoo.com) - Get the latest financial news and data.")

# Financial Data & Ratios Section
def financial_data():
    st.sidebar.header("Financial Data & Ratios")
    ticker = yf.Ticker(option)
    stock_info = ticker.info

    st.write(f"## {option} Financial Data")
    st.write(f"**Market Cap**: {stock_info.get('marketCap', 'N/A')}")
    st.write(f"**P/E Ratio**: {stock_info.get('trailingPE', 'N/A')}")
    st.write(f"**Price-to-Sales**: {stock_info.get('priceToSalesTrailing12Months', 'N/A')}")
    st.write(f"**Dividend Yield**: {stock_info.get('dividendYield', 'N/A')}")
    st.write(f"**Debt-to-Equity Ratio**: {stock_info.get('debtToEquity', 'N/A')}")

    st.write("### Explanation of Ratios")
    st.write("""
    - **P/E Ratio**: Price-to-Earnings ratio. A higher ratio might indicate that a stock is overvalued.
    - **Debt-to-Equity**: Shows how much debt the company has compared to equity. A higher ratio means more debt.
    - **Dividend Yield**: Indicates how much a company returns to shareholders in dividends.
    """)

if __name__ == '__main__':
    main()
