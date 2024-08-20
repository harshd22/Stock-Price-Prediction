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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import catboost
from fbprophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

# Title and sidebar information
st.title('Stock Price Predictions')
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

# Function to display dynamic header with stock price and change
def display_stock_header(stock_data):
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        previous_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        price_change = current_price - previous_close
        percent_change = (price_change / previous_close) * 100 if previous_close != 0 else 0
        st.markdown(f"### Current Stock Price: **${current_price:.2f}**")
        st.markdown(f"### Change: **${price_change:.2f}** ({percent_change:.2f}%)")
    else:
        st.markdown("### No data available.")

# Main function to handle app logic
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Candlestick', 'Financial Info', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Candlestick':
        candlestick_chart()
    elif option == 'Financial Info':
        financial_info()
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
        display_stock_header(data)  # Display header with stock price and change
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
        st.write(f"**Market Capitalization:** {format_market_cap(info.get('marketCap', 0))}")
        st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
        st.write(f"**Dividend Yield:** {format_dividend_yield(info.get('dividendYield', 'N/A'))}")
        st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**Enterprise Value:** {format_market_cap(info.get('enterpriseValue', 0))}")
    else:
        st.write('No financial information available.')

# Format Market Cap and Dividend Yield
def format_market_cap(value):
    if value == 0:
        return 'No information available'
    elif value >= 1e12:
        return f"${value / 1e12:.2f} Trillion"
    elif value >= 1e9:
        return f"${value / 1e9:.2f} Billion"
    elif value >= 1e6:
        return f"${value / 1e6:.2f} Million"
    else:
        return f"${value:.2f}"

def format_dividend_yield(value):
    if value == 'N/A':
        return 'No information available'
    return f"{value * 100:.2f}%"

# Prediction function
def predict():
    if not data.empty:
        model = st.radio('Choose a model', ['LinearRegression', 'Lasso', 'Ridge', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'SVR', 'DecisionTreeRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'LightGBM', 'CatBoost', 'Prophet'])
        num = st.number_input('How many days forecast?', value=5)
        num = int(num)
        if st.button('Predict'):
            if model == 'LinearRegression':
                engine = LinearRegression()
                model_engine(engine, num)
            elif model == 'Lasso':
                engine = Lasso()
                model_engine(engine, num)
            elif model == 'Ridge':
                engine = Ridge()
                model_engine(engine, num)
            elif model == 'RandomForestRegressor':
                engine = RandomForestRegressor()
                model_engine(engine, num)
            elif model == 'ExtraTreesRegressor':
                engine = ExtraTreesRegressor()
                model_engine(engine, num)
            elif model == 'GradientBoostingRegressor':
                engine = GradientBoostingRegressor()
                model_engine(engine, num)
            elif model == 'AdaBoostRegressor':
                engine = AdaBoostRegressor()
                model_engine(engine, num)
            elif model == 'SVR':
                engine = SVR()
                model_engine(engine, num)
            elif model == 'DecisionTreeRegressor':
                engine = DecisionTreeRegressor()
                model_engine(engine, num)
            elif model == 'KNeighborsRegressor':
                engine = KNeighborsRegressor()
                model_engine(engine, num)
            elif model == 'XGBoostRegressor':
                engine = XGBRegressor()
                model_engine(engine, num)
            elif model == 'LightGBM':
                engine = lgb.LGBMRegressor()
                model_engine(engine, num)
            elif model == 'CatBoost':
                engine = catboost.CatBoostRegressor(silent=True)
                model_engine(engine, num)
            elif model == 'Prophet':
                prophet_forecast(num)
    else:
        st.write('No data available to make predictions.')

# Model engine for predictions
def model_engine(model, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

# Prophet forecasting function
def prophet_forecast(num):
    df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=num)
    forecast = model.predict(future)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(num))

if __name__ == '__main__':
    main()
