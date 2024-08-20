import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date

# Title and sidebar information
st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Harsh Dugad](www.linkedin.com/in/harsh-dugad-90067923b)")

# List of market indices
INDEX_LIST = [
    '^GSPC',  # S&P 500 (USA)
    '^DJI',   # Dow Jones (USA)
    '^IXIC',  # NASDAQ (USA)
    '^RUT',   # Russell 2000 (USA)
    '^N225',  # Nikkei 225 (Japan)
    '000001.SS',  # Shanghai Composite (China)
    '000300.SS',  # CSI 300 (China)
    '^FTSE',  # FTSE 100 (UK)
    '^DAX',   # DAX (Germany)
    '^CAC',   # CAC 40 (France)
    '^NSEI',  # Nifty 50 (India)
    '^BSESN', # Sensex (India)
    '^BANKNIFTY',  # Nifty Bank (India)
    '^STI',   # Straits Times Index (Singapore)
]

@st.cache_resource
def get_stock_data(symbol, start_date, end_date, interval='1d'):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data found for the selected time frame and date range.")
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

@st.cache_resource
def get_real_time_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if data.empty:
            raise ValueError("No data found.")
        current_price = data['Close'].iloc[-1]
        return current_price
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to calculate percentage change
def calculate_percentage_change(symbol):
    try:
        data = get_stock_data(symbol, datetime.date.today() - datetime.timedelta(days=1), datetime.date.today(), interval='1d')
        if len(data) < 2:
            return 0
        current_price = data['Close'].iloc[-1]
        previous_close = data['Close'].iloc[-2]
        return (current_price - previous_close) / previous_close * 100
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

# Function to display dynamic header with real-time indices data
def display_index_header(index_list):
    st.header('Market Indices')
    changes = []
    
    for index in index_list:
        current_price = get_real_time_data(index)
        if current_price is not None:
            change = calculate_percentage_change(index)
            changes.append((index, current_price, change))
    
    if changes:
        sorted_changes = sorted(changes, key=lambda x: x[2], reverse=True)
        top_gainer = sorted_changes[0]
        top_loser = sorted_changes[-1]
        
        st.markdown(f"### Top Gainer: **{top_gainer[0]}** - ${top_gainer[1]:,.2f} ({top_gainer[2]:.2f}%)")
        st.markdown(f"### Top Loser: **{top_loser[0]}** - ${top_loser[1]:,.2f} ({top_loser[2]:.2f}%)")
    else:
        st.markdown("### No data available for top gainers and losers.")

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

# Display dynamic header for indices
if st.sidebar.button('Show Top Gainers and Losers'):
    display_index_header(INDEX_LIST)

# Download the data globally for access in different functions
data = get_stock_data(option, start_date, end_date, interval=interval_map.get(time_frame, '1d'))

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
        st.write(f"**Market Capitalization:** {format_number(info.get('marketCap', 'N/A'))}")
        st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
        st.write(f"**Dividend Yield:** {format_dividend_yield(info.get('dividendYield', 'N/A'))}")
        st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**Enterprise Value:** {format_number(info.get('enterpriseValue', 'N/A'))}")
    else:
        st.write('No financial information available.')

# Format number function for market capitalization
def format_number(number):
    if number == 'N/A':
        return 'N/A'
    number = float(number)  # Ensure number is float
    if number >= 1e12:
        return f"${number / 1e12:.2f} Trillion"
    elif number >= 1e9:
        return f"${number / 1e9:.2f} Billion"
    elif number >= 1e6:
        return f"${number / 1e6:.2f} Million"
    else:
        return f"${number:.2f}"

# Format dividend yield function
def format_dividend_yield(yield_value):
    if yield_value == 'N/A':
        return 'N/A'
    return f"{float(yield_value) * 100:.2f}%"

# Prediction function
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

# Model engine for predictions
def model_engine(model, num):
    df = data[['Open', 'High', 'Low', 'Close']].copy()
    df['preds'] = df['Close'].shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df['preds'].values
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

if __name__ == '__main__':
    main()
