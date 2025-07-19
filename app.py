import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Title and sidebar information
st.title('Dynamic Stock Price Prediction and Financial Insights Platform')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Harsh Dugad](https://www.linkedin.com/in/harsh-dugad-90067923b/)")

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

# Get news headlines for the stock
def get_stock_news(op):
    try:
        ticker = yf.Ticker(op)
        news = ticker.news
        if news:
            return news[:5]  # Return top 5 news
        else:
            return []
    except Exception as e:
        return []

# Main function to handle app logic
def main():
    option = st.sidebar.selectbox('Make a choice', ['Recent Data', 'Line Chart', 'Buy/Sell Recommendation', 'Financial Info', 'News', 'Predict'])
    if option == 'Recent Data':
        dataframe()
    elif option == 'Line Chart':
        line_chart()
    elif option == 'Buy/Sell Recommendation':
        buy_sell_recommendation()
    elif option == 'Financial Info':
        financial_info()
    elif option == 'News':
        news_section()
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

# Dropdown for time frame (not used for line chart, but kept for compatibility)
time_frame = st.sidebar.selectbox('Select Time Frame', ['1d', '1wk', '1mo', '1y'])
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
            st.sidebar.error(f'No data found for symbol {option} with interval {interval}')
        else:
            st.sidebar.success(f'Data successfully retrieved for {option} with interval {interval}')
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = get_stock_data(option, start_date, end_date, interval=interval_map.get(time_frame, '1d'))
info = get_stock_info(option)
scaler = StandardScaler()

# Display recent data
def dataframe():
    st.header('Recent Data')
    if not data.empty:
        st.dataframe(data.tail(10))
    else:
        st.write('No data available to display.')

# Line chart visualization
def line_chart():
    st.header('Line Chart of Close Price')
    if not data.empty:
        st.line_chart(data['Close'])
    else:
        st.write('No data available for line chart.')

# Buy/Sell Recommendation based on recent price movement (no indicators)
def buy_sell_recommendation():
    st.header('Buy/Sell/Hold Recommendation')
    if not data.empty and len(data) > 5:
        # Simple logic: if last close > mean of last 5 closes by 1%, recommend Buy; if < by 1%, Sell; else Hold
        last_close = data['Close'].iloc[-1]
        mean_last5 = data['Close'].iloc[-6:-1].mean()
        if last_close > mean_last5 * 1.01:
            st.success('Recommendation: BUY')
        elif last_close < mean_last5 * 0.99:
            st.error('Recommendation: SELL')
        else:
            st.info('Recommendation: HOLD')
        st.write(f"Last Close: {last_close:.2f}")
        st.write(f"Mean of Previous 5 Closes: {mean_last5:.2f}")
    else:
        st.write('Not enough data for recommendation.')

# Financial Information
def financial_info():
    st.header('Financial Information')
    if info:
        def format_value(value):
            if value is None:
                return 'N/A'
            if value >= 1e12:
                return f"{value / 1e12:.2f} Trillion"
            elif value >= 1e9:
                return f"{value / 1e9:.2f} Billion"
            elif value >= 1e6:
                return f"{value / 1e6:.2f} Million"
            else:
                return f"{value:.2f}"
        st.write(f"**Market Capitalization:** {format_value(info.get('marketCap', None))}")
        st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
        dividend_yield = info.get('dividendYield', None)
        if dividend_yield is not None:
            st.write(f"**Dividend Yield:** {dividend_yield * 100:.2f}%")
        else:
            st.write("**Dividend Yield:** No information available")
        st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**Enterprise Value:** {format_value(info.get('enterpriseValue', None))}")
    else:
        st.write('No financial information available.')

# News Section
def news_section():
    st.header('Recent News Headlines')
    news = get_stock_news(option)
    if news:
        for item in news:
            st.markdown(f"- [{item.get('title', 'No Title')}]({item.get('link', '#')})")
    else:
        st.write('No news available for this stock.')

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
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"**R-squared Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    forecast = model.predict(x_forecast)
    forecast_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, num + 1)]
    forecast_df = pd.DataFrame(data={'Date': forecast_dates, 'Forecast': forecast})
    st.write(forecast_df)

# Run the app
if __name__ == "__main__":
    main() 
