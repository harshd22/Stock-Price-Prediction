import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressorimport streamlit as st
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

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Prediction & Insights", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #181c20;}
    .block-container {padding-top: 2rem;}
    .stButton>button {background-color: #1a73e8; color: white;}
    .stDataFrame {background-color: #23272f;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#1a73e8;'>📈 Dynamic Stock Price Prediction & Financial Insights</h1>", unsafe_allow_html=True)
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by <a href='https://www.linkedin.com/in/harsh-dugad-90067923b/' target='_blank'>Harsh Dugad</a>", unsafe_allow_html=True)

# --- Data Fetching Functions ---
@st.cache_resource
def get_stock_data(op, start_date, end_date, interval='1d'):
    try:
        df = yf.download(op, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data found for the selected time frame and date range.")
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_stock_info(op):
    try:
        ticker = yf.Ticker(op)
        info = ticker.info
        return info
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

def get_stock_news(op):
    try:
        ticker = yf.Ticker(op)
        news = ticker.news
        if news:
            return news[:5]
        else:
            return []
    except Exception as e:
        return []

# --- Sidebar Inputs ---
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (in days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
time_frame = st.sidebar.selectbox('Select Time Frame', ['1d', '1wk', '1mo', '1y'])
interval_map = {'1d': '1d', '1wk': '1wk', '1mo': '1mo', '1y': '1y'}

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

# --- App Sections ---
def dataframe():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🗂️ Recent Data</h3>", unsafe_allow_html=True)
    if not data.empty:
        st.dataframe(data.tail(10), use_container_width=True)
    else:
        st.write('No data available to display.')

def line_chart():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>📊 Line Chart of Close Price</h3>", unsafe_allow_html=True)
    if not data.empty:
        st.line_chart(data['Close'])
    else:
        st.write('No data available for line chart.')

def buy_sell_recommendation():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>💡 Buy/Sell/Hold Recommendation</h3>", unsafe_allow_html=True)
    if not data.empty and len(data['Close'].dropna()) > 6:
        closes = data['Close'].dropna()
        last_close = float(closes.iloc[-1])
        mean_last5 = float(closes.iloc[-6:-1].mean())
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Last Close", f"{last_close:.2f}")
        with col2:
            st.metric("Mean of Previous 5 Closes", f"{mean_last5:.2f}")
        if last_close > mean_last5 * 1.01:
            st.success('Recommendation: BUY 🚀')
        elif last_close < mean_last5 * 0.99:
            st.error('Recommendation: SELL ⚠️')
        else:
            st.info('Recommendation: HOLD 🤝')
    else:
        st.write('Not enough data for recommendation.')

def volatility_meter():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🌪️ Volatility Meter (Last 30 Days)</h3>", unsafe_allow_html=True)
    if not data.empty and len(data) > 30:
        returns = data['Close'].pct_change().dropna()
        vol = returns[-30:].std() * 100
        st.metric("Volatility (std dev of daily returns)", f"{vol:.2f}%")
        if vol > 3:
            st.warning("High volatility!")
        elif vol > 1.5:
            st.info("Moderate volatility.")
        else:
            st.success("Low volatility.")
    else:
        st.write('Not enough data for volatility analysis.')

def best_worst_day():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🏆 Best & Worst Day (Close Price Change)</h3>", unsafe_allow_html=True)
    if not data.empty and len(data) > 1:
        returns = data['Close'].pct_change().dropna()
        best = returns.idxmax()
        worst = returns.idxmin()
        st.write(f"Best day: <b>{best.date()}</b> ({returns.max()*100:.2f}%)", unsafe_allow_html=True)
        st.write(f"Worst day: <b>{worst.date()}</b> ({returns.min()*100:.2f}%)", unsafe_allow_html=True)
    else:
        st.write('Not enough data for best/worst day analysis.')

def financial_info():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>💰 Financial Information</h3>", unsafe_allow_html=True)
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Market Cap:** {format_value(info.get('marketCap', None))}")
            st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        with col2:
            st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
        with col3:
            st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
            st.write(f"**Enterprise Value:** {format_value(info.get('enterpriseValue', None))}")
    else:
        st.write('No financial information available.')

def news_section():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>📰 Recent News Headlines</h3>", unsafe_allow_html=True)
    if st.button("Refresh News"):
        st.session_state['news'] = get_stock_news(option)
    news = st.session_state.get('news', get_stock_news(option))
    if news:
        for item in news:
            title = item.get('title') or item.get('providerPublishTime') or 'No Title'
            link = item.get('link', '#')
            st.markdown(f"- [{title}]({link})")
    else:
        st.write('No news available for this stock.')

def predict():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🤖 Price Prediction</h3>", unsafe_allow_html=True)
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
    st.metric("R-squared Score", f"{r2:.2f}")
    st.metric("Mean Absolute Error", f"{mae:.2f}")
    forecast = model.predict(x_forecast)
    forecast_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, num + 1)]
    forecast_df = pd.DataFrame(data={'Date': forecast_dates, 'Forecast': forecast})
    st.dataframe(forecast_df, use_container_width=True)

# --- Main App Logic ---
option_menu = [
    'Recent Data', 'Line Chart', 'Buy/Sell Recommendation', 'Volatility Meter', 'Best/Worst Day', 'Financial Info', 'News', 'Predict'
]
selected = st.sidebar.selectbox('Choose Section', option_menu)
if selected == 'Recent Data':
    dataframe()
elif selected == 'Line Chart':
    line_chart()
elif selected == 'Buy/Sell Recommendation':
    buy_sell_recommendation()
elif selected == 'Volatility Meter':
    volatility_meter()
elif selected == 'Best/Worst Day':
    best_worst_day()
elif selected == 'Financial Info':
    financial_info()
elif selected == 'News':
    news_section()
else:
    predict()

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ by <a href='https://www.linkedin.com/in/harsh-dugad-90067923b/' target='_blank'>Harsh Dugad</a></div>", unsafe_allow_html=True) 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Prediction & Insights", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #181c20;}
    .block-container {padding-top: 2rem;}
    .stButton>button {background-color: #1a73e8; color: white;}
    .stDataFrame {background-color: #23272f;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#1a73e8;'>📈 Dynamic Stock Price Prediction & Financial Insights</h1>", unsafe_allow_html=True)
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by <a href='https://www.linkedin.com/in/harsh-dugad-90067923b/' target='_blank'>Harsh Dugad</a>", unsafe_allow_html=True)

# --- Data Fetching Functions ---
@st.cache_resource
def get_stock_data(op, start_date, end_date, interval='1d'):
    try:
        df = yf.download(op, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data found for the selected time frame and date range.")
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_stock_info(op):
    try:
        ticker = yf.Ticker(op)
        info = ticker.info
        return info
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

def get_stock_news(op):
    try:
        ticker = yf.Ticker(op)
        news = ticker.news
        if news:
            return news[:5]
        else:
            return []
    except Exception as e:
        return []

# --- Sidebar Inputs ---
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (in days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
time_frame = st.sidebar.selectbox('Select Time Frame', ['1d', '1wk', '1mo', '1y'])
interval_map = {'1d': '1d', '1wk': '1wk', '1mo': '1mo', '1y': '1y'}

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

# --- App Sections ---
def dataframe():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🗂️ Recent Data</h3>", unsafe_allow_html=True)
    if not data.empty:
        st.dataframe(data.tail(10), use_container_width=True)
    else:
        st.write('No data available to display.')

def line_chart():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>📊 Line Chart of Close Price</h3>", unsafe_allow_html=True)
    if not data.empty:
        st.line_chart(data['Close'])
    else:
        st.write('No data available for line chart.')

def buy_sell_recommendation():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>💡 Buy/Sell/Hold Recommendation</h3>", unsafe_allow_html=True)
    if not data.empty and len(data) > 6:
        last_close = float(data['Close'].iloc[-1])
        mean_last5 = float(data['Close'].iloc[-6:-1].mean())
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Last Close", f"{last_close:.2f}")
        with col2:
            st.metric("Mean of Previous 5 Closes", f"{mean_last5:.2f}")
        if last_close > mean_last5 * 1.01:
            st.success('Recommendation: BUY 🚀')
        elif last_close < mean_last5 * 0.99:
            st.error('Recommendation: SELL ⚠️')
        else:
            st.info('Recommendation: HOLD 🤝')
    else:
        st.write('Not enough data for recommendation.')

def volatility_meter():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🌪️ Volatility Meter (Last 30 Days)</h3>", unsafe_allow_html=True)
    if not data.empty and len(data) > 30:
        returns = data['Close'].pct_change().dropna()
        vol = returns[-30:].std() * 100
        st.metric("Volatility (std dev of daily returns)", f"{vol:.2f}%")
        if vol > 3:
            st.warning("High volatility!")
        elif vol > 1.5:
            st.info("Moderate volatility.")
        else:
            st.success("Low volatility.")
    else:
        st.write('Not enough data for volatility analysis.')

def best_worst_day():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🏆 Best & Worst Day (Close Price Change)</h3>", unsafe_allow_html=True)
    if not data.empty and len(data) > 1:
        returns = data['Close'].pct_change().dropna()
        best = returns.idxmax()
        worst = returns.idxmin()
        st.write(f"Best day: <b>{best.date()}</b> ({returns.max()*100:.2f}%)", unsafe_allow_html=True)
        st.write(f"Worst day: <b>{worst.date()}</b> ({returns.min()*100:.2f}%)", unsafe_allow_html=True)
    else:
        st.write('Not enough data for best/worst day analysis.')

def financial_info():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>💰 Financial Information</h3>", unsafe_allow_html=True)
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Market Cap:** {format_value(info.get('marketCap', None))}")
            st.write(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        with col2:
            st.write(f"**Price to Book Ratio:** {info.get('priceToBook', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
        with col3:
            st.write(f"**Forward PE Ratio:** {info.get('forwardPE', 'N/A')}")
            st.write(f"**Enterprise Value:** {format_value(info.get('enterpriseValue', None))}")
    else:
        st.write('No financial information available.')

def news_section():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>📰 Recent News Headlines</h3>", unsafe_allow_html=True)
    news = get_stock_news(option)
    if news:
        for item in news:
            title = item.get('title') or item.get('providerPublishTime') or 'No Title'
            link = item.get('link', '#')
            st.markdown(f"- [{title}]({link})")
    else:
        st.write('No news available for this stock.')

def predict():
    st.markdown("---")
    st.markdown("<h3 style='color:#1a73e8;'>🤖 Price Prediction</h3>", unsafe_allow_html=True)
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
    st.metric("R-squared Score", f"{r2:.2f}")
    st.metric("Mean Absolute Error", f"{mae:.2f}")
    forecast = model.predict(x_forecast)
    forecast_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, num + 1)]
    forecast_df = pd.DataFrame(data={'Date': forecast_dates, 'Forecast': forecast})
    st.dataframe(forecast_df, use_container_width=True)

# --- Main App Logic ---
option_menu = [
    'Recent Data', 'Line Chart', 'Buy/Sell Recommendation', 'Volatility Meter', 'Best/Worst Day', 'Financial Info', 'News', 'Predict'
]
selected = st.sidebar.selectbox('Choose Section', option_menu)
if selected == 'Recent Data':
    dataframe()
elif selected == 'Line Chart':
    line_chart()
elif selected == 'Buy/Sell Recommendation':
    buy_sell_recommendation()
elif selected == 'Volatility Meter':
    volatility_meter()
elif selected == 'Best/Worst Day':
    best_worst_day()
elif selected == 'Financial Info':
    financial_info()
elif selected == 'News':
    news_section()
else:
    predict()

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ by <a href='https://www.linkedin.com/in/harsh-dugad-90067923b/' target='_blank'>Harsh Dugad</a></div>", unsafe_allow_html=True) 
