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
import requests

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Prediction & Insights", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #181c20;}
    .block-container {padding-top: 2rem;}
    .stButton>button {background-color: #1a73e8; color: white;}
    .stDataFrame {background-color: #23272f;}
    .ticker {
        background: #23272f;
        color: #fff;
        padding: 0.5em 1em;
        font-size: 1.1em;
        border-radius: 8px;
        margin-bottom: 1em;
        overflow-x: auto;
        white-space: nowrap;
        animation: ticker 20s linear infinite;
    }
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#1a73e8;'>📈 Dynamic Stock Price Prediction & Financial Insights</h1>", unsafe_allow_html=True)
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by Harsh Dugad")

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

# NewsAPI.org integration
def get_newsapi_news(query):
    api_key = "7227a18d537b49779ffb88a209d0d0e3"
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles[:5]  # Top 5 news
    else:
        return []

# --- Home Screen: Top Gainers & Losers ---
def get_top_movers():
    # Use a set of popular tickers for demo; for real use, fetch from a screener API
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
        'BA', 'JPM', 'WMT', 'DIS', 'NKE', 'V', 'MA', 'PYPL', 'ADBE', 'CRM', 'CSCO', 'QCOM', 'ORCL', 'PEP', 'KO'
    ]
    data = yf.download(tickers, period='2d', interval='1d', group_by='ticker', progress=False)
    movers = []
    for t in tickers:
        try:
            close = data[t]['Close']
            if len(close) < 2:
                continue
            change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            movers.append({'symbol': t, 'change': change, 'last': close.iloc[-1]})
        except Exception:
            continue
    movers = sorted(movers, key=lambda x: x['change'], reverse=True)
    gainers = movers[:5]
    losers = movers[-5:][::-1]
    return gainers, losers

def home_screen():
    st.markdown("---")
    st.markdown("<h2 style='color:#1a73e8;'>🏠 Home</h2>", unsafe_allow_html=True)
    st.markdown("<h4>Welcome to the Dynamic Stock Price Prediction & Financial Insights Platform!</h4>", unsafe_allow_html=True)
    st.write("Explore top gainers and losers, get predictions, news, and more.")
    gainers, losers = get_top_movers()
    st.markdown("<h5 style='color:green;'>Top Gainers</h5>", unsafe_allow_html=True)
    gainer_ticker = " | ".join([f"<b>{g['symbol']}</b> ({g['change']:+.2f}%)" for g in gainers])
    st.markdown(f"<div class='ticker'>{gainer_ticker}</div>", unsafe_allow_html=True)
    st.markdown("<h5 style='color:red;'>Top Losers</h5>", unsafe_allow_html=True)
    loser_ticker = " | ".join([f"<b>{l['symbol']}</b> ({l['change']:+.2f}%)" for l in losers])
    st.markdown(f"<div class='ticker'>{loser_ticker}</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4>🔍 Use the sidebar to search for a stock and explore predictions, news, and more!</h4>", unsafe_allow_html=True)

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
    news = get_newsapi_news(option)
    if news:
        for item in news:
            title = item.get('title', 'No Title')
            link = item.get('url', '#')
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
    'Home', 'Recent Data', 'Line Chart', 'Buy/Sell Recommendation', 'Financial Info', 'News', 'Predict'
]
selected = st.sidebar.selectbox('Choose Section', option_menu)
if selected == 'Home':
    home_screen()
elif selected == 'Recent Data':
    dataframe()
elif selected == 'Line Chart':
    line_chart()
elif selected == 'Buy/Sell Recommendation':
    buy_sell_recommendation()
elif selected == 'Financial Info':
    financial_info()
elif selected == 'News':
    news_section()
else:
    predict()

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ by Harsh Dugad</div>", unsafe_allow_html=True) 
