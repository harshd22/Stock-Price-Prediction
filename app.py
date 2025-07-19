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
import random

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

# --- Home Screen: Top Gainers & Losers, Market Summary, Sentiment, Quick Links ---
def get_top_movers():
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
        'BA', 'JPM', 'WMT', 'DIS', 'NKE', 'V', 'MA', 'PYPL', 'ADBE', 'CRM', 'CSCO', 'QCOM', 'ORCL', 'PEP', 'KO',
        'T', 'GE', 'GM', 'F', 'UBER', 'LYFT', 'SHOP', 'BABA', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS',
        'NIFTYBEES.NS', 'BANKBEES.NS', 'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV', 'BTC-USD', 'ETH-USD'
    ]
    data = yf.download(tickers, period='7d', interval='1d', group_by='ticker', progress=False)
    movers = []
    for t in tickers:
        try:
            close = data[t]['Close'].dropna()
            if len(close) < 2:
                continue
            last_two = close[-2:]
            change = (last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100
            movers.append({'symbol': t, 'change': change, 'last': last_two.iloc[-1]})
        except Exception:
            continue
    # Only positive gainers and negative losers
    gainers = [m for m in sorted(movers, key=lambda x: x['change'], reverse=True) if m['change'] > 0][:10]
    losers = [m for m in sorted(movers, key=lambda x: x['change']) if m['change'] < 0][:10]
    return gainers, losers

def get_market_summary():
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW JONES': '^DJI',
        'NIFTY 50': '^NSEI',
        'BANK NIFTY': '^NSEBANK',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'BTC-USD': 'BTC-USD',
        'ETH-USD': 'ETH-USD',
    }
    data = yf.download(list(indices.values()), period='7d', interval='1d', group_by='ticker', progress=False)
    summary = []
    for name, symbol in indices.items():
        try:
            close = data[symbol]['Close'].dropna()
            if len(close) < 2:
                continue
            last_two = close[-2:]
            change = (last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100
            summary.append({'name': name, 'last': last_two.iloc[-1], 'change': change})
        except Exception:
            continue
    return summary

def get_market_sentiment(gainers, losers):
    bullish = sum(1 for g in gainers if g['change'] > 2)
    bearish = sum(1 for l in losers if l['change'] < -2)
    if bullish > bearish and bullish >= 5:
        return 'Bullish', '🟢'
    elif bearish > bullish and bearish >= 5:
        return 'Bearish', '🔴'
    else:
        return 'Neutral', '🟡'

# --- Home Screen: Market Overview, Trending, Fun Fact, Sentiment Poll ---
def home_screen():
    st.markdown("---")
    st.markdown("<h2 style='color:#1a73e8;'>🏠 Home</h2>", unsafe_allow_html=True)
    st.markdown("<h4>Welcome to the Dynamic Stock Price Prediction & Financial Insights Platform!</h4>", unsafe_allow_html=True)
    st.write("Explore market overview, trending stocks, news, and more.")

    # Market Overview Cards (unchanged)
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW JONES': '^DJI',
        'NIFTY 50': '^NSEI',
        'BTC-USD': 'BTC-USD',
        'ETH-USD': 'ETH-USD',
    }
    data = yf.download(list(indices.values()), period='7d', interval='1d', group_by='ticker', progress=False)
    cols = st.columns(len(indices))
    for i, (name, symbol) in enumerate(indices.items()):
        try:
            close = data[symbol]['Close'].dropna()
            if len(close) < 2:
                continue
            last_two = close[-2:]
            change = (last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100
            color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
            arrow = '▲' if change > 0 else '▼' if change < 0 else '■'
            with cols[i]:
                st.markdown(f"<div style='text-align:center; background:#23272f; border-radius:10px; padding:1em; margin-bottom:1em;'><b>{name}</b><br><span style='color:{color}; font-size:1.2em;'>{last_two.iloc[-1]:.2f} {arrow} {change:+.2f}%</span></div>", unsafe_allow_html=True)
        except Exception:
            continue

    # Trending Stocks (unchanged)
    st.markdown("<h5 style='color:#1a73e8;'>🔥 Trending Stocks</h5>", unsafe_allow_html=True)
    trending = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'META', 'GOOGL', 'MSFT', 'NFLX', 'AMD', 'BTC-USD']
    tdata = yf.download(trending, period='7d', interval='1d', group_by='ticker', progress=False)
    tcols = st.columns(len(trending))
    for i, t in enumerate(trending):
        try:
            close = tdata[t]['Close'].dropna()
            if len(close) < 2:
                continue
            last_two = close[-2:]
            change = (last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100
            color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
            arrow = '▲' if change > 0 else '▼' if change < 0 else '■'
            with tcols[i]:
                st.markdown(f"<div style='text-align:center; background:#181c20; border-radius:10px; padding:0.5em;'><b>{t}</b><br><span style='color:{color};'>{last_two.iloc[-1]:.2f} {arrow} {change:+.2f}%</span></div>", unsafe_allow_html=True)
        except Exception:
            continue

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Did You Know? Finance Fact ---
    st.markdown("<h5 style='color:#1a73e8;'>💡 Did You Know?</h5>", unsafe_allow_html=True)
    finance_facts = [
        "The first stock exchange was established in Amsterdam in 1602.",
        "The term 'bull market' comes from the way a bull attacks, thrusting its horns upward.",
        "Warren Buffett bought his first stock at age 11.",
        "The New York Stock Exchange was founded in 1792.",
        "The ticker symbol for Berkshire Hathaway is BRK.A and BRK.B.",
        "The largest one-day percentage drop in the Dow Jones was on Black Monday, 1987.",
        "The S&P 500 covers about 80% of the U.S. equity market capitalization.",
        "India's Sensex index was launched in 1986.",
        "The word 'stock' comes from the Old English 'stocc', meaning tree trunk.",
        "The longest bull market in history lasted from 2009 to 2020."
    ]
    st.info(random.choice(finance_facts))

    # --- Market Sentiment Poll ---
    st.markdown("<h5 style='color:#1a73e8;'>📊 Market Sentiment Poll</h5>", unsafe_allow_html=True)
    if 'sentiment_votes' not in st.session_state:
        st.session_state['sentiment_votes'] = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('🐂 Bullish'):
            st.session_state['sentiment_votes']['Bullish'] += 1
    with col2:
        if st.button('🐻 Bearish'):
            st.session_state['sentiment_votes']['Bearish'] += 1
    with col3:
        if st.button('😐 Neutral'):
            st.session_state['sentiment_votes']['Neutral'] += 1
    st.write("**Current Poll Results:**")
    st.write(st.session_state['sentiment_votes'])

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

# --- Stock Screener Section ---
def stock_screener():
    st.markdown("---")
    st.markdown("<h2 style='color:#1a73e8;'>🔎 Stock Screener</h2>", unsafe_allow_html=True)
    ticker = st.text_input('Enter a stock ticker (e.g., AAPL, RELIANCE.NS)')
    if ticker:
        info = yf.Ticker(ticker).info
        st.write(info)

# --- Stock Comparison Section ---
def stock_comparison():
    st.markdown("---")
    st.markdown("<h2 style='color:#1a73e8;'>📊 Stock Comparison (Financials)</h2>", unsafe_allow_html=True)
    # Example: sector to tickers mapping (expand as needed)
    sector_map = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'TCS.NS', 'INFY.NS'],
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'JPM', 'BAC'],
        'Automotive': ['TSLA', 'F', 'GM', 'TATAMOTORS.NS'],
        'FMCG': ['ITC.NS', 'HINDUNILVR.NS'],
        'Energy': ['RELIANCE.NS', 'XOM', 'BPCL.NS'],
    }
    sector = st.selectbox('Select Sector', list(sector_map.keys()))
    tickers = sector_map[sector]
    stock1 = st.selectbox('Select First Stock', tickers, key='cmp1')
    stock2 = st.selectbox('Select Second Stock', tickers, key='cmp2')
    if stock1 and stock2 and stock1 != stock2:
        stocks = [stock1, stock2]
        fin_data = []
        for s in stocks:
            info = yf.Ticker(s).info
            fin_data.append({
                'Symbol': s,
                'Name': info.get('shortName', ''),
                'Gross Margin': info.get('grossMargins', 'N/A'),
                'EBITDA': info.get('ebitda', 'N/A'),
                'Revenue': info.get('totalRevenue', 'N/A'),
                'Net Income': info.get('netIncomeToCommon', 'N/A'),
                'PE Ratio': info.get('trailingPE', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
            })
        df = pd.DataFrame(fin_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.write('Select two different stocks to compare.')

# --- Main App Logic ---
option_menu = [
    'Home', 'Recent Data', 'Line Chart', 'Buy/Sell Recommendation', 'Stock Screener', 'Stock Comparison', 'Financial Info', 'News', 'Predict'
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
elif selected == 'Stock Screener':
    stock_screener()
elif selected == 'Stock Comparison':
    stock_comparison()
elif selected == 'Financial Info':
    financial_info()
elif selected == 'News':
    news_section()
else:
    predict()

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made with ❤️ by Harsh Dugad</div>", unsafe_allow_html=True) 
