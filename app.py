import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import datetime

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Dynamic Stock Price Prediction Platform")

# Sidebar Inputs
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, max_value=30, value=5)
model_name = st.sidebar.selectbox(
    "Prediction Model",
    ["Linear Regression", "Random Forest", "Extra Trees", "KNN", "XGBoost"]
)
show_tech = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_candle = st.sidebar.checkbox("Show Candlestick Chart", value=True)
run = st.sidebar.button("Run Prediction")

# --- Data Fetching ---
@st.cache_data(show_spinner=False)
def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# --- Feature Engineering ---
def add_features(df):
    df = df.copy()
    # Technical indicators
    df["SMA_14"] = SMAIndicator(df["Close"], window=14).sma_indicator()
    df["EMA_14"] = EMAIndicator(df["Close"], window=14).ema_indicator()
    df["RSI_14"] = RSIIndicator(df["Close"], window=14).rsi()
    bb = BollingerBands(df["Close"])
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df = df.dropna()
    return df

# --- Signal Generation ---
def generate_signal(row):
    # Example using RSI and MACD
    if row["RSI_14"] < 30 and row["MACD"] > 0:
        return "Buy"
    elif row["RSI_14"] > 70 and row["MACD"] < 0:
        return "Sell"
    else:
        return "Hold"

# --- Model Selection ---
def get_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=100, random_state=42)
    elif name == "KNN":
        return KNeighborsRegressor(n_neighbors=5)
    elif name == "XGBoost":
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    else:
        return LinearRegression()

# --- Prediction Pipeline ---
def predict_prices(df, forecast_days, model):
    df = df.copy()
    # Create target column
    df["Target"] = df["Close"].shift(-forecast_days)
    df = df.dropna()
    features = ["Close", "SMA_14", "EMA_14", "RSI_14", "BB_H", "BB_L", "MACD"]
    X = df[features].values
    y = df["Target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # Forecast
    last_X = df[features].values[-forecast_days:]
    last_X_scaled = scaler.transform(last_X)
    forecast = model.predict(last_X_scaled).flatten()
    last_date = df.index[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i+1) for i in range(forecast_days)]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})
    return r2, mae, forecast_df

# --- Main App Logic ---
if run:
    try:
        df = fetch_data(symbol, start_date, end_date)
        if df.empty:
            st.error("No data found for the selected symbol and date range.")
        else:
            st.success(f"Data loaded for {symbol} ({len(df)} rows)")
            df_feat = add_features(df)
            # Add trading signals to historical data
            df_feat["Signal"] = df_feat.apply(generate_signal, axis=1)
            latest_signal = df_feat["Signal"].iloc[-1]
            st.subheader(f"Latest Trading Signal: :blue[{latest_signal}]")
            if show_tech:
                st.subheader("Technical Indicators")
                st.line_chart(df_feat[["Close", "SMA_14", "EMA_14"]])
                st.line_chart(df_feat[["RSI_14", "MACD"]])
            if show_candle:
                st.subheader("Candlestick Chart")
                fig = go.Figure(data=[go.Candlestick(
                    x=df_feat.index,
                    open=df_feat["Open"],
                    high=df_feat["High"],
                    low=df_feat["Low"],
                    close=df_feat["Close"]
                )])
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Model Training & Prediction")
            model = get_model(model_name)
            r2, mae, forecast_df = predict_prices(df_feat, forecast_days, model)
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**Mean Absolute Error:** {mae:.3f}")
            # Add signals to forecasted days based on predicted price movement
            last_close = df_feat["Close"].iloc[-1]
            forecast_df["Signal"] = forecast_df["Forecast"].apply(
                lambda x: "Buy" if x > last_close * 1.01 else ("Sell" if x < last_close * 0.99 else "Hold")
            )
            st.write("**Forecasted Prices and Signals:**")
            st.dataframe(forecast_df)
            st.line_chart(pd.Series(forecast_df["Forecast"].values, index=forecast_df["Date"]))
            st.write("**Forecasted Signals:**")
            st.dataframe(forecast_df[["Date", "Signal"]])
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Configure your options in the sidebar and click 'Run Prediction'.")

st.markdown("---")
st.caption("Created by [Harsh Dugad](https://www.linkedin.com/in/harsh-dugad-90067923b/)") 
