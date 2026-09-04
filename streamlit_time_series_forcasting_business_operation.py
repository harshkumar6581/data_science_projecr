import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8faff 0%, #eef2ff 50%, #f8fafc 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .dashboard-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .dashboard-subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 22px;
        min-height: 145px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.9);
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.14);
    }
    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 12px;
    }
    .kpi-value {
        font-size: 31px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 6px;
    }
    .kpi-description {
        font-size: 12px;
        color: #94a3b8;
    }
    .kpi-blue { border-left: 5px solid #2563eb; }
    .kpi-purple { border-left: 5px solid #7c3aed; }
    .kpi-green { border-left: 5px solid #059669; }
    .kpi-orange { border-left: 5px solid #ea580c; }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 35px;
        margin-bottom: 18px;
    }
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    }
    .insight-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
        min-height: 120px;
    }
    .insight-title {
        font-size: 15px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 8px;
    }
    .insight-value {
        font-size: 20px;
        font-weight: 800;
        color: #4f46e5;
    }
    .footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">
        Time Series Forecasting for Business Operations
    </div>
    <div class="dashboard-subtitle">
        Analyze historical business data and generate future forecasts using statistical and deep learning models.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DEMO DATA
# ============================================================

def create_demo_data():
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="D")
    n = len(dates)
    trend = np.linspace(100, 220, n)
    yearly_seasonality = 25 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    monthly_seasonality = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 8, n)
    sales = np.maximum(trend + yearly_seasonality + monthly_seasonality + noise, 20)
    return pd.DataFrame({"Date": dates, "Sales": sales})

# ============================================================
# CSV LOADER
# ============================================================

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    required_columns = ["Date", "Sales"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"CSV file must contain: Date and Sales. Missing: {missing_columns}")
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    return df.dropna(subset=["Date", "Sales"]).drop_duplicates(subset=["Date"]).sort_values("Date")

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    non_zero = actual != 0
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100 if np.sum(non_zero) > 0 else 0
    return mae, rmse, mape

def create_sequences(data, lookback=12):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown('<h2 style="text-align:center; margin-bottom:25px;">Forecasting Control</h2>', unsafe_allow_html=True)
data_source = st.sidebar.radio("Data Source", ["Demo Dataset", "Upload CSV"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) if data_source == "Upload CSV" else None
model_type = st.sidebar.selectbox("Forecasting Model", ["SARIMA", "LSTM"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", min_value=3, max_value=24, value=12)

# ============================================================
# LOAD DATA
# ============================================================

if data_source == "Demo Dataset":
    df = create_demo_data()
else:
    if uploaded_file is None:
        st.info("Please upload a CSV file containing Date and Sales columns.")
        st.stop()
    df = load_csv(uploaded_file)
    if df is None:
        st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["Date", "Sales"]).sort_values("Date").set_index("Date")
monthly_sales = df["Sales"].resample("MS").sum().dropna()

# ============================================================
# KPI CALCULATIONS & RENDER
# ============================================================

total_sales = float(df["Sales"].sum())
average_sales = float(df["Sales"].mean())
max_sales = float(df["Sales"].max())
data_points = int(len(df))

st.markdown('<div class="section-title">Business Overview</div>', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-title">Total Sales</div><div class="kpi-value">{total_sales:,.0f}</div><div class="kpi-description">Total historical sales</div></div>', unsafe_allow_html=True)

with kpi2:
    st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-title">Average Sales</div><div class="kpi-value">{average_sales:,.2f}</div><div class="kpi-description">Average sales per day</div></div>', unsafe_allow_html=True)

with kpi3:
    st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-title">Maximum Sales</div><div class="kpi-value">{max_sales:,.0f}</div><div class="kpi-description">Highest recorded value</div></div>', unsafe_allow_html=True)

with kpi4:
    st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-title">Data Points</div><div class="kpi-value">{data_points:,}</div><div class="kpi-description">Historical observations</div></div>', unsafe_allow_html=True)

# ============================================================
# DATA OVERVIEW
# ============================================================

st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)
overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    overview_df = pd.DataFrame({
        "Metric": ["Start Date", "End Date", "Total Records", "Missing Values"],
        "Value": [df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"), str(len(df)), str(df["Sales"].isna().sum())]
    })
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

with overview_col2:
    st.markdown('<div class="info-box"><b>Dataset Information</b><br><br>The dashboard converts the historical data into a monthly time series for forecasting.<br><br><b>Required CSV columns:</b><br>Date — date of observation<br>Sales — numerical business value</div>', unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================

st.markdown('<div class="section-title">Historical Sales Trend</div>', unsafe_allow_html=True)
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df.index, df["Sales"], linewidth=1)
ax1.set_title("Daily Sales Trend", fontsize=15, fontweight="bold")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
ax1.grid(alpha=0.25)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

st.markdown('<div class="section-title">Monthly Sales Analysis</div>', unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(monthly_sales.index, monthly_sales.values, linewidth=2)
ax2.set_title("Monthly Sales", fontsize=15, fontweight="bold")
ax2.set_xlabel("Month")
ax2.set_ylabel("Sales")
ax2.grid(alpha=0.25)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ============================================================
# SEASONAL DECOMPOSITION
# ============================================================

st.markdown('<div class="section-title">Seasonality Analysis</div>', unsafe_allow_html=True)
if len(monthly_sales) >= 24:
    decomposition = seasonal_decompose(monthly_sales, model="additive", period=12)
    fig3, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(decomposition.observed)
    axes[0].set_title("Observed")
    axes[1].plot(decomposition.trend)
    axes[1].set_title("Trend")
    axes[2].plot(decomposition.seasonal)
    axes[2].set_title("Seasonality")
    axes[3].plot(decomposition.resid)
    axes[3].set_title("Residual")
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.warning("At least 24 monthly observations are recommended for seasonal decomposition.")

# ============================================================
# MODELING
# ============================================================

st.markdown('<div class="section-title">Forecasting Model</div>', unsafe_allow_html=True)
if len(monthly_sales) < 36:
    st.error(f"Not enough monthly data for reliable forecasting. Current: {len(monthly_sales)}, Recommended: 36+")
    st.stop()

test_size = max(6, int(len(monthly_sales) * 0.20))
train = monthly_sales.iloc[:-test_size]
test = monthly_sales.iloc[-test_size:]

if model_type == "SARIMA":
    st.write("Training SARIMA model...")
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False, maxiter=300)
        predictions = fitted_model.forecast(steps=len(test))
        predictions.index = test.index
        mae, rmse, mape = calculate_metrics(test.values, predictions.values)

        st.success("SARIMA model trained successfully.")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")
        m3.metric("MAPE", f"{mape:.2f}%")

        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(14, 5))
        ax4.plot(train.index, train.values, label="Train")
        ax4.plot(test.index, test.values, label="Actual Test")
        ax4.plot(predictions.index, predictions.values, label="Predicted")
        ax4.set_title("Actual vs Predicted Sales", fontsize=15, fontweight="bold")
        ax4.legend()
        ax4.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        st.markdown('<div class="section-title">Future Forecast</div>', unsafe_allow_html=True)
        future_model = SARIMAX(monthly_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        future_fit = future_model.fit(disp=False, maxiter=300)
        future_result = future_fit.get_forecast(steps=forecast_horizon)
        future_forecast = future_result.predicted_mean
        confidence = future_result.conf_int()

        fig5, ax5 = plt.subplots(figsize=(14, 6))
        ax5.plot(monthly_sales.index, monthly_sales.values, label="Historical")
        ax5.plot(future_forecast.index, future_forecast.values, label="Forecast", linewidth=3)
        ax5.fill_between(future_forecast.index, confidence.iloc[:, 0], confidence.iloc[:, 1], alpha=0.2, label="Confidence Interval")
        ax5.set_title("Future Sales Forecast", fontsize=15, fontweight="bold")
        ax5.legend()
        ax5.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        forecast_table = pd.DataFrame({"Date": future_forecast.index.strftime("%Y-%m-%d"), "Forecast": future_forecast.values.round(2)})
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        st.download_button(label="Download Forecast CSV", data=forecast_table.to_csv(index=False).encode("utf-8"), file_name="sales_forecast.csv", mime="text/csv")
    except Exception as e:
        st.error("SARIMA model could not be trained.")
        st.code(str(e))

else:
    st.write("Training LSTM model...")
    try:
        values = monthly_sales.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        lookback = 12
        X, y = create_sequences(scaled_values, lookback)
        split_index = int(len(X) * 0.80)

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=0)

        predictions_scaled = lstm_model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled).flatten()
        actual = scaler.inverse_transform(y_test).flatten()

        mae, rmse, mape = calculate_metrics(actual, predictions)
        st.success("LSTM model trained successfully.")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")
        m3.metric("MAPE", f"{mape:.2f}%")

        test_dates = monthly_sales.index[lookback + split_index:][:len(predictions)]
        fig6, ax6 = plt.subplots(figsize=(14, 5))
        ax6.plot(test_dates, actual, label="Actual")
        ax6.plot(test_dates, predictions, label="Predicted", linewidth=2)
        ax6.set_title("LSTM Actual vs Predicted", fontsize=15, fontweight="bold")
        ax6.legend()
        ax6.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        # Future Forecast
        last_sequence = scaled_values[-lookback:].reshape(1, lookback, 1)
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(forecast_horizon):
            next_prediction = lstm_model.predict(current_sequence, verbose=0)[0][0]
            future_predictions.append(next_prediction)
            current_sequence = np.append(current_sequence[:, 1:, :], [[[next_prediction]]], axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq="MS")

        fig7, ax7 = plt.subplots(figsize=(14, 6))
        ax7.plot(monthly_sales.index, monthly_sales.values, label="Historical")
        ax7.plot(future_dates, future_predictions, label="LSTM Forecast", linewidth=3)
        ax7.set_title("LSTM Future Sales Forecast", fontsize=15, fontweight="bold")
        ax7.legend()
        ax7.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

        forecast_table = pd.DataFrame({"Date": future_dates.strftime("%Y-%m-%d"), "Forecast": future_predictions.round(2)})
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        st.download_button(label="Download Forecast CSV", data=forecast_table.to_csv(index=False).encode("utf-8"), file_name="lstm_sales_forecast.csv", mime="text/csv")
    except Exception as e:
        st.error("LSTM model could not be trained.")
        st.code(str(e))

# ============================================================
# BUSINESS INSIGHTS & FOOTER
# ============================================================

st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)
first_value, last_value = float(monthly_sales.iloc[0]), float(monthly_sales.iloc[-1])
growth = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

best_month, best_value = monthly_sales.idxmax(), float(monthly_sales.max())
lowest_month, lowest_value = monthly_sales.idxmin(), float(monthly_sales.min())

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(f'<div class="insight-card"><div class="insight-title">Overall Growth</div><div class="insight-value">{growth:.2f}%</div><div>Change from first to latest month</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown(f'<div class="insight-card"><div class="insight-title">Best Performing Month</div><div class="insight-value">{best_month.strftime("%B %Y")}</div><div>Sales: {best_value:,.0f}</div></div>', unsafe_allow_html=True)
with i3:
    st.markdown(f'<div class="insight-card"><div class="insight-card"><div class="insight-title">Lowest Performing Month</div><div class="insight-value">{lowest_month.strftime("%B %Y")}</div><div>Sales: {lowest_value:,.0f}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Project Information</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><b>Project:</b> Time Series Forecasting for Business Operations<br><br><b>Techniques:</b> SARIMA, LSTM, Seasonality Analysis<br><br><b>Dashboard:</b> Streamlit</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Time Series Forecasting Dashboard | Built with Python, Pandas, Statsmodels, TensorFlow and Streamlit</div>', unsafe_allow_html=True)