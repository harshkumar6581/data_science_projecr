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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5f7ff 0%,
            #eef2ff 45%,
            #f8fafc 100%
        );
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #1e1b4b 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* KPI cards */
    .kpi-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        box-shadow: 0px 8px 25px rgba(15, 23, 42, 0.08);
        border-left: 6px solid #6366f1;
        min-height: 130px;
    }

    .kpi-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Section headers */
    .section-header {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Info cards */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0px 5px 20px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
        font-weight: 700;
        padding: 10px 20px;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 10px;
        background: linear-gradient(
            90deg,
            #059669,
            #10b981
        );
        color: white;
        font-weight: 700;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_data(uploaded_file):

    df = pd.read_csv(uploaded_file)

    required_columns = ["Date", "Sales"]

    for column in required_columns:
        if column not in df.columns:
            st.error(
                f"Required column '{column}' not found in dataset."
            )
            st.stop()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Sales"]
    )

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"]
    )

    df = df.set_index("Date")

    return df


def create_demo_data():

    np.random.seed(42)

    dates = pd.date_range(
        start="2021-01-01",
        end="2025-12-31",
        freq="D"
    )

    n = len(dates)

    trend = np.linspace(
        100,
        220,
        n
    )

    weekly = 20 * np.sin(
        2 * np.pi * np.arange(n) / 7
    )

    yearly = 35 * np.sin(
        2 * np.pi * np.arange(n) / 365.25
    )

    noise = np.random.normal(
        0,
        12,
        n
    )

    sales = (
        trend
        + weekly
        + yearly
        + noise
    )

    sales = np.maximum(
        sales,
        20
    )

    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales
    })

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.set_index("Date")

    return df


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted)
            / actual
        )
    ) * 100

    return mae, rmse, mape


def create_lstm_sequences(
    data,
    lookback
):

    X = []
    y = []

    for i in range(
        lookback,
        len(data)
    ):

        X.append(
            data[
                i-lookback:i,
                0
            ]
        )

        y.append(
            data[i, 0]
        )

    return np.array(X), np.array(y)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="margin-bottom:0;">
        Sales Forecasting
        </h2>
        <p style="color:#cbd5e1;">
        Business Intelligence Dashboard
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    data_source = st.radio(
        "Data Source",
        [
            "Demo Dataset",
            "Upload CSV"
        ]
    )

    uploaded_file = None

    if data_source == "Upload CSV":

        uploaded_file = st.file_uploader(
            "Upload Sales CSV",
            type=["csv"]
        )

    st.divider()

    forecast_months = st.slider(
        "Forecast Months",
        min_value=3,
        max_value=24,
        value=12
    )

    model_choice = st.selectbox(
        "Forecasting Model",
        [
            "SARIMA",
            "LSTM"
        ]
    )

    st.divider()

    st.caption(
        "Time Series Forecasting Project"
    )


# =========================================================
# LOAD DATA
# =========================================================

if data_source == "Upload CSV":

    if uploaded_file is not None:

        df = load_data(
            uploaded_file
        )

    else:

        st.info(
            "Upload a CSV file to start forecasting."
        )

        st.stop()

else:

    df = create_demo_data()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
    Sales Forecasting & Business Analytics
    </div>

    <div class="sub-title">
    Analyze historical sales, discover patterns and forecast future business demand.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BASIC DATA PREPARATION
# =========================================================

df["Sales"] = df["Sales"].interpolate()

monthly_sales = (
    df["Sales"]
    .resample("MS")
    .sum()
)


# =========================================================
# KPI SECTION
# =========================================================

total_sales = df["Sales"].sum()

average_daily_sales = df["Sales"].mean()

highest_sales = df["Sales"].max()

latest_month_sales = monthly_sales.iloc[-1]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
            Total Sales
            </div>

            <div class="kpi-value">
            {total_sales:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
            Average Daily Sales
            </div>

            <div class="kpi-value">
            {average_daily_sales:,.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
            Highest Daily Sales
            </div>

            <div class="kpi-value">
            {highest_sales:,.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
            Latest Monthly Sales
            </div>

            <div class="kpi-value">
            {latest_month_sales:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DATA OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-header">Data Overview</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="info-card">
        <b>Dataset Information</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    overview = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Start Date",
            "End Date",
            "Missing Values"
        ],
        "Value": [
            len(df),
            len(df.columns),
            str(df.index.min().date()),
            str(df.index.max().date()),
            int(df.isnull().sum().sum())
        ]
    })

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">
        <b>Recent Sales Data</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )


# =========================================================
# SALES TREND
# =========================================================

st.markdown(
    '<div class="section-header">Sales Trend</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(15, 5)
)

ax.plot(
    df.index,
    df["Sales"],
    linewidth=1.5
)

ax.set_title(
    "Daily Sales Trend",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")

ax.grid(
    alpha=0.2
)

st.pyplot(
    fig,
    use_container_width=True
)


# =========================================================
# MONTHLY ANALYSIS
# =========================================================

st.markdown(
    '<div class="section-header">Monthly Sales Analysis</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(15, 5)
)

ax.plot(
    monthly_sales.index,
    monthly_sales,
    linewidth=2
)

ax.fill_between(
    monthly_sales.index,
    monthly_sales,
    alpha=0.15
)

ax.set_title(
    "Monthly Sales Performance",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Date")
ax.set_ylabel("Monthly Sales")

ax.grid(
    alpha=0.2
)

st.pyplot(
    fig,
    use_container_width=True
)


# =========================================================
# SEASONALITY
# =========================================================

st.markdown(
    '<div class="section-header">Seasonality Analysis</div>',
    unsafe_allow_html=True
)

if len(monthly_sales) >= 24:

    decomposition = seasonal_decompose(
        monthly_sales,
        model="additive",
        period=12
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10),
        sharex=True
    )

    axes[0].plot(
        decomposition.observed
    )

    axes[0].set_title(
        "Observed"
    )

    axes[1].plot(
        decomposition.trend
    )

    axes[1].set_title(
        "Trend"
    )

    axes[2].plot(
        decomposition.seasonal
    )

    axes[2].set_title(
        "Seasonality"
    )

    axes[3].plot(
        decomposition.resid
    )

    axes[3].set_title(
        "Residual"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

else:

    st.warning(
        "At least 24 months of data is recommended for seasonal analysis."
    )


# =========================================================
# MODELING SECTION
# =========================================================

st.markdown(
    '<div class="section-header">Forecasting Model</div>',
    unsafe_allow_html=True
)


if len(monthly_sales) < 36:

    st.warning(
        "For reliable SARIMA/LSTM forecasting, use at least 36 months of monthly data."
    )

else:

    split_index = int(
        len(monthly_sales) * 0.80
    )

    train = monthly_sales.iloc[
        :split_index
    ]

    test = monthly_sales.iloc[
        split_index:
    ]


    # =====================================================
    # SARIMA
    # =====================================================

    if model_choice == "SARIMA":

        with st.spinner(
            "Training SARIMA model..."
        ):

            model = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(
                    1,
                    1,
                    1,
                    12
                ),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            result = model.fit(
                disp=False
            )

            predictions = result.get_forecast(
                steps=len(test)
            ).predicted_mean

            predictions.index = test.index

            mae, rmse, mape = calculate_metrics(
                test,
                predictions
            )


        # ---------------------------------------------
        # Model Metrics
        # ---------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{mae:,.2f}"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{rmse:,.2f}"
            )

        with col3:

            st.metric(
                "MAPE",
                f"{mape:.2f}%"
            )


        # ---------------------------------------------
        # Actual vs Prediction
        # ---------------------------------------------

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.plot(
            train.index,
            train,
            label="Training"
        )

        ax.plot(
            test.index,
            test,
            label="Actual"
        )

        ax.plot(
            predictions.index,
            predictions,
            label="SARIMA Forecast",
            linewidth=2
        )

        ax.set_title(
            "SARIMA: Actual vs Forecast",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")

        ax.legend()

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


        # ---------------------------------------------
        # Future Forecast
        # ---------------------------------------------

        st.markdown(
            "### Future Sales Forecast"
        )

        final_model = SARIMAX(
            monthly_sales,
            order=(1, 1, 1),
            seasonal_order=(
                1,
                1,
                1,
                12
            ),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        final_result = final_model.fit(
            disp=False
        )

        future = final_result.get_forecast(
            steps=forecast_months
        )

        future_mean = future.predicted_mean

        confidence = future.conf_int()


        # ---------------------------------------------
        # Future Forecast Chart
        # ---------------------------------------------

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.plot(
            monthly_sales.index,
            monthly_sales,
            label="Historical"
        )

        ax.plot(
            future_mean.index,
            future_mean,
            label="Forecast",
            linewidth=2
        )

        ax.fill_between(
            future_mean.index,
            confidence.iloc[:, 0],
            confidence.iloc[:, 1],
            alpha=0.15,
            label="Confidence Interval"
        )

        ax.set_title(
            f"Next {forecast_months} Months Sales Forecast",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")

        ax.legend()

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


        # ---------------------------------------------
        # Forecast Table
        # ---------------------------------------------

        forecast_table = pd.DataFrame({
            "Date": future_mean.index,
            "Forecast": future_mean.values,
            "Lower Bound": confidence.iloc[:, 0].values,
            "Upper Bound": confidence.iloc[:, 1].values
        })

        st.dataframe(
            forecast_table.round(2),
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------
        # Download
        # ---------------------------------------------

        csv = forecast_table.to_csv(
            index=False
        )

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )


    # =====================================================
    # LSTM
    # =====================================================

    else:

        lookback = 12

        scaler = MinMaxScaler()

        train_scaled = scaler.fit_transform(
            train.values.reshape(-1, 1)
        )

        test_scaled = scaler.transform(
            test.values.reshape(-1, 1)
        )


        X_train, y_train = create_lstm_sequences(
            train_scaled,
            lookback
        )


        combined = np.concatenate(
            [
                train_scaled[-lookback:],
                test_scaled
            ]
        )

        X_test, y_test = create_lstm_sequences(
            combined,
            lookback
        )


        X_train = X_train.reshape(
            X_train.shape[0],
            X_train.shape[1],
            1
        )

        X_test = X_test.reshape(
            X_test.shape[0],
            X_test.shape[1],
            1
        )


        with st.spinner(
            "Training LSTM model..."
        ):

            lstm = Sequential()

            lstm.add(
                LSTM(
                    64,
                    return_sequences=True,
                    input_shape=(
                        lookback,
                        1
                    )
                )
            )

            lstm.add(
                Dropout(0.2)
            )

            lstm.add(
                LSTM(32)
            )

            lstm.add(
                Dropout(0.2)
            )

            lstm.add(
                Dense(1)
            )

            lstm.compile(
                optimizer="adam",
                loss="mse"
            )


            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )


            lstm.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=16,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            )


        # ---------------------------------------------
        # Predictions
        # ---------------------------------------------

        scaled_prediction = lstm.predict(
            X_test,
            verbose=0
        )

        predictions = scaler.inverse_transform(
            scaled_prediction
        ).flatten()

        actual = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()


        mae, rmse, mape = calculate_metrics(
            actual,
            predictions
        )


        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{mae:,.2f}"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{rmse:,.2f}"
            )

        with col3:

            st.metric(
                "MAPE",
                f"{mape:.2f}%"
            )


        # ---------------------------------------------
        # LSTM Chart
        # ---------------------------------------------

        prediction_index = test.index

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.plot(
            prediction_index,
            actual,
            label="Actual"
        )

        ax.plot(
            prediction_index,
            predictions,
            label="LSTM Forecast",
            linewidth=2
        )

        ax.set_title(
            "LSTM: Actual vs Forecast",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")

        ax.legend()

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.markdown(
    '<div class="section-header">Business Insights</div>',
    unsafe_allow_html=True
)

growth = (
    (
        monthly_sales.iloc[-1]
        - monthly_sales.iloc[0]
    )
    / monthly_sales.iloc[0]
) * 100


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f"""
        <div class="info-card">

        <h4>Sales Growth</h4>

        <p>
        Sales changed by approximately
        <b>{growth:.2f}%</b>
        over the available period.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    highest_month = monthly_sales.idxmax()

    st.markdown(
        f"""
        <div class="info-card">

        <h4>Best Sales Month</h4>

        <p>
        The highest monthly sales occurred around
        <b>{highest_month.strftime("%B %Y")}</b>.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
    ">

    <b>Sales Forecasting & Business Analytics</b>

    <br>

    Built with Python, Pandas, Statsmodels,
    TensorFlow and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)