import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#F5F7FA;
}

/* Hero */

.hero{
background-image:url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05");
background-size:cover;
background-position:center;
height:320px;
border-radius:18px;
padding:40px;
display:flex;
align-items:center;
color:white;
margin-bottom:25px;
}

.hero h1{
font-size:42px;
}

.hero p{
font-size:20px;
}

/* Cards */

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,.08);
}

/* Search Box */

.searchbox{
background:white;
padding:20px;
border-radius:15px;
margin-top:-60px;
position:relative;
box-shadow:0 6px 15px rgba(0,0,0,.15);
}

/* Metric */

[data-testid="stMetric"]{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0 2px 8px rgba(0,0,0,.1);
}

/* Button */

.stButton>button{
background:#0057D9;
color:white;
border:none;
height:50px;
border-radius:8px;
font-size:18px;
font-weight:bold;
width:100%;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.image("assets/images.jpg", width=120)

    st.title("⚙ Settings")

    future_months = st.slider(
        "Forecast Months",
        1,
        24,
        12
    )

    st.markdown("---")

    st.success("RNN Forecast Model")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

loader = DataLoader("data/airline_passengers.csv")
df = loader.load_data()

# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.markdown("""
<div class="hero">
<div>

<h1>Travel Anywhere Around The World</h1>

<p>Predict Future Airline Passenger Demand using RNN</p>

</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SEARCH CARD
# ---------------------------------------------------------

st.markdown('<div class="searchbox">', unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)

with c1:
    st.selectbox(
        "Departure",
        ["New York","London","Tokyo","Delhi"]
    )

with c2:
    st.selectbox(
        "Destination",
        ["Paris","Singapore","Dubai","Sydney"]
    )

with c3:
    st.date_input("Departure Date")

with c4:
    st.date_input("Return Date")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

mae,mse,rmse=Evaluator().evaluate()

m1,m2,m3=st.columns(3)

m1.metric("MAE",round(mae,2))
m2.metric("MSE",round(mse,2))
m3.metric("RMSE",round(rmse,2))

st.write("")

# ---------------------------------------------------------
# DATA + TREND
# ---------------------------------------------------------

left,right=st.columns([1,2])

with left:

    st.subheader("Passenger Dataset")

    st.dataframe(df,height=400)

with right:

    st.subheader("Historical Passenger Trend")

    fig=px.line(
        df,
        x=df.index,
        y="Passengers",
        markers=True
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# ---------------------------------------------------------
# FORECAST
# ---------------------------------------------------------

st.header("Future Passenger Forecast")

if st.button("Generate Forecast"):

    with st.spinner("Running RNN Model..."):

        forecaster=Forecaster()

        future=forecaster.forecast(future_months)

        future_dates=pd.date_range(
            start=df.index[-1]+pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df=pd.DataFrame({

            "Month":future_dates,

            "Predicted Passengers":future.flatten()

        })

    col1,col2=st.columns([1,2])

    with col1:

        st.subheader("Forecast Values")

        st.dataframe(forecast_df)

        csv=forecast_df.to_csv(index=False).encode()

        st.download_button(
            "Download CSV",
            csv,
            "forecast.csv",
            "text/csv"
        )

    with col2:

        fig=go.Figure()

        fig.add_trace(go.Scatter(

            x=df.index,

            y=df["Passengers"],

            name="Historical"

        ))

        fig.add_trace(go.Scatter(

            x=forecast_df["Month"],

            y=forecast_df["Predicted Passengers"],

            name="Forecast",

            line=dict(color="red",dash="dash")

        ))

        fig.update_layout(

            template="plotly_white",

            height=500

        )

        st.plotly_chart(fig,use_container_width=True)