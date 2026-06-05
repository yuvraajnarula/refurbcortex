import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app.core.ev_soh_forecaster import EVSoHForecaster
from app.utils.logger import app_logger

st.set_page_config(page_title="🔋 EV Fleet Analytics", layout="wide")
st.title("🔋 EV State-of-Health Fleet Analytics")
st.caption("Predictive degradation tracking, warranty risk scoring, and margin impact analysis")

@st.cache_data(ttl=300)
def generate_fleet_data(n=500):
    np.random.seed(42)
    ages = np.random.uniform(0.5, 8, n)
    kms = np.random.uniform(5000, 120000, n)
    fast_charge = np.random.uniform(0.05, 0.7, n)
    temps = np.random.uniform(22, 45, n)
    brands = np.random.choice(["Tata", "MG", "Hyundai", "BYD"], n)
    models = np.random.choice(["Nexon", "ZS EV", "Ioniq", "Seal"], n)

    forecaster = EVSoHForecaster()
    soh = [forecaster.predict(a, k, fc, t)["predicted_soh_pct"] for a, k, fc, t in zip(ages, kms, fast_charge, temps)]
    
    df = pd.DataFrame({
        "age_years": ages, "odometer_km": kms, "fast_charge_pct": fast_charge*100, "avg_temp": temps,
        "brand": brands, "model": models, "soh_pct": soh,
        "warranty_risk": [s < 70 for s in soh],
        "margin_impact": [0.4 if s < 55 else 0.7 if s < 70 else 0.9 if s < 85 else 1.0 for s in soh]
    })
    return df

df = generate_fleet_data()
forecaster = EVSoHForecaster()

# 🔹 Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Fleet Size", len(df))
c2.metric("🔋 Avg SoH", f"{df['soh_pct'].mean():.1f}%")
c3.metric("⚠️ Warranty Risk %", f"{df['warranty_risk'].mean()*100:.1f}%")
c4.metric("💸 Margin Multiplier", f"{df['margin_impact'].mean():.2f}x")

st.divider()

# 🔹 Interactive Charts
tab1, tab2, tab3 = st.tabs(["📉 Degradation Trends", "⚠️ Risk Heatmap", "🔍 Single Vehicle Forecaster"])

with tab1:
    fig1 = px.scatter(df, x="age_years", y="soh_pct", color="brand", size="odometer_km",
                      title="SoH vs Vehicle Age (Size = Odometer)", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="odometer_km", y="soh_pct", color="fast_charge_pct",
                      title="SoH vs Odometer (Color = Fast Charge %)", marginal_y="box")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    risk_df = df.groupby(["brand", "model"]).agg(
        avg_soh=("soh_pct", "mean"),
        risk_pct=("warranty_risk", "mean"),
        avg_margin=("margin_impact", "mean")
    ).reset_index()

    fig3 = px.imshow(risk_df.pivot(index="brand", columns="model", values="risk_pct"),
                     text_auto=".0%", color_continuous_scale="RdYlGn_r",
                     title="Warranty Risk % by Brand/Model")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("🔋 Predict SoH for a Single EV")
    col1, col2 = st.columns(2)
    s_age = col1.number_input("Age (Years)", 0.5, 10.0, 3.0)
    s_km = col2.number_input("Odometer (km)", 1000, 200000, 45000)
    s_fc = st.slider("Fast Charge %", 0, 100, 30)
    s_temp = st.slider("Avg Climate °C", 15, 50, 32)

    if st.button("🔍 Run Forecast", type="primary"):
        pred = forecaster.predict(s_age, s_km, s_fc/100, s_temp)
        st.success(f"✅ Predicted SoH: **{pred['predicted_soh_pct']}%** | Tier: `{pred['health_tier']}`")
        
        # Margin impact gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pred["predicted_soh_pct"],
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [None, 100]}, "bar": {"color": "darkblue"},
                   "steps": [{"range": [40, 55], "color": "red"}, {"range": [55, 70], "color": "orange"},
                             {"range": [70, 85], "color": "yellow"}, {"range": [85, 100], "color": "green"}]}))
        st.plotly_chart(fig_g, use_container_width=True)