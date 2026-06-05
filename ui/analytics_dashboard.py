import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3, os

st.set_page_config(page_title="📊 RefurbCortex Analytics", layout="wide")
st.title("📊 Pilot Analytics Dashboard")

db_path = "./data/feedback.db"
if not os.path.exists(db_path):
    st.warning("📉 No feedback data yet. Run inspections & submit feedback to populate dashboard.")
    st.stop()

df = pd.read_sql_query("SELECT * FROM feedback_logs", sqlite3.connect(db_path))
df["logged_at"] = pd.to_datetime(df["logged_at"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Inspections", len(df))
col2.metric("📈 Median Error Rate", f"{df['error_rate'].median()*100:.1f}%")
col3.metric("💸 Avg Margin Impact", f"{df['absolute_error'].mean():,.0f} ₹")
col4.metric("🔄 Human Overrides", f"{df['human_override'].sum()}")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("📉 Error Rate Over Time")
    fig1 = px.line(df.sort_values("logged_at"), x="logged_at", y="error_rate", markers=True)
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("🎯 Error Distribution")
    fig2 = px.histogram(df, x="error_rate", nbins=20, title="Prediction Error Spread")
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

st.dataframe(df[["inspection_id", "predicted_cost", "actual_cost", "error_rate", "human_override", "logged_at"]])