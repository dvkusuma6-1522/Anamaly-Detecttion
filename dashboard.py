import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed_trend_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Load model
model = joblib.load("models/isolation_forest.pkl")

# Features
features = df[['post_count', 'likes', 'shares', 'engagement_rate']]

# Predictions
df['anomaly'] = model.predict(features)
df['anomaly_label'] = df['anomaly'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# UI
st.title("📊 Social Media Trend Anomaly Dashboard")
st.markdown("### 🚀 AI-powered anomaly detection using engagement metrics")

# Filters
platform = st.selectbox("Select Platform", df['platform'].unique())
platform_df = df[df['platform'] == platform]

hashtag = st.selectbox("Select Hashtag", platform_df['hashtag'].unique())
filtered_df = platform_df[platform_df['hashtag'] == hashtag]

# Anomalies
anomalies = filtered_df[filtered_df['anomaly_label'] == "Anomaly"]

# Metrics
col1, col2 = st.columns(2)
col1.metric("Total Data Points", len(filtered_df))
col2.metric("🚨 Anomalies", len(anomalies))

# 🚨 ALERT BOX
if len(anomalies) > 0:
    st.warning(f"🚨 {len(anomalies)} anomalies detected!")
else:
    st.success("✅ No anomalies detected")

# 📈 Dynamic Title
st.subheader(f"📈 Trend for {hashtag} on {platform}")

# Graph
plt.figure(figsize=(10,5))
plt.plot(filtered_df['date'], filtered_df['post_count'], label="Trend")

# Highlight anomalies
plt.scatter(anomalies['date'], anomalies['post_count'], marker='o')

plt.legend()
st.pyplot(plt)

# 🤖 Explanation Function
def explain(row):
    if row['post_count'] > filtered_df['post_count'].mean() * 2:
        return "📈 Sudden spike in posts"
    elif row['engagement_rate'] > 0.8:
        return "🔥 High engagement anomaly"
    elif abs(row['z_score']) > 3:
        return "📊 Statistical outlier"
    return "Normal"

# Apply explanation
anomalies = anomalies.copy()
anomalies['reason'] = anomalies.apply(explain, axis=1)

# Table
st.subheader("🚨 Detected Anomalies with Explanation")
st.dataframe(anomalies)