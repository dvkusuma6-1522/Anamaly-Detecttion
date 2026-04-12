import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import bcrypt

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")

# ================== 🎨 GLOBAL UI ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
    color: white;
}

/* Split login container */
.login-container {
    display: flex;
    height: 80vh;
    border-radius: 20px;
    overflow: hidden;
}

/* Left panel */
.left-panel {
    flex: 1;
    background: linear-gradient(135deg, #6a5acd, #ff7eb3);
    padding: 40px;
    color: white;
}

/* Right panel */
.right-panel {
    flex: 1;
    background: #111;
    padding: 40px;
}

/* Input fields */
input {
    border-radius: 10px !important;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# ================== 🔐 SESSION ==================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================== 🔐 USERS ==================
users = {
    "admin": bcrypt.hashpw("1234".encode(), bcrypt.gensalt())
}

# ================== 🔐 LOGIN ==================
def login():
    col1, col2 = st.columns(2)

    # LEFT SIDE (WELCOME)
    with col1:
        st.markdown("""
        <div style='padding:40px;
                    background:linear-gradient(135deg,#6a5acd,#ff7eb3);
                    border-radius:20px;
                    height:500px;
                    color:white;'>
            <h1>Welcome Back 👋</h1>
            <p style='font-size:16px'>
            Access your AI-powered anomaly dashboard.<br><br>
            Monitor trends, detect anomalies, and gain insights in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT SIDE (LOGIN FORM)
    with col2:
        st.markdown("### 🔐 User Login")

        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")

        if st.button("Login", use_container_width=True):
            if username in users:
                if bcrypt.checkpw(password.encode(), users[username]):
                    st.session_state.logged_in = True
                    st.success("Login successful ✅")
                    st.rerun()
                else:
                    st.error("Incorrect password ❌")
            else:
                st.error("User not found ❌")

# ================== 🚪 LOGOUT ==================
def logout():
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# ================== 🔐 AUTH ==================
if not st.session_state.logged_in:
    login()
    st.stop()

# ================== DASHBOARD ==================
logout()

# ================== 📂 SIDEBAR ==================
st.sidebar.title("📊 Dashboard Menu")
page = st.sidebar.radio("Navigate", ["Overview", "Anomalies", "Insights"])
st.sidebar.header("🔍 Filters")

# ================== 📊 LOAD DATA ==================
df = pd.read_csv("../data/processed_trend_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

model = joblib.load("../models/isolation_forest.pkl")

features = df[['post_count', 'likes', 'shares', 'engagement_rate']]
df['anomaly'] = model.predict(features)
df['anomaly_label'] = df['anomaly'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# ================== 🎯 FILTERS ==================
platform = st.sidebar.selectbox("Select Platform", df['platform'].unique())
platform_df = df[df['platform'] == platform]

hashtag = st.sidebar.selectbox("Select Hashtag", platform_df['hashtag'].unique())
filtered_df = platform_df[platform_df['hashtag'] == hashtag]

anomalies = filtered_df[filtered_df['anomaly_label'] == "Anomaly"]

# ================== 🎯 TITLE ==================
st.title("📊 Social Media Trend Anomaly Dashboard")
st.markdown("### 🚀 AI-powered anomaly detection using engagement metrics")

# ================== 📊 KPI CARDS ==================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='card'>📊 Total Data Points<br><h2>{len(filtered_df)}</h2></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'>🚨 Anomalies<br><h2>{len(anomalies)}</h2></div>", unsafe_allow_html=True)

with col3:
    avg_eng = round(filtered_df['engagement_rate'].mean(), 2)
    st.markdown(f"<div class='card'>📈 Avg Engagement<br><h2>{avg_eng}</h2></div>", unsafe_allow_html=True)

# ================== 🚨 ALERT ==================
if len(anomalies) > 0:
    st.error(f"🚨 {len(anomalies)} anomalies detected!")
else:
    st.success("✅ No anomalies detected")

# ================== 🔥 TOP ANOMALY ==================
if len(anomalies) > 0:
    top = anomalies.iloc[0]
    st.warning(f"🔥 Biggest anomaly on {top['date'].date()} with {top['post_count']} posts")

# ================== 📈 GRAPH ==================
st.subheader(f"📈 Trend for #{hashtag} on {platform}")

fig = px.line(filtered_df, x='date', y='post_count')

fig.add_scatter(
    x=anomalies['date'],
    y=anomalies['post_count'],
    mode='markers',
    name='Anomalies'
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ================== 🤖 AI INSIGHTS ==================
if page == "Insights":
    st.subheader("🤖 AI Insights")

    if len(anomalies) > 0:
        st.info(f"Spike detected for #{hashtag}. Likely viral or campaign-driven trend.")
    else:
        st.success("No unusual behavior detected. Trends are stable.")

# ================== 🧠 EXPLANATION ==================
def explain(row):
    if row['post_count'] > filtered_df['post_count'].mean() * 2:
        return "📈 Sudden spike in posts"
    elif row['engagement_rate'] > 0.8:
        return "🔥 High engagement anomaly"
    elif 'z_score' in row and abs(row['z_score']) > 3:
        return "📊 Statistical outlier"
    return "Normal"

anomalies = anomalies.copy()
anomalies['reason'] = anomalies.apply(explain, axis=1)

# ================== 📂 PAGE LOGIC ==================
if page == "Overview":
    st.subheader("📋 Data Overview")
    st.dataframe(filtered_df, use_container_width=True)

elif page == "Anomalies":
    st.subheader("🚨 Detected Anomalies with Explanation")

    display_cols = [
        'date', 'platform', 'hashtag',
        'post_count', 'likes', 'shares',
        'engagement_rate', 'reason'
    ]

    st.dataframe(
        anomalies[display_cols].sort_values('date', ascending=False),
        use_container_width=True
    )

    st.download_button(
        label="⬇ Download Anomalies CSV",
        data=anomalies.to_csv(index=False),
        file_name="anomalies.csv",
        mime="text/csv"
    )

# ================== 📂 RAW DATA ==================
with st.expander("📂 View Full Data"):
    st.dataframe(filtered_df, use_container_width=True)
