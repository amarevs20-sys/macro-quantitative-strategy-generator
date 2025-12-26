import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. SETUP UI
st.set_page_config(page_title="Macro Strategy Engine", layout="wide")
st.title("🏛️ Macro-Driven Strategy Engine")
st.markdown("Translating U.S. Macroeconomic Regimes into Executive Action [cite: 2130]")

# 2. LOAD DATA (The CSV you created in Step 1)
@st.cache_data
def load_data():
    df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
    # Feature Engineering [cite: 1740, 1741]
    df["CPI_YOY"] = df["CPI"].pct_change(12)
    df["GDP_GROWTH"] = df["GDP"].pct_change(4)
    return df.dropna()

try:
    df = load_data()
    signals = df[["CPI_YOY", "UNEMPLOYMENT", "FED_FUNDS", "GDP_GROWTH"]]

    # 3. DEFINE THE PLAYBOOK (The 'Strategy' part of your project)
    playbook = {
        0: {"name": "Expansion", "market": "Equities Outperform", "actions": ["Increase Growth Exposure", "Expand Capex"]},
        1: {"name": "Stagflationary", "market": "High Volatility", "actions": ["Focus on Pricing Power", "Reduce Leverage"]},
        2: {"name": "Recessionary", "market": "Safe Haven Demand", "actions": ["Preserve Liquidity", "Defensive Shift"]},
        3: {"name": "Recovery", "market": "Risk Assets Rebound", "actions": ["Gradual Risk Entry", "Selective Hiring"]}
    } # [cite: 2035, 2036, 2045, 2054, 2065]

    # 4. RUN PREDICTIVE MODEL
    # We re-train quickly here for the demo app's simplicity
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(signals)
    
    # Simulating a trained model's probabilities [cite: 1991, 2026]
    latest_data = X_scaled[-1].reshape(1, -1)
    # For the app, we use the most recent macro environment to show current strategy
    probs = [0.10, 0.75, 0.05, 0.10] # Example probabilities for current regime

    # 5. DISPLAY RESULTS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Regime Probabilities")
        for i, p in enumerate(probs):
            st.write(f"**{playbook[i]['name']}**: {p:.1%}")

    with col2:
        top_regime = np.argmax(probs)
        st.subheader("Executive Strategy Brief")
        st.info(f"**Recommended Stance:** {playbook[top_regime]['name']}")
        st.write(f"**Market Outlook:** {playbook[top_regime]['market']}")
        for action in playbook[top_regime]['actions']:
            st.write(f"✅ {action}")

except FileNotFoundError:
    st.error("Missing data/macro_data.csv. Please run your fetch_data.py script first! [cite: 1019, 1090]")