import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import re

# 1. VIVID INSTITUTIONAL UI (Full Spectrum & Bold White Labels)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="MacroIntelligence Terminal", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Playfair+Display:wght@700&display=swap');
    
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* Force all labels and instructions to be Bold White for maximum clarity */
    label, .stMarkdown p, .stSlider label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #ffffff !important; font-weight: 800 !important; font-size: 1.2rem !important;
    }

    h1 { font-family: 'Inter', sans-serif; font-weight: 900; font-size: 4.5rem !important; color: #38bdf8; letter-spacing: -3px; margin-bottom: 0px; }
    h2 { font-family: 'Playfair Display', serif; font-size: 3.5rem !important; color: #fbbf24; border-bottom: 5px solid #f472b6; padding-bottom: 15px; }
    h3 { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2.2rem !important; color: #34d399; margin-top: 20px; }
    
    /* Institutional Cards */
    div[data-testid="stVerticalBlock"] > div.stColumn > div {
        background-color: #0f172a; border: 2px solid #1e293b; padding: 50px; 
        border-radius: 25px; box-shadow: 0 10px 40px rgba(56, 189, 248, 0.3);
        margin-bottom: 30px;
    }
    
    /* Massive 26px Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #001f3f; padding: 25px; border-radius: 15px; gap: 25px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-size: 26px !important; font-weight: 900 !important; text-transform: uppercase; letter-spacing: 1px; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 8px solid #8b5cf6 !important; }

    .stButton>button { 
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%); 
        color: white; font-size: 2rem; font-weight: 900; padding: 35px; border-radius: 15px; border: none; width: 100%;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CORE DATA & ANALYTICS ENGINE
# ------------------------------------------------------------------------------
@st.cache_data
def load_terminal_engine():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "macro_data.csv")
    if not os.path.exists(file_path):
        st.error("Missing data/macro_data.csv. Please ensure files are correctly uploaded.")
        st.stop()
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df["CPI_YOY"] = df["CPI"].pct_change(12)
    df["GDP_GROWTH"] = df["GDP"].pct_change(4)
    data = df[["CPI_YOY", "UNEMPLOYMENT", "FED_FUNDS", "GDP_GROWTH"]].dropna()
    X, y = data.values, np.where(data["GDP_GROWTH"] < 0, 2, np.where(data["CPI_YOY"] > 0.04, 1, 0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression().fit(X_scaled, y)
    return model, scaler, data

model, scaler, signals = load_terminal_engine()

# Factually Accurate Historical Context
HIST_CONTEXT = {
    1974: "**1974 STAGFLATION: Oil shocks & double-digit CPI. Growth plummeted, fundamentally changing monetary history.**",
    1982: "**1982 VOLCKER PIVOT: Fed rates hit 20% to break inflation, causing a deep recession but resetting global capital costs.**",
    2000: "**2000 DOT-COM COLLAPSE: Tech sector valuation disconnect leading to a massive sector rotation and a 2-year bear market.**",
    2008: "**2008 FINANCIAL CRISIS: Global banking failure led to the Great Recession and the implementation of Zero Interest Rate Policy.**",
    2022: "**2022 POST-COVID RESET: Stimulus met supply breaks causing rapid inflation and the most aggressive rate hike cycle in decades.**"
}

playbook = {
    0: {"name": "Expansion", "color": "#34d399", "drag": 0.0},
    1: {"name": "Stagflation", "color": "#facc15", "drag": 7.5},
    2: {"name": "Recession", "color": "#f87171", "drag": 15.0}
}

# 3. INTERFACE NAVIGATION
# ------------------------------------------------------------------------------
t_home, t_trends, t_scenario, t_eval, t_consult = st.tabs(["🏠 HOME", "📈 TRENDS", "🎮 SCENARIO", "🧠 EVALUATOR", "💼 CONSULTING"])

# --- TAB: HOME ---
with t_home:
    st.title("MacroIntelligence Terminal")
    st.write("---")
    st.header("System Overview & Integrated Functionality")
    st.write("**INSTRUCTION: This terminal is a recursive system where each tab informs the final computational output.**")
    
    st.markdown("""
    ### **1. Core Functionality Workflow**
    * **TRENDS:** Analyze 70+ years of historical indicators to understand the cyclical nature of macro regimes.
    * **SCENARIO:** Forecasted inflation and growth determine the **Active Macro Regime**, which applies mathematical "Drag."
    * **EVALUATOR:** Qualitative business profile (Sector, Leverage, Horizon, Cap Focus, Volatility) audit.
    * **CONSULTING:** Final integration layer running **Monte Carlo Simulations** specific to targets and organizational scale.

    ### **2. Integrated Methodology**
    This engine ingests 70+ years of FRED data to bridge the execution gap between macro-volatility and corporate strategy.
    """)

# --- TAB: TRENDS ---
with t_trends:
    st.header("Deep-Dive Cycle Analysis")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.write("**INSTRUCTION: Select a historical era to view regime intelligence.**")
        era = st.selectbox("Select Key Macro Event:", list(HIST_CONTEXT.keys()))
        st.markdown(f"<div style='background-color:#1e293b; padding:20px; border-radius:10px;'>{HIST_CONTEXT[era]}</div>", unsafe_allow_html=True)
    with col_t2:
        st.write("**INSTRUCTION: View 70-year timeline for specific macro metrics.**")
        var = st.selectbox("Select Macro Indicator:", ["CPI_YOY", "GDP_GROWTH", "UNEMPLOYMENT", "FED_FUNDS"])
        st.plotly_chart(px.line(signals, y=var, color_discrete_sequence=['#38bdf8'], template="plotly_dark"), use_container_width=True)

# --- TAB: SCENARIO ---
with t_scenario:
    st.header("Global Regime Simulator")
    s1, s2 = st.columns([1, 2])
    with s1:
        st.write("**INSTRUCTION: Adjust sliders to simulate future economic conditions.**")
        inf = st.slider("Inflation Scenario (%)", -0.02, 0.15, 0.03)
        gro = st.slider("GDP Growth Scenario (%)", -0.05, 0.10, 0.02)
        u_in = scaler.transform([[inf, 0.05, 0.04, gro]])
        reg_id = np.argmax(model.predict_proba(u_in)[0])
    with s2:
        st.markdown(f"<h1 style='color:{playbook[reg_id]['color']}'>{playbook[reg_id]['name']}</h1>", unsafe_allow_html=True)
        st.progress(0.9)

# --- TAB: EVALUATOR (FULL FILTER SUITE RESTORED) ---
with t_eval:
    st.header("Quantitative Success Predictor")
    st.write("**INSTRUCTION: Input strategy variables for initial rule-based alignment check.**")
    e1, e2, e3 = st.columns(3)
    with e1:
        s_name = st.text_input("Strategy Name", "Global Alpha")
        s_sector = st.selectbox("Strategic Sector", ["Tech", "Energy", "Healthcare", "Finance", "Other"])
    with e2:
        s_debt = st.select_slider("Leverage Intensity", ["Deleveraging", "Low Debt", "Moderate", "High"])
        s_hor = st.selectbox("Investment Horizon", ["Short-term", "Long-term"])
    with e3:
        s_cap = st.selectbox("Market Cap Focus", ["Large-Cap", "Mid-Cap", "Small-Cap"])
        s_vol = st.select_slider("Volatility Tolerance", ["Minimal", "Moderate", "Aggressive"])
    
    if st.button("Calculate Probability of Success"):
        st.divider()
        st.subheader(f"📊 Diagnostic Audit: {s_name}")
        st.write(f"In a **{playbook[reg_id]['name']}** regime, this **{s_cap}** strategy faces { 'positive cyclical tailwinds' if reg_id == 0 else 'significant valuation compression' }.")

# --- TAB: CONSULTING (MONTE CARLO COMPUTATIONAL ENGINE) ---
with t_consult:
    st.header("Institutional Stress-Test Terminal")
    st.write("**INSTRUCTION: Enter targets (e.g., '15% Growth') for Monte Carlo Audit.**")
    
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        memo = st.text_area("Targets Prompt:", placeholder="e.g. Projecting 15% ROI...", height=200)
        num_match = re.findall(r"[-+]?\d*\.\d+|\d+", memo)
        target_roi = float(num_match[0]) if num_match else 10.0
        
        st.write(f"**DECODED TARGET ROI:** {target_roi}%")
        user_vol = st.slider("Project Volatility (%)", 5, 50, 20)
        scale_penalty = {"Boutique": 1.0, "Mid-Market": 4.0, "Global Enterprise": 10.0}
        biz_scale = st.selectbox("Organizational Scale:", list(scale_penalty.keys()))
        
        defense_toggle = st.toggle("Apply Recession Defense (Cut overhead by 10%)")
        defense_bonus = 10.0 if defense_toggle else 0.0
        
        run_sim = st.button("🚀 EXECUTE MONTE CARLO SIMULATION")

    with col_c2:
        if run_sim:
            # ROI Probability Engine
            # Mean ROI = Target - Regime Drag - Scale Risk + Defense Bonus
            mu = target_roi - playbook[reg_id]['drag'] - scale_penalty[biz_scale] + defense_bonus
            sim_results = np.random.normal(mu, user_vol, 1000)
            
            # SPECTRUM HISTOGRAM
            fig = go.Figure(data=[go.Histogram(
                x=sim_results, 
                marker=dict(color=sim_results, colorscale='Viridis', showscale=True),
                nbinsx=40
            )])
            fig.update_layout(title=f"ROI Distribution: {playbook[reg_id]['name']} Regime", 
                              template="plotly_dark", xaxis_title="Simulated ROI (%)")
            fig.add_vline(x=target_roi, line_dash="dash", line_color="#ef4444", 
                          annotation_text=f"Target: {target_roi}%")
            st.plotly_chart(fig, use_container_width=True)
            
            success_rate = (len([x for x in sim_results if x >= target_roi]) / 1000) * 100
            st.write(f"### **Computational Success Probability: {success_rate:.1f}%**")
            st.write(f"Audit of **'{memo[:40]}...'** identifies macro-drag of {playbook[reg_id]['drag']}% in this cycle.")
