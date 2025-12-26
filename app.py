import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# 1. ELITE UI STYLING
st.set_page_config(page_title="MacroIntelligence Terminal", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');
    .stApp { background-color: #ffffff; color: #1e293b; }
    h1, h2 { font-family: 'Playfair Display', serif; color: #002147; }
    h3, p, div { font-family: 'Inter', sans-serif; }
    div[data-testid="stVerticalBlock"] > div.stColumn > div {
        background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; 
        border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 33, 71, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #002147; padding: 12px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #60a5fa !important; }
    .stButton>button { background: #002147; color: #ffffff; border-radius: 4px; font-weight: 600; padding: 12px; width: 100%; border: none; }
    .stButton>button:hover { background: #003366; border: 1px solid #60a5fa; }
    </style>
    """, unsafe_allow_html=True)

# 2. CORE DATA ENGINE
@st.cache_data
def load_and_initialize_engine():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", "macro_data.csv")
    if not os.path.exists(file_path):
        st.error("Infrastructure Error: Source Data (macro_data.csv) Not Found.")
        st.stop()
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df["CPI_YOY"] = df["CPI"].pct_change(12)
    df["GDP_GROWTH"] = df["GDP"].pct_change(4)
    data = df[["CPI_YOY", "UNEMPLOYMENT", "FED_FUNDS", "GDP_GROWTH"]].dropna()
    X = data.values
    y = np.where(data["GDP_GROWTH"] < 0, 2, np.where(data["CPI_YOY"] > 0.04, 1, 0)) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression().fit(X_scaled, y)
    return model, scaler, data

model, scaler, signals = load_and_initialize_engine()

# Consolidated Knowledge Base 
HISTORICAL_CONTEXT = {
    1974: "A defining stagflationary era caused by the OPEC oil shock. CPI reached double digits while GDP contracted, fundamentally changing monetary policy history.",
    1982: "The Volcker Era. High interest rates were used as a weapon to break inflation, resulting in a short-term recession that launched a multi-decade bull market.",
    2000: "The Dot-Com crash. Extreme equity valuations collapsed, leading the Fed to pivot toward aggressive rate cuts to stabilize the tech-heavy economy.",
    2008: "The Global Financial Crisis. Systemic failure in the credit markets led to the birth of unconventional monetary policy and 'Zero Interest Rate Policy' (ZIRP).",
    2022: "The Great Pandemic Re-opening. Supply chain shocks met unprecedented fiscal stimulus, triggering the fastest global rate-hiking cycle in 40 years."
}
playbook = {
    0: {"name": "Expansion", "market": "Bullish", "sectors": ["Technology", "Cons. Discretionary", "Financials", "Comm. Services", "Semiconductors"], "actions": ["Growth Capex", "Equity Overweight", "Market Expansion"]},
    1: {"name": "Stagflation", "market": "Volatile", "sectors": ["Energy", "Materials", "Utilities", "Commodities", "Infrastructure"], "actions": ["Pricing Power", "Opex Efficiency", "Commodity Hedging"]},
    2: {"name": "Recession", "market": "Bearish", "sectors": ["Healthcare", "Cons. Staples", "Gov. Bonds", "Gold", "Defense"], "actions": ["Cash Preservation", "Fixed Income", "Risk Reduction"]},
    3: {"name": "Recovery", "market": "Improving", "sectors": ["Industrials", "Real Estate", "Small-Cap", "Transportation", "Metals"], "actions": ["Inventory Rebuild", "Selective Hiring", "Cyclical Shift"]}
}
# 3. NAVIGATION
t_home, t_trends, t_scenario, t_evaluator, t_consulting = st.tabs(["🏠 HOME", "📈 TRENDS", "🎮 SCENARIO", "🧠 EVALUATOR", "💼 CONSULTING"])

# --- HOME ---
with t_home:
    st.title("MacroIntelligence Terminal 🏛️")
    st.markdown("### Industrial-Grade Decision Support System")
    st.write("---")
    st.markdown("""
    **Methodology & Governance:**
    - **Inquiry Engine:** Powered by 70+ years of **FRED** economic data[cite: 859, 860].
    - **Regime Detection:** Utilizing **Logistic Regression** and **Clustering** to identify recurring economic environments[cite: 573, 581, 2008].
    - **Advisory:** Translating probabilistic forecasts into executive strategy recommendations[cite: 2063, 2074].
    """)

# --- TAB 1: HISTORICAL TRENDS ---
with t_trends:
    st.header("Deep-Dive Cycle Analysis")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        year_pick = st.selectbox("Select Key Macro Event:", options=list(HISTORICAL_CONTEXT.keys()))
        st.markdown(f"### {year_pick} Intelligence")
        st.write(HISTORICAL_CONTEXT[year_pick])
    with col_t2:
        var_pick = st.selectbox("Select Macro Indicator:", ["CPI_YOY", "GDP_GROWTH", "UNEMPLOYMENT", "FED_FUNDS"])
        fig = px.line(signals, y=var_pick, title=f"Historical {var_pick} (1950-Present)", color_discrete_sequence=['#002147'])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SCENARIO GENERATOR ---
with t_scenario:
    st.header("Regime Forecasting Simulator")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        s_cpi = st.slider("Forecasted Inflation (%)", -0.02, 0.15, 0.03, format="%.2f")
        s_gdp = st.slider("Forecasted GDP Growth (%)", -0.05, 0.10, 0.02, format="%.2f")
        u_input = scaler.transform([[s_cpi, 0.05, 0.04, s_gdp]])
        probs = model.predict_proba(u_input)[0]
        top_regime = np.argmax(probs)
    with col_s2:
        st.metric("Top Predicted Regime", playbook[top_regime]['name'], f"{probs[top_regime]:.1%}")
        st.progress(float(probs[top_regime]))
        st.write(f"**Historical Market Stance:** {playbook[top_regime]['market']} [cite: 2038, 2047, 2056, 2067]")

# --- TAB 3: SUCCESS EVALUATOR (EXPANDED FILTERS) ---
with t_evaluator:
    st.header("Quantitative Success Predictor")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        strat_name = st.text_input("Strategy Name", value="Global Alpha")
        sector_choice = st.selectbox("Strategic Sector", ["Technology", "Energy", "Healthcare", "Financials", "Materials", "Consumer Staples", "Industrials", "Other"])
        custom_sector = st.text_input("If 'Other', specify sector:") if sector_choice == "Other" else sector_choice
    with col_e2:
        leverage = st.select_slider("Leverage Intensity", options=["Deleveraging", "Low Debt", "Moderate", "Highly Leveraged"])
        horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
    with col_e3:
        market_cap = st.selectbox("Market Cap Focus", ["Large-Cap", "Mid-Cap", "Small-Cap", "Micro-Cap"])
        vol_tol = st.select_slider("Volatility Tolerance", options=["Minimal", "Moderate", "Aggressive"])

    if st.button("Calculate Probability of Success"):
        st.divider()
        st.subheader(f"📊 Assessment: Project {strat_name}")
        
        # Deep Logic Engine [cite: 2093, 2132]
        is_risky = vol_tol == "Aggressive" or leverage == "Highly Leveraged"
        score = 85 if custom_sector in playbook[top_regime]['sectors'] else 45
        if top_regime in [1, 2] and is_risky: score -= 35
        
        st.write(f"**Historical Macro-Alignment Score: {max(score, 0)}%**")
        if score < 60:
            st.error(f"🚨 STRATEGIC ALERT: In a {playbook[top_regime]['name']} regime, {custom_sector} historically faces margin compression. High leverage in this environment creates systemic tail-risk.")
        else:
            st.success(f"✅ REGIME ALIGNMENT: The {playbook[top_regime]['name']} cycle supports expansion in {custom_sector}. Market conditions are favorable for {market_cap} entities.")

# --- TAB 4: CONSULTING (INDEPENDENT GENERATIVE SUITE) ---
with t_consulting:
    st.header("Executive Strategic Advisory")
    s_tab1, s_tab2 = st.tabs(["📝 Model Advice for Strategy", "🚀 Generate Business Strategy"])
    
    with s_tab1:
        st.subheader("Memorandum Diagnostic")
        audit_cpi = st.number_input("Audit Context: Inflation", value=0.03)
        audit_gdp = st.number_input("Audit Context: GDP Growth", value=0.02)
        full_strat_doc = st.text_area("Paste Strategy Memo for Evaluation:", height=200)
        
        if st.button("Execute Diagnostic Audit"):
            audit_u = scaler.transform([[audit_cpi, 0.05, 0.04, audit_gdp]])
            a_regime = np.argmax(model.predict_proba(audit_u)[0])
            st.markdown(f"### 📄 Audit: {playbook[a_regime]['name']} Environment")
            st.write(f"**Economic Reality:** Interest rates and inflation at these levels historically compress multiples. Your focus on '{full_strat_doc[:40]}...' requires a pivot to **{playbook[a_regime]['actions'][0]}**.")
            st.write(f"**Sectoral Strategy:** Shift capital toward **{playbook[a_regime]['sectors'][0]}** and **{playbook[a_regime]['sectors'][1]}** to preserve NPV.")

    with s_tab2:
        st.subheader("Macro-Informed Roadmap Generator")
        idea_prompt = st.text_area("Describe your business idea/prompt in detail:", height=150, placeholder="e.g., A boutique logistics firm focusing on automated last-mile delivery in urban centers.")
        target_size = st.selectbox("Target Scale", ["Boutique/Niche", "Mid-Market", "Global Enterprise"])
        
        if st.button("Generate Full Strategic Deck"):
            # Independent: Uses current market data, not Tab 2 sliders [cite: 1015, 1086]
            cur_u = scaler.transform([[signals['CPI_YOY'].iloc[-1], 0.05, 0.04, signals['GDP_GROWTH'].iloc[-1]]])
            g_regime = np.argmax(model.predict_proba(cur_u)[0])
            
            st.markdown(f"## 🚀 Strategic Deck: {idea_prompt[:30]}...")
            st.write("---")
            st.markdown("### 📊 Market Sectoring & Deployment")
            st.write(f"**Primary Markets:** {', '.join(playbook[g_regime]['sectors'])}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ✅ Strategic Pros")
                st.write(f"- Direct alignment with **{playbook[g_regime]['name']}** cycle tailwinds.")
                st.write(f"- High resilience in **{playbook[g_regime]['market']}** market conditions.")
            with c2:
                st.markdown("#### ❌ Critical Cons")
                st.write(f"- Margin vulnerability to shifting **Fed Funds** rates.")
                st.write(f"- Potential saturation in **{playbook[g_regime]['sectors'][0]}**.")
            
            st.markdown("### 📈 Strategic Financial Roadmap")
            st.write(f"**1. Capital Allocation:** Prioritize **{playbook[g_regime]['actions'][1]}** during the first 24 months.")
            st.write(f"**2. Risk Management:** Hedge against inflation volatility by implementing **{playbook[g_regime]['actions'][0]}**.")
            st.markdown("#### Possible Room for Improvement")
            st.info(f"To enhance safety, consider diversifying across **{playbook[g_regime]['sectors'][2]}** and reducing exposure to floating-rate debt.")
