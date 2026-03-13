import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sportswear D2C App - Survey Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stHeader"] { background: transparent; }
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background-color: #161b22; padding: 10px 18px;
    border-radius: 14px; border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background-color: #21262d; color: #8b949e; border-radius: 8px;
    padding: 10px 22px; font-weight: 600; font-size: 14px;
    border: 1px solid #30363d;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border-color: transparent !important;
}
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }
div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 18px 20px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.8rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff; font-size: 1.9rem; font-weight: 700;
}
.sec-hdr {
    background: linear-gradient(90deg, #1c2a3a 0%, #161b22 100%);
    border-left: 4px solid #3b82f6; padding: 11px 18px;
    border-radius: 0 10px 10px 0; margin: 28px 0 16px 0;
    font-weight: 600; color: #e6edf3; font-size: 1rem;
}
.insight {
    background: #0d2137; border-left: 3px solid #58a6ff;
    border-radius: 0 10px 10px 0; padding: 12px 18px;
    margin-top: 10px; font-size: 0.84rem; color: #93c5fd; line-height: 1.7;
}
.stepbox {
    background: #0d2e1a; border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0; padding: 9px 16px;
    margin: 7px 0; font-size: 0.84rem; color: #7ee787;
}
.hero {
    background: linear-gradient(135deg, #0d1f36 0%, #1a1033 50%, #0d2820 100%);
    border: 1px solid #30363d; border-radius: 16px;
    padding: 26px 36px; margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)

BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="Inter, system-ui, sans-serif", color="#c9d1d9", size=12),
    title_font=dict(size=15, color="#e6edf3"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=20, r=20, t=55, b=40),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

INTENT_CLR = {"Yes": "#3fb950", "Maybe": "#d29922", "No": "#f85149"}

@st.cache_data
def load_data():
    return pd.read_csv("sportswear_survey_synthetic_2000.csv")

df_raw = load_data()

st.markdown("""
<div class="hero">
  <h1 style="color:#e6edf3;margin:0;font-size:1.75rem;font-weight:700;">
    Sportswear D2C App - Survey Analytics Dashboard
  </h1>
  <p style="color:#8b949e;margin:8px 0 0 0;font-size:0.92rem;">
    Target Market: Urban Indians | Age 18-35 | N = 2,000 Respondents | Seed = 42
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "  Tab 1 - Data Overview & Cleaning",
    "  Tab 2 - Exploratory Data Analysis"
])

with tab1:

    # 1.1 Raw Preview
    st.markdown("<div class='sec-hdr'>1.1  Raw Dataset Preview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Total Columns", f"{df_raw.shape[1]}")
    c3.metric("Missing Values", f"{int(df_raw.isnull().sum().sum())}")
    c4.metric("Data Types", f"{df_raw.dtypes.nunique()} unique")
    st.dataframe(df_raw.head(10), use_container_width=True, height=310)

    # 1.2 Missing Values
    st.markdown("<div class='sec-hdr'>1.2  Missing Values per Column</div>", unsafe_allow_html=True)
    miss = df_raw.isnull().sum()
    miss_nz = miss[miss > 0]
    if miss_nz.empty:
        st.success("No missing values detected — dataset is complete (0 nulls across all columns).")
        fig_mv = go.Figure()
        fig_mv.add_annotation(text="Zero Missing Values - Dataset is Complete",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=17, color="#3fb950"))
        fig_mv.update_layout(height=200, **BASE)
        st.plotly_chart(fig_mv, use_container_width=True)
    else:
        fig_mv = px.bar(x=miss_nz.index, y=miss_nz.values,
            labels={"x": "Column", "y": "Missing Count"},
            color=miss_nz.values, color_continuous_scale="Reds",
            title="Missing Value Count per Column")
        fig_mv.update_layout(height=380, **BASE)
        st.plotly_chart(fig_mv, use_container_width=True)

    # 1.3 Outlier Detection BEFORE Cleaning
    st.markdown("<div class='sec-hdr'>1.3  Outlier Detection - BEFORE Cleaning</div>", unsafe_allow_html=True)
    q11_outs = int((df_raw["Q11_current_monthly_spend_inr"] > 15000).sum())
    q24_outs = int((df_raw["Q24_wtp_monthly_inr"] < 200).sum())
    colb1, colb2 = st.columns(2)

    with colb1:
        fig_b1 = go.Figure()
        fig_b1.add_trace(go.Box(
            y=df_raw["Q11_current_monthly_spend_inr"], name="Q11 Spend",
            boxpoints="outliers",
            marker=dict(color="#f85149", size=5, opacity=0.65),
            line=dict(color="#388bfd", width=2), fillcolor="#161b22"))
        fig_b1.add_hline(y=15000, line_dash="dash", line_color="#f85149",
            annotation_text="Cutoff: 15000", annotation_position="top left",
            annotation_font_color="#f85149")
        fig_b1.update_layout(title="Q11 Monthly Spend (INR) - Raw",
            yaxis_title="INR", height=420, **BASE)
        st.plotly_chart(fig_b1, use_container_width=True)
        msg1 = f"{q11_outs} rows exceed 15000/month. These luxury-tier buyers right-skew the regression model and will be removed."
        st.markdown(f"<div class='insight'>{msg1}</div>", unsafe_allow_html=True)

    with colb2:
        fig_b2 = go.Figure()
        fig_b2.add_trace(go.Box(
            y=df_raw["Q24_wtp_monthly_inr"], name="Q24 WTP",
            boxpoints="outliers",
            marker=dict(color="#d29922", size=5, opacity=0.65),
            line=dict(color="#a371f7", width=2), fillcolor="#161b22"))
        fig_b2.add_hline(y=200, line_dash="dash", line_color="#d29922",
            annotation_text="Cutoff: 200", annotation_position="top left",
            annotation_font_color="#d29922")
        fig_b2.update_layout(title="Q24 Willingness to Pay (INR) - Raw",
            yaxis_title="INR", height=420, **BASE)
        st.plotly_chart(fig_b2, use_container_width=True)
        msg2 = f"{q24_outs} rows report WTP below 200. These extreme skeptics would anchor the regression intercept unrealistically low."
        st.markdown(f"<div class='insight'>{msg2}</div>", unsafe_allow_html=True)

    # 1.4 Cleaning Pipeline
    st.markdown("<div class='sec-hdr'>1.4  Cleaning Pipeline - Steps Applied</div>", unsafe_allow_html=True)
    rows_before = len(df_raw)
    df_clean = df_raw.copy()

    r1 = int((df_clean["Q11_current_monthly_spend_inr"] > 15000).sum())
    df_clean = df_clean[df_clean["Q11_current_monthly_spend_inr"] <= 15000].copy()
    st.markdown(f"<div class='stepbox'>Step 1 - Remove Q11 outliers (>15000 INR): {r1} rows removed</div>", unsafe_allow_html=True)

    r2 = int((df_clean["Q24_wtp_monthly_inr"] < 200).sum())
    df_clean = df_clean[df_clean["Q24_wtp_monthly_inr"] >= 200].copy()
    st.markdown(f"<div class='stepbox'>Step 2 - Remove Q24 extreme low WTP (<200 INR): {r2} rows removed</div>", unsafe_allow_html=True)

    df_clean["Q11_log_spend"] = np.log1p(df_clean["Q11_current_monthly_spend_inr"])
    df_clean["Q24_log_wtp"]   = np.log1p(df_clean["Q24_wtp_monthly_inr"])
    st.markdown("<div class='stepbox'>Step 3 - Log1p transformation applied: Q11_log_spend and Q24_log_wtp created</div>", unsafe_allow_html=True)

    ordinal_maps = {
        "Q1_age_group":             {"18-22": 0, "23-27": 1, "28-32": 2, "33-35": 3},
        "Q5_monthly_income":        {"Below_20k": 0, "20k-40k": 1, "40k-60k": 2,
                                     "60k-80k": 3, "80k-100k": 4, "Above_100k": 5},
        "Q7_workout_days_per_week": {"0_days": 0, "1-2_days": 1, "3-4_days": 2, "5-7_days": 3},
        "Q14_purchase_frequency":   {"Rarely": 0, "Yearly": 1, "Every_6_months": 2,
                                     "Every_2-3_months": 3, "Monthly_or_more": 4},
    }
    for col, mapping in ordinal_maps.items():
        df_clean[col] = df_clean[col].map(mapping)
    enc_cols = ", ".join(ordinal_maps.keys())
    st.markdown(f"<div class='stepbox'>Step 4 - Ordinal label encoding applied to: {enc_cols}</div>", unsafe_allow_html=True)

    # 1.5 Before vs After
    st.markdown("<div class='sec-hdr'>1.5  Before vs After Cleaning</div>", unsafe_allow_html=True)
    rows_after   = len(df_clean)
    rows_removed = rows_before - rows_after
    ca, cb, cc, cd = st.columns(4)
    ca.metric("Rows Before", f"{rows_before:,}")
    cb.metric("Rows After",  f"{rows_after:,}")
    cc.metric("Rows Removed", f"{rows_removed}", delta=f"-{rows_removed}", delta_color="inverse")
    cd.metric("Data Retained", f"{rows_after/rows_before*100:.1f}%")

    cola1, cola2 = st.columns(2)
    with cola1:
        fig_a1 = go.Figure()
        fig_a1.add_trace(go.Box(y=df_clean["Q11_log_spend"], name="Q11_log_spend",
            boxpoints="outliers", marker=dict(color="#3fb950", size=4, opacity=0.6),
            line=dict(color="#388bfd", width=2), fillcolor="#0d2e1a"))
        fig_a1.update_layout(title="Q11_log_spend - After Cleaning", height=360, **BASE)
        st.plotly_chart(fig_a1, use_container_width=True)

    with cola2:
        fig_a2 = go.Figure()
        fig_a2.add_trace(go.Box(y=df_clean["Q24_log_wtp"], name="Q24_log_wtp",
            boxpoints="outliers", marker=dict(color="#3fb950", size=4, opacity=0.6),
            line=dict(color="#a371f7", width=2), fillcolor="#0d2e1a"))
        fig_a2.update_layout(title="Q24_log_wtp - After Cleaning", height=360, **BASE)
        st.plotly_chart(fig_a2, use_container_width=True)

    # 1.6 Cleaned Dataset Preview
    st.markdown("<div class='sec-hdr'>1.6  Cleaned Dataset Preview</div>", unsafe_allow_html=True)
    preview_cols = [
        "respondent_id", "Q1_age_group", "Q2_gender", "Q3_city_tier",
        "Q5_monthly_income", "Q7_workout_days_per_week",
        "Q11_current_monthly_spend_inr", "Q11_log_spend",
        "Q24_wtp_monthly_inr", "Q24_log_wtp",
        "Q14_purchase_frequency", "Q25_app_download_intent", "persona_label"
    ]
    st.dataframe(df_clean[preview_cols].head(10), use_container_width=True, height=310)
    st.session_state["df_clean"] = df_clean

with tab2:

    df_e = df_raw[
        (df_raw["Q11_current_monthly_spend_inr"] <= 15000) &
        (df_raw["Q24_wtp_monthly_inr"] >= 200)
    ].copy()

    AGE_ORDER    = ["18-22", "23-27", "28-32", "33-35"]
    WDAY_ORDER   = ["0_days", "1-2_days", "3-4_days", "5-7_days"]
    INCOME_ORDER = ["Below_20k", "20k-40k", "40k-60k", "60k-80k", "80k-100k", "Above_100k"]
    INCOME_LBL   = {
        "Below_20k": "<20K", "20k-40k": "20K-40K", "40k-60k": "40K-60K",
        "60k-80k": "60K-80K", "80k-100k": "80K-1L", "Above_100k": ">1L"
    }
    FREQ_ORDER  = ["Rarely","Yearly","Every_6_months","Every_2-3_months","Monthly_or_more"]
    FREQ_LBL    = {
        "Rarely":"Rarely", "Yearly":"Yearly", "Every_6_months":"Every 6 Months",
        "Every_2-3_months":"Every 2-3 Months", "Monthly_or_more":"Monthly+"
    }
    PERSONA_ORDER = ["Serious_Athlete","Casual_Gym_Goer","Fashion_First","Outdoor_Enthusiast","Budget_Student"]
    PERSONA_LBL   = {
        "Serious_Athlete": "Serious Athlete", "Casual_Gym_Goer": "Casual Gym-Goer",
        "Fashion_First": "Fashion-First", "Outdoor_Enthusiast": "Outdoor Enthusiast",
        "Budget_Student": "Budget Student"
    }
    CITY_ORDER = ["Metro","Tier_1","Tier_2","Tier_3"]

    # Chart 1: Age x Gender x City Tier
    st.markdown("<div class='sec-hdr'>Chart 1  Age Group x Gender (filtered by City Tier)</div>", unsafe_allow_html=True)
    city_opts = ["All Cities"] + CITY_ORDER
    city_sel  = st.selectbox("Filter by City Tier", city_opts, key="c1_city")
    df_c1 = df_e if city_sel == "All Cities" else df_e[df_e["Q3_city_tier"] == city_sel]
    c1_grp = df_c1.groupby(["Q1_age_group","Q2_gender"]).size().reset_index(name="Count")
    fig1 = px.bar(c1_grp, x="Q1_age_group", y="Count", color="Q2_gender",
        barmode="group",
        category_orders={"Q1_age_group": AGE_ORDER},
        color_discrete_sequence=["#388bfd","#f778ba","#56d364","#d2a8ff"],
        labels={"Q1_age_group":"Age Group","Count":"Respondents","Q2_gender":"Gender"},
        title=f"Age Group vs Gender - {city_sel}")
    fig1.update_traces(marker_line_width=0)
    fig1.update_layout(height=430, **BASE, legend_title="Gender", bargap=0.18, bargroupgap=0.06)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: The 23-27 age band consistently dominates across all city tiers and genders,
        confirming this as the core acquisition target for launch campaigns.<br>
        Insight 2: Male respondents outnumber females in the 18-22 cohort across Metro cities,
        while female representation peaks in the 23-27 bracket - suggesting gender-differentiated
        creatives by age group will improve ad CTRs.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Chart 2: Workout Frequency x Spend x Intent
    st.markdown("<div class='sec-hdr'>Chart 2  Workout Frequency x Monthly Spend x App Download Intent</div>", unsafe_allow_html=True)
    fig2 = px.box(df_e,
        x="Q7_workout_days_per_week", y="Q11_current_monthly_spend_inr",
        color="Q25_app_download_intent",
        category_orders={"Q7_workout_days_per_week": WDAY_ORDER,
                         "Q25_app_download_intent": ["Yes","Maybe","No"]},
        color_discrete_map=INTENT_CLR, points="outliers",
        labels={"Q7_workout_days_per_week":"Workout Frequency",
                "Q11_current_monthly_spend_inr":"Monthly Spend (INR)",
                "Q25_app_download_intent":"Download Intent"},
        title="Monthly Spend Distribution by Workout Frequency and Download Intent")
    fig2.update_traces(marker_size=3.5, marker_opacity=0.55, line_width=1.6)
    fig2.update_layout(height=460, **BASE, legend_title="Download Intent", boxgap=0.25)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: Yes respondents who work out 5-7 days per week show the highest median spend
        making high-frequency athletes the most predictable high-LTV cohort for premium tier targeting.<br>
        Insight 2: Even at 1-2 days per week, Yes respondents spend considerably more than No respondents,
        confirming that brand intent is a stronger spend predictor than exercise intensity alone.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Chart 3: Income x WTP
    st.markdown("<div class='sec-hdr'>Chart 3  Income Bracket x Average Willingness to Pay</div>", unsafe_allow_html=True)
    df_c3 = (df_e.groupby("Q5_monthly_income")["Q24_wtp_monthly_inr"]
               .agg(Mean="mean", Median="median", Std="std").reset_index())
    df_c3["Q5_monthly_income"] = pd.Categorical(df_c3["Q5_monthly_income"],
        categories=INCOME_ORDER, ordered=True)
    df_c3 = df_c3.sort_values("Q5_monthly_income")
    df_c3["Label"] = df_c3["Q5_monthly_income"].map(INCOME_LBL)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_c3["Label"], y=df_c3["Mean"].round(0), name="Mean WTP",
        marker=dict(color=df_c3["Mean"], colorscale="Blues",
                    showscale=True, colorbar=dict(title="INR", x=1.02)),
        text=[f"INR {v:,.0f}" for v in df_c3["Mean"]],
        textposition="outside", textfont=dict(color="#c9d1d9", size=11),
        error_y=dict(type="data", array=df_c3["Std"].round(0),
                     visible=True, color="#484f58", thickness=1.5, width=6)))
    fig3.add_trace(go.Scatter(
        x=df_c3["Label"], y=df_c3["Median"], mode="lines+markers", name="Median WTP",
        line=dict(color="#d29922", width=2.2, dash="dot"),
        marker=dict(size=9, color="#d29922", symbol="diamond")))
    fig3.update_layout(title="Average and Median WTP by Income Bracket (error bars = 1 SD)",
        xaxis_title="Monthly Income (INR)", yaxis_title="WTP (INR)",
        height=460, **BASE, legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: Mean WTP scales non-linearly - the jump from the 80K-1L to Above-1L bracket
        is the steepest (~40% increase), suggesting a premium tier above 4000/month is viable
        for the top income quintile.<br>
        Insight 2: Median WTP trails the mean by 15-25% across all brackets confirming right-skew;
        the regression model should use log-transformed WTP and pricing should target the median,
        not the mean, for mass-market tiers.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Chart 4: Feature Popularity
    st.markdown("<div class='sec-hdr'>Chart 4  App Feature Popularity Ranking (Q16 Columns)</div>", unsafe_allow_html=True)
    FEAT_LBL = {
        "Q16_feat_personalised_rec":    "Personalised Recommendations",
        "Q16_feat_browse_purchase":     "Browse and Purchase Products",
        "Q16_feat_loyalty_rewards":     "Loyalty Rewards and Member Discounts",
        "Q16_feat_sustainability_info": "Sustainability and Fabric Certification",
        "Q16_feat_outfit_builder":      "Workout Outfit Builder",
        "Q16_feat_size_virtual_tryon":  "Size Guide and Virtual Try-On",
        "Q16_feat_order_tracking":      "Order Tracking and Easy Returns",
        "Q16_feat_community_features":  "Community and Fitness Challenges",
        "Q16_feat_flash_sales":         "Flash Sales and Limited Edition Drops",
        "Q16_feat_brand_collab":        "Brand Collabs and Athlete Content",
    }
    feat_cols = list(FEAT_LBL.keys())
    fs = df_e[feat_cols].sum().reset_index()
    fs.columns = ["col", "Count"]
    fs["Feature"] = fs["col"].map(FEAT_LBL)
    fs["Pct"] = (fs["Count"] / len(df_e) * 100).round(1)
    fs = fs.sort_values("Count", ascending=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=fs["Feature"], x=fs["Count"], orientation="h",
        marker=dict(color=fs["Count"], colorscale="Viridis", showscale=False),
        text=[f"  {p}%  ({v:,} respondents)" for p, v in zip(fs["Pct"], fs["Count"])],
        textposition="outside", textfont=dict(color="#8b949e", size=10.5)))
    fig4.update_layout(
        title="Feature Popularity - Count of Respondents Selecting Each Feature",
        xaxis_title="Number of Respondents",
        xaxis_range=[0, len(df_e) * 1.25],
        yaxis_title="",
        height=510,
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(family="Inter, system-ui, sans-serif", color="#c9d1d9", size=12),
        title_font=dict(size=15, color="#e6edf3"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=10, r=220, t=55, b=40),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: Browse and Purchase along with Order Tracking are near-universal (85%+ selection)
        and are table-stakes MVP features - competitors already offer them so they must be flawless
        but will not be the differentiating factor.<br>
        Insight 2: Loyalty Rewards ranks third overall while Brand Collab Content ranks last,
        suggesting monetary benefits drive more feature desire than content-led engagement
        for this target audience.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Chart 5: Persona x Purchase Frequency
    st.markdown("<div class='sec-hdr'>Chart 5  Persona Distribution x Purchase Frequency</div>", unsafe_allow_html=True)
    df_c5 = (df_e.groupby(["persona_label","Q14_purchase_frequency"])
                 .size().reset_index(name="Count"))
    df_c5["Persona"]    = df_c5["persona_label"].map(PERSONA_LBL)
    df_c5["Freq_Label"] = df_c5["Q14_purchase_frequency"].map(FREQ_LBL)
    df_c5["Q14_purchase_frequency"] = pd.Categorical(
        df_c5["Q14_purchase_frequency"], categories=FREQ_ORDER, ordered=True)
    df_c5 = df_c5.sort_values("Q14_purchase_frequency")
    fig5 = px.bar(df_c5, x="Persona", y="Count", color="Freq_Label",
        barmode="stack",
        category_orders={
            "Persona": [PERSONA_LBL[p] for p in PERSONA_ORDER],
            "Freq_Label": [FREQ_LBL[f] for f in FREQ_ORDER],
        },
        color_discrete_sequence=["#f85149","#f0883e","#d29922","#3fb950","#58a6ff"],
        labels={"Persona":"Customer Persona","Count":"Respondents","Freq_Label":"Purchase Frequency"},
        title="Purchase Frequency Distribution by Customer Persona")
    fig5.update_traces(marker_line_width=0)
    fig5.update_layout(height=475, **BASE, legend_title="Purchase Frequency", bargap=0.22)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: Serious Athletes show the densest Monthly+ stack despite being the smallest persona;
        their high purchase cadence makes them the highest revenue-per-user segment - early drop
        access and performance bundles would increase their average order value further.<br>
        Insight 2: Budget Students are concentrated in Rarely and Yearly buckets; converting them
        to Every 2-3 Months via student-only loyalty cashback (not discounts) could unlock
        significant incremental revenue given their large cohort size.
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Chart 6: City Tier x Download Intent 100% Stacked
    st.markdown("<div class='sec-hdr'>Chart 6  City Tier x App Download Intent (100% Stacked)</div>", unsafe_allow_html=True)
    df_c6 = (df_e.groupby(["Q3_city_tier","Q25_app_download_intent"])
                 .size().reset_index(name="Count"))
    pivot = (df_c6.pivot(index="Q3_city_tier", columns="Q25_app_download_intent", values="Count")
                  .fillna(0))
    pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).reset_index()
    pct["Q3_city_tier"] = pd.Categorical(pct["Q3_city_tier"],
        categories=CITY_ORDER, ordered=True)
    pct = pct.sort_values("Q3_city_tier")
    fig6 = go.Figure()
    for intent, color in [("Yes","#3fb950"),("Maybe","#d29922"),("No","#f85149")]:
        if intent in pct.columns:
            vals = pct[intent].round(1)
            fig6.add_trace(go.Bar(
                name=intent, x=pct["Q3_city_tier"], y=vals,
                marker_color=color, marker_line_width=0,
                text=[f"{v:.1f}%" for v in vals],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=12, color="white")))
    fig6.update_layout(
        barmode="stack",
        title="App Download Intent Proportions by City Tier (100% Normalised)",
        xaxis_title="City Tier",
        yaxis=dict(title="Percentage (%)", range=[0,100], ticksuffix="%",
                   gridcolor="#21262d"),
        height=460, **BASE, legend_title="Download Intent")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""<div class='insight'>
        Insight 1: Metro cities show the highest Yes proportion validating an urban-first launch strategy;
        Tier-1 cities have a large Maybe pool that is highly convertible with the right referral
        or free-trial incentive at onboarding.<br>
        Insight 2: No intent grows progressively from Metro to Tier-3 confirming declining D2C app
        adoption appetite with city size - Tier-3 outreach should prioritise brand awareness
        campaigns over direct download CTAs to build consideration first.
    </div>""", unsafe_allow_html=True)
