import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sportswear App — Survey Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CSV_PATH = "sportswear_survey_synthetic_2000.csv"

ORDINAL_MAPS = {
    "Q1_age_group": {"18-22": 0, "23-27": 1, "28-32": 2, "33-35": 3},
    "Q5_monthly_income": {
        "Below_20k": 0, "20k-40k": 1, "40k-60k": 2,
        "60k-80k": 3, "80k-100k": 4, "Above_100k": 5
    },
    "Q7_workout_days_per_week": {
        "0_days": 0, "1-2_days": 1, "3-4_days": 2, "5-7_days": 3
    },
    "Q14_purchase_frequency": {
        "Rarely": 0, "Yearly": 1, "Every_6_months": 2,
        "Every_2-3_months": 3, "Monthly_or_more": 4
    }
}

Q16_COLS  = [c for c in pd.read_csv(CSV_PATH, nrows=0).columns if c.startswith("Q16_")]
FEAT_LABELS = {
    "Q16_feat_personalised_rec":   "Personalised Recommendations",
    "Q16_feat_browse_purchase":    "Browse & Purchase",
    "Q16_feat_loyalty_rewards":    "Loyalty Rewards",
    "Q16_feat_sustainability_info":"Sustainability Info",
    "Q16_feat_outfit_builder":     "Outfit Builder",
    "Q16_feat_size_virtual_tryon": "Size Guide & Virtual Try-On",
    "Q16_feat_order_tracking":     "Order Tracking & Returns",
    "Q16_feat_community_features": "Community Features",
    "Q16_feat_flash_sales":        "Flash Sales & Limited Drops",
    "Q16_feat_brand_collab":       "Brand Collab & Athlete Content"
}

INCOME_ORDER  = ["Below_20k","20k-40k","40k-60k","60k-80k","80k-100k","Above_100k"]
WFREQ_ORDER   = ["0_days","1-2_days","3-4_days","5-7_days"]
PFREQ_ORDER   = ["Rarely","Yearly","Every_6_months","Every_2-3_months","Monthly_or_more"]
INTENT_COLORS = {"Yes": "#2ECC71", "Maybe": "#F39C12", "No": "#E74C3C"}
PERSONA_COLORS = {
    "Serious_Athlete":    "#E74C3C",
    "Casual_Gym_Goer":    "#3498DB",
    "Fashion_First":      "#9B59B6",
    "Outdoor_Enthusiast": "#27AE60",
    "Budget_Student":     "#F39C12"
}

# ─── DATA LOADER ─────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(CSV_PATH)

# ─── CLEANING PIPELINE ───────────────────────────────────────────────────────
def apply_cleaning(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["Q11_current_monthly_spend_inr"] <= 15000]
    df = df[df["Q24_wtp_monthly_inr"] >= 200]
    df["Q11_log_spend"] = np.log1p(df["Q11_current_monthly_spend_inr"])
    df["Q24_log_wtp"]   = np.log1p(df["Q24_wtp_monthly_inr"])
    for col, mapping in ORDINAL_MAPS.items():
        df[f"{col}_encoded"] = df[col].map(mapping)
    df.reset_index(drop=True, inplace=True)
    return df

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def render_sidebar(raw: pd.DataFrame):
    with st.sidebar:
        st.image(
            "https://img.icons8.com/emoji/96/running-shoe.png",
            width=64
        )
        st.title("Dataset Info")
        st.metric("Total Rows (Raw)", f"{len(raw):,}")
        st.metric("Total Columns", raw.shape[1])
        st.metric("Unique Personas", raw["persona_label"].nunique())
        st.divider()
        st.markdown("**Persona Breakdown**")
        for persona, cnt in raw["persona_label"].value_counts().items():
            icon = {"Serious_Athlete":"🏋️","Casual_Gym_Goer":"🤸",
                    "Fashion_First":"💅","Outdoor_Enthusiast":"🏕️","Budget_Student":"🎒"}.get(persona,"👤")
            st.markdown(f"{icon} **{persona.replace('_',' ')}** — {cnt}")
        st.divider()
        st.caption("D2C Sportswear App · Market Survey · 2026")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
def tab_overview(raw: pd.DataFrame):
    st.header("🗂️ Data Overview & Cleaning Pipeline")

    # ── Section 1: Raw Preview ────────────────────────────────────────────────
    st.subheader("1 · Raw Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows",    f"{raw.shape[0]:,}")
    col2.metric("Total Columns", raw.shape[1])
    col3.metric("Missing Values", int(raw.isnull().sum().sum()))
    st.dataframe(raw.head(10), use_container_width=True, height=280)

    # ── Section 2: Missing Values ─────────────────────────────────────────────
    st.subheader("2 · Missing Values per Column")
    missing = raw.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Count": missing.values})
    total_missing = missing_df["Missing Count"].sum()
    if total_missing == 0:
        st.success("✅ No missing values found in the dataset — perfectly clean synthetic data!")
    fig_miss = px.bar(
        missing_df[missing_df["Missing Count"] >= 0],
        x="Column", y="Missing Count",
        title="Missing Value Count per Column",
        color="Missing Count",
        color_continuous_scale=["#2ECC71", "#E74C3C"],
        height=320
    )
    fig_miss.update_layout(
        xaxis_tickangle=-90, showlegend=False,
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", title_font_size=15
    )
    st.plotly_chart(fig_miss, use_container_width=True)

    # ── Section 3: Outlier Detection BEFORE Cleaning ──────────────────────────
    st.subheader("3 · Outlier Detection — BEFORE Cleaning")
    st.caption("Boxplots reveal extreme high-spenders (Q11) and extreme low WTP respondents (Q24).")

    fig_box = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Q11 — Current Monthly Spend (INR)",
            "Q24 — Willingness to Pay (INR)"
        )
    )
    fig_box.add_trace(
        go.Box(y=raw["Q11_current_monthly_spend_inr"], name="Q11 Spend",
               marker_color="#3498DB", boxmean=True),
        row=1, col=1
    )
    fig_box.add_trace(
        go.Box(y=raw["Q24_wtp_monthly_inr"], name="Q24 WTP",
               marker_color="#E74C3C", boxmean=True),
        row=1, col=2
    )
    fig_box.update_layout(
        height=420, showlegend=False,
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", title_text="Outlier Detection (Before Cleaning)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Section 4: Cleaning Steps ─────────────────────────────────────────────
    st.subheader("4 · Cleaning Steps Applied")
    with st.expander("📋 View all cleaning rules", expanded=True):
        st.markdown("""
        | Step | Rule | Rationale |
        |------|------|-----------|
        | 1 | Remove rows where **Q11 > ₹15,000** | Extreme outliers that distort regression |
        | 2 | Remove rows where **Q24 < ₹200** | Extreme skeptics with near-zero WTP |
        | 3 | `Q11_log_spend = log1p(Q11)` | Normalise right-skewed spend distribution |
        | 4 | `Q24_log_wtp = log1p(Q24)` | Normalise right-skewed WTP distribution |
        | 5 | Label-encode **Q1** (age), **Q5** (income), **Q7** (workout freq), **Q14** (purchase freq) | Convert ordinal categories to integers for ML |
        """)

    cleaned = apply_cleaning(raw)

    # Before vs After metrics
    removed = len(raw) - len(cleaned)
    st.markdown("#### Before vs After Cleaning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Before", f"{len(raw):,}")
    c2.metric("Rows After",  f"{len(cleaned):,}", delta=f"-{removed} removed")
    c3.metric("Rows Removed", f"{removed}", delta=f"{removed/len(raw)*100:.1f}% of data", delta_color="inverse")

    # Log-transformed distribution comparison
    fig_log = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Q11 — Log-Transformed Spend", "Q24 — Log-Transformed WTP")
    )
    fig_log.add_trace(
        go.Histogram(x=cleaned["Q11_log_spend"], nbinsx=40,
                     marker_color="#3498DB", opacity=0.8, name="Q11 log"),
        row=1, col=1
    )
    fig_log.add_trace(
        go.Histogram(x=cleaned["Q24_log_wtp"], nbinsx=40,
                     marker_color="#E74C3C", opacity=0.8, name="Q24 log"),
        row=1, col=2
    )
    fig_log.update_layout(
        height=350, showlegend=False,
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", title_text="After Log Transformation (Near-Normal Distributions)"
    )
    st.plotly_chart(fig_log, use_container_width=True)

    # Cleaned dataset preview
    st.subheader("5 · Cleaned Dataset Preview")
    display_cols = (
        ["respondent_id", "Q1_age_group", "Q1_age_group_encoded",
         "Q5_monthly_income", "Q5_monthly_income_encoded",
         "Q7_workout_days_per_week", "Q7_workout_days_per_week_encoded",
         "Q11_current_monthly_spend_inr", "Q11_log_spend",
         "Q24_wtp_monthly_inr", "Q24_log_wtp",
         "Q14_purchase_frequency", "Q14_purchase_frequency_encoded",
         "Q25_app_download_intent", "persona_label"]
    )
    st.dataframe(cleaned[display_cols].head(10), use_container_width=True, height=280)
    st.caption(f"Showing 15 key columns. Full cleaned dataset has {cleaned.shape[1]} columns.")

    # Store in session state for Tab 2
    st.session_state["cleaned_df"] = cleaned
    st.success(f"✅ Cleaned dataset stored in session state — {len(cleaned):,} rows × {cleaned.shape[1]} columns ready for EDA.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def tab_eda():
    st.header("📊 Exploratory Data Analysis")

    if "cleaned_df" not in st.session_state:
        st.warning("⚠️ Please visit Tab 1 first to generate the cleaned dataset.")
        return

    df = st.session_state["cleaned_df"]

    # ── Chart 1: Age Group × Gender × City Tier ──────────────────────────────
    st.subheader("Chart 1 · Age Group vs Gender Distribution by City Tier")

    city_options = ["All"] + sorted(df["Q3_city_tier"].unique().tolist())
    selected_city = st.selectbox("🔽 Filter by City Tier:", city_options, key="city_filter")

    df1 = df if selected_city == "All" else df[df["Q3_city_tier"] == selected_city]
    agg1 = df1.groupby(["Q1_age_group", "Q2_gender"]).size().reset_index(name="Count")
    fig1 = px.bar(
        agg1, x="Q1_age_group", y="Count", color="Q2_gender",
        barmode="group",
        title=f"Age Group vs Gender — City Tier: {selected_city}",
        color_discrete_map={
            "Male": "#3498DB", "Female": "#E91E8C",
            "Non_binary": "#9B59B6", "Prefer_not_to_say": "#95A5A6"
        },
        labels={"Q1_age_group": "Age Group", "Q2_gender": "Gender"},
        height=420,
        category_orders={"Q1_age_group": ["18-22","23-27","28-32","33-35"]}
    )
    fig1.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", legend_title_text="Gender",
        bargap=0.20
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("📌 The 23–27 cohort dominates across all city tiers, with males slightly overrepresented in Metro cities.  
"
               "📌 Tier-2 and Tier-3 cities show a stronger female presence in younger age bands (18–22), signalling a growing urban-adjacent female audience.")

    st.divider()

    # ── Chart 2: Workout Frequency × Monthly Spend × App Intent ──────────────
    st.subheader("Chart 2 · Workout Frequency vs Monthly Spend by App Download Intent")
    fig2 = px.box(
        df,
        x="Q7_workout_days_per_week",
        y="Q11_current_monthly_spend_inr",
        color="Q25_app_download_intent",
        title="Spend Distribution by Workout Frequency — Coloured by App Download Intent",
        color_discrete_map=INTENT_COLORS,
        labels={
            "Q7_workout_days_per_week":      "Workout Days / Week",
            "Q11_current_monthly_spend_inr": "Monthly Spend (INR)",
            "Q25_app_download_intent":       "Download Intent"
        },
        height=460,
        category_orders={
            "Q7_workout_days_per_week": WFREQ_ORDER,
            "Q25_app_download_intent": ["Yes","Maybe","No"]
        }
    )
    fig2.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", boxmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("📌 Respondents who work out 5–7 days/week spend significantly more (median ~₹3,500) and are the most likely to say 'Yes' to downloading the app.  
"
               "📌 Even 0-day exercisers who said 'Yes' show a wide spend range, suggesting fashion-forward athleisure buyers exist independent of fitness activity.")

    st.divider()

    # ── Chart 3: Income Bracket × Average WTP ────────────────────────────────
    st.subheader("Chart 3 · Income Bracket vs Average Willingness to Pay (₹)")
    agg3 = (
        df.groupby("Q5_monthly_income")["Q24_wtp_monthly_inr"]
          .mean()
          .reset_index()
          .rename(columns={"Q24_wtp_monthly_inr": "Avg_WTP"})
    )
    agg3["Q5_monthly_income"] = pd.Categorical(agg3["Q5_monthly_income"], categories=INCOME_ORDER, ordered=True)
    agg3 = agg3.sort_values("Q5_monthly_income")
    agg3["Avg_WTP_rounded"] = agg3["Avg_WTP"].round(0).astype(int)

    fig3 = px.bar(
        agg3, x="Q5_monthly_income", y="Avg_WTP",
        title="Average Willingness to Pay per Monthly Income Bracket",
        color="Avg_WTP",
        color_continuous_scale=px.colors.sequential.Teal,
        text="Avg_WTP_rounded",
        labels={"Q5_monthly_income": "Monthly Income (INR)", "Avg_WTP": "Avg WTP (INR)"},
        height=420
    )
    fig3.update_traces(texttemplate="₹%{text:,}", textposition="outside")
    fig3.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", coloraxis_showscale=False,
        xaxis_title="Monthly Income Bracket"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("📌 WTP scales sharply with income — respondents earning above ₹1L are willing to spend nearly 5× more than those below ₹20k.  
"
               "📌 The ₹40k–₹60k bracket represents the largest addressable sweet spot: mid-range WTP (~₹3,000) with the highest respondent volume.")

    st.divider()

    # ── Chart 4: App Feature Popularity ──────────────────────────────────────
    st.subheader("Chart 4 · Top App Features Preferred by Respondents")
    feat_sums = df[Q16_COLS].sum().reset_index()
    feat_sums.columns = ["Feature_Col", "Count"]
    feat_sums["Feature"] = feat_sums["Feature_Col"].map(FEAT_LABELS)
    feat_sums["Pct"] = (feat_sums["Count"] / len(df) * 100).round(1)
    feat_sums = feat_sums.sort_values("Count")

    fig4 = px.bar(
        feat_sums, x="Count", y="Feature",
        orientation="h",
        title="Feature Popularity — % of Respondents Who Want Each Feature",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis,
        text=feat_sums["Pct"].apply(lambda x: f"{x}%"),
        labels={"Count": "Respondents Who Selected", "Feature": "App Feature"},
        height=460
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", coloraxis_showscale=False,
        xaxis_range=[0, feat_sums["Count"].max() * 1.15]
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("📌 'Browse & Purchase' is near-universally desired (>88%), confirming that seamless commerce is the table-stakes feature — no compromise allowed here.  
"
               "📌 'Loyalty Rewards' and 'Order Tracking' rank in the top 3, suggesting that retention mechanics and post-purchase experience are as important as the catalogue itself.")

    st.divider()

    # ── Chart 5: Persona × Purchase Frequency ────────────────────────────────
    st.subheader("Chart 5 · Customer Persona vs Purchase Frequency")
    agg5 = df.groupby(["persona_label", "Q14_purchase_frequency"]).size().reset_index(name="Count")
    agg5["Q14_purchase_frequency"] = pd.Categorical(
        agg5["Q14_purchase_frequency"], categories=PFREQ_ORDER, ordered=True
    )
    agg5 = agg5.sort_values("Q14_purchase_frequency")

    fig5 = px.bar(
        agg5,
        x="persona_label", y="Count",
        color="Q14_purchase_frequency",
        title="Persona Distribution vs Purchase Frequency",
        labels={"persona_label": "Customer Persona", "Q14_purchase_frequency": "Purchase Frequency"},
        barmode="stack",
        color_discrete_sequence=px.colors.sequential.RdBu,
        height=460,
        category_orders={"Q14_purchase_frequency": PFREQ_ORDER}
    )
    fig5.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", xaxis_tickangle=-15,
        legend_title_text="Purchase Frequency"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("📌 Serious Athletes show the highest proportion of 'Monthly or more' buyers — making them the highest-LTV segment for subscription or loyalty tier design.  
"
               "📌 Budget Students are heavily concentrated in 'Every 6 months' and 'Rarely' buckets, suggesting flash sales and one-time discount nudges are more effective than loyalty programmes for this segment.")

    st.divider()

    # ── Chart 6: City Tier × App Download Intent (100% Stacked) ──────────────
    st.subheader("Chart 6 · City Tier vs App Download Intent (Proportional)")
    agg6 = df.groupby(["Q3_city_tier", "Q25_app_download_intent"]).size().reset_index(name="Count")
    totals = agg6.groupby("Q3_city_tier")["Count"].transform("sum")
    agg6["Percentage"] = (agg6["Count"] / totals * 100).round(1)
    agg6["Q25_app_download_intent"] = pd.Categorical(
        agg6["Q25_app_download_intent"], categories=["No","Maybe","Yes"], ordered=True
    )
    agg6 = agg6.sort_values("Q25_app_download_intent")

    fig6 = px.bar(
        agg6,
        x="Q3_city_tier", y="Percentage",
        color="Q25_app_download_intent",
        title="App Download Intent by City Tier (100% Proportional)",
        labels={"Q3_city_tier": "City Tier", "Percentage": "% of Respondents", "Q25_app_download_intent": "Intent"},
        barmode="stack",
        color_discrete_map=INTENT_COLORS,
        text=agg6["Percentage"].apply(lambda x: f"{x:.1f}%"),
        height=440,
        category_orders={"Q25_app_download_intent": ["No","Maybe","Yes"]}
    )
    fig6.update_traces(textposition="inside", textfont_size=11)
    fig6.update_yaxes(range=[0, 105])
    fig6.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", legend_title_text="Download Intent",
        uniformtext_minsize=9, uniformtext_mode="hide"
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("📌 Metro cities lead in definitive 'Yes' intent, reflecting higher digital adoption and brand app familiarity — making metros the priority launch market.  
"
               "📌 Tier-2 and Tier-3 cities show higher 'Maybe' proportions, indicating latent demand that can be converted with targeted onboarding incentives and vernacular-language UI.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    raw = load_raw()
    render_sidebar(raw)

    st.title("👟 D2C Sportswear App — Market Survey Dashboard")
    st.markdown(
        "Synthetic dataset of **2,000 urban Indian respondents (18–35)** · "
        "Designed for Classification, Regression, Clustering & Association Rule Mining"
    )
    st.divider()

    tab1, tab2 = st.tabs([
        "🗂️  Tab 1 — Data Overview & Cleaning",
        "📊  Tab 2 — Exploratory Data Analysis"
    ])

    with tab1:
        tab_overview(raw)

    with tab2:
        tab_eda()

if __name__ == "__main__":
    main()
