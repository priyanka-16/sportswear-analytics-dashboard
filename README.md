# 👟 Sportswear D2C App — Survey Analytics Dashboard

An interactive Streamlit dashboard built for a **Direct-to-Consumer (D2C) sportswear and athleisure brand** targeting urban Indians aged 18–35. This project covers the full pipeline from synthetic survey data generation to exploratory data analysis — structured specifically for downstream machine learning (Classification, Regression, Clustering, Association Rule Mining).

---

## 📁 Repository Structure

```
📦 sportswear-d2c-survey-dashboard
 ┣ 📄 app.py                              # Main Streamlit application
 ┣ 📄 sportswear_survey_synthetic_2000.csv  # Synthetic survey dataset (N=2,000)
 ┣ 📄 requirements.txt                    # Python dependencies
 ┗ 📄 README.md                           # Project documentation (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sportswear-d2c-survey-dashboard.git
cd sportswear-d2c-survey-dashboard
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## 📊 Dashboard Overview

The app is structured into **2 tabs**:

### Tab 1 — Data Overview & Cleaning
| Section | Description |
|---|---|
| 1.1 Raw Preview | First 10 rows, shape, column count, missing value summary |
| 1.2 Missing Values | Bar chart of nulls per column (auto-adapts to real survey data) |
| 1.3 Outlier Detection | Boxplots for `Q11_spend` and `Q24_wtp` **before** cleaning, with cutoff lines |
| 1.4 Cleaning Pipeline | 4 sequential steps: outlier removal → log transform → ordinal encoding |
| 1.5 Before vs After | Row count metrics + AFTER boxplots on log-transformed columns |
| 1.6 Cleaned Preview | Filtered 13-column preview of the cleaned dataset |

### Tab 2 — Exploratory Data Analysis
| # | Chart | Type |
|---|---|---|
| 1 | Age Group × Gender (City Tier filter) | Grouped Bar with dropdown |
| 2 | Workout Frequency × Spend × App Intent | Box Plot (3-colour by intent) |
| 3 | Income Bracket × Willingness to Pay | Bar + Median Line + Error Bars |
| 4 | App Feature Popularity Ranking | Horizontal Bar (Viridis gradient) |
| 5 | Persona × Purchase Frequency | Stacked Bar (5 personas × 5 frequencies) |
| 6 | City Tier × App Download Intent | 100% Stacked Bar (proportions) |

---

## 🤖 ML Readiness — Algorithm Mapping

This dataset was designed from scratch to support four ML techniques simultaneously:

| Algorithm | Target / Purpose | Key Columns |
|---|---|---|
| **Classification** | Predict app download intent (Yes/Maybe/No) | `Q25_app_download_intent` as target; Q1–Q5, Q7, Q10, Q16_*, Q22 as features |
| **Regression** | Predict monthly spend & WTP | `Q11_log_spend`, `Q24_log_wtp` as targets; Q5, Q7, Q13_*, Q14, Q17 as features |
| **K-Means Clustering** | Identify customer personas | Q7, Q8, Q11, Q17–Q20, Q13_factor_style, Q16_feat_* (scaled) |
| **Association Rule Mining** | Discover feature & product co-preferences | Q16_feat_* (10 binary cols) × Q21_prod_* (7 binary cols) × Q6_act_* (9 binary cols) |

> ⚠️ `persona_label` is the **hidden ground truth** for K-Means validation only (Adjusted Rand Index). Never include it as a training feature.

---

## 🧬 Dataset — 5 Latent Customer Personas

The synthetic data is seeded around 5 realistic personas (used as a latent generation engine):

| Persona | Share | Profile |
|---|---|---|
| 🚶 Casual Gym-Goer | 27.3% | 3–4 days/week, moderate spend, brand loyal |
| 👗 Fashion-First | 21.9% | Style > performance, influencer-driven, metro-heavy |
| 🏋️ Serious Athlete | 18.4% | 5–7 days/week, high spend, fabric quality focused |
| 🏔️ Outdoor Enthusiast | 16.7% | Running/cycling/trekking, sustainability conscious |
| 🎓 Budget Student | 15.7% | Low income, flash-sale driven, price sensitive |

---

## 🗂️ Dataset Schema (75 Feature Columns)

| Column Group | Questions | Count | Encoding |
|---|---|---|---|
| Demographics | Q1–Q5 | 5 | Categorical / Ordinal (encoded) |
| Fitness Lifestyle | Q6–Q10 | 20 | Binary multi-select + categorical |
| Shopping Behaviour | Q11–Q15 | 26 | Numeric + binary multi-select |
| App Feature Preferences | Q16–Q21 | 21 | Binary + Likert (1–5) |
| Digital Behaviour | Q22 | 1 | Categorical |
| ML Targets | Q24, Q25 | 2 | Numeric (regression) + Categorical (classification) |

---

## ⚙️ Data Cleaning Steps Applied

1. **Remove extreme Q11 outliers** — rows where `Q11_current_monthly_spend_inr > ₹15,000`
2. **Remove extreme Q24 skeptics** — rows where `Q24_wtp_monthly_inr < ₹200`
3. **Log transformation** — `log1p(Q11)` → `Q11_log_spend` | `log1p(Q24)` → `Q24_log_wtp`
4. **Ordinal encoding** — Q1 (age), Q5 (income), Q7 (workout days), Q14 (purchase frequency)

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.10+` | Core language |
| `Streamlit` | Dashboard framework |
| `Plotly` | Interactive visualisations |
| `Pandas` | Data manipulation |
| `NumPy` | Numerical computing + synthetic data generation |
| `scikit-learn` | (Ready for ML pipeline — classification, clustering, regression) |

---

## 🔮 Coming Soon (Planned Tabs)

- **Tab 3** — Classification Model (Random Forest / XGBoost) with feature importance
- **Tab 4** — Regression Model (Predict spend & WTP) with residual plots
- **Tab 5** — K-Means Clustering (Elbow + Silhouette + Cluster Profiling)
- **Tab 6** — Association Rule Mining (Apriori — Feature & Product Rules)

---

## 👤 Author

**Payal Manwani**  
MBA Candidate | Marketing & Strategy  
*Survey design, data generation, and dashboard built as part of applied market research for a D2C sportswear brand launch project.*

---

## 📄 License

This project is for academic and educational purposes.  
Dataset is fully synthetic — generated with `numpy.random.default_rng(seed=42)`.  
No real consumer data has been used.
