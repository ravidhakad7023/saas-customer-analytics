# PipelineIQ: CRM Analytics Dashboard

A SaaS analytics portfolio project built on SaaS customer data (598 customers). The Streamlit dashboard surfaces churn drivers, cohort retention, survival economics, pricing elasticity, and marketing attribution.

---

## 🚀 Live Demos
Live deployment links will be added after the final Streamlit Community Cloud and Tableau Public deployments are published.

---

## 📊 The Business Problem
A SaaS business has three customer tiers, a worsening churn problem, and limited CS capacity. Where does every dollar of retention investment produce the most return? This project answers that question through interconnected analyses, moving from raw data to prioritized, quantified business decisions.

**Workflow**: Raw SaaS customer data → data cleaning and validation → exploratory analysis → SQL analytics → statistical modeling → business intelligence dashboard → quantified business insights.

### Key Findings
1. **Starter Unit Economics**: Starter represents approximately 61% of customers, with an observed 10.1% monthly churn rate and approximately 39% survival to CAC breakeven. 
2. **Rising Churn**: Observed monthly churn increased from approximately 5.9% (Q1 2022 cohorts) to 9.2% (Q2 2023 cohorts) over the observed period.
3. **Enterprise Revenue Concentration**: Enterprise represents roughly 15% of customers but accounts for 45% of active MRR, with substantially lower observed monthly churn (3.1%) than Starter.
4. **Billing Cycle Adverse Selection**: The data shows an observed difference between annual and monthly billing churn where annual subscribers counter-intuitively display higher churn (e.g., Enterprise annual 5.0% vs monthly 2.3%).
5. **Marketing Attribution**: Last-click attribution assigns 42% of conversions to Organic and 18% to Referral. RidgeCV marketing mix modeling estimates 28% for Organic and 31% for Referral, indicating that last-click attribution may understate Referral's contribution.

---
## What This Project Shows Analytically

Beyond the business findings, the project demonstrates several analytical practices:

- **Honest uncertainty quantification:** Models include confidence intervals, explicit assumptions, and documented limitations. The pricing analysis is explicitly labeled simulated/illustrative rather than treated as historical causal evidence.

- **Methodological justification:** Customer-month churn is calculated as `SUM(churned) / SUM(dur)` rather than `churned.mean()`, which represents a lifetime churn proportion. Ridge regression is used for marketing mix modeling to handle multicollinearity while retaining correlated channel features.

- **Causal thinking:** Regression Discontinuity and Difference-in-Differences are used where the analysis requires causal identification, rather than relying only on simple group comparisons.

- **Data integrity as a first-class concern:** Data cleaning and validation document anomalies such as negative MRR, extreme outliers, inconsistent tier labels, and invalid dates before downstream analysis.

- **SQL as a production-style analytics artifact:** The project includes 10 documented SQL queries against `saas_intel.db`, covering core retention, churn, customer, and unit-economics questions.

## 📈 Key Metrics
| Metric | Value | Reference |
|---|---:|---:|
| Overall monthly churn | 7.18%/mo | 2%/mo reference |
| Starter monthly churn | 10.22%/mo | 2%/mo reference |
| Enterprise monthly churn | 3.10%/mo | 2%/mo reference |
| Starter survival to breakeven | 39% | >50% reference |
| Enterprise net LTV | +$4,195 | — |
| Churn trend | +57% over 6 quarters | — |
| NRR (estimated) | 85% | 106% reference |
| Annual vs monthly churn | Annual > Monthly | Annual < Monthly reference |

---

## 🛠️ Skills Demonstrated
- **SQL**: CTEs, JOINs, GROUP BY, cohort analysis, aggregations and derived metrics.
- **Analytics**: Churn and Retention Analysis, LTV to CAC ratios, Pricing elasticity (Illustrative), Marketing attribution (MMM), SaaS-adapted RFM-style segmentation.
- **BI**: Streamlit Dashboards, Excel Reporting.
- **Statistics**: Regression (RidgeCV), Time-series cross-validation, Causal inference (Regression Discontinuity & Difference-in-Differences).

---

## 🔬 Analytical Approach
This project leverages specific methodologies to move beyond basic averages:
- **Customer-Months Churn Rate**: Churn is measured as `SUM(churned) / SUM(dur)` rather than a simple `churned.mean()`, converting a lifetime proportion into a true monthly rate.
- **RidgeCV Marketing Mix Modeling**: Ridge regression was selected over OLS to handle multicollinearity among correlated marketing channels while retaining all channel features. Time-series cross-validation was used to select the regularization strength and reduce data leakage risk.
- **Causal Inference**: Where applicable, findings were validated using Regression Discontinuity (RD) and Difference-in-Differences (DiD) to distinguish correlation from causation.
- **SQL as a Production Artifact**: 10 production-style SQL queries support cohort, churn, retention, and KPI analysis against the compiled `saas_intel.db`.
- **Data Integrity**: Extensive EDA and cleaning processes are documented to audit missing values, outliers, and data-entry errors.

---

## 🗂️ Project Deliverables & Reproducibility
This project strictly delineates its analytical deliverables:
1. **Python Notebooks**: Complete data cleaning, EDA, and statistical models.
2. **SQL Analyses**: 10 production-style queries used for core metrics, run against `saas_intel.db`.
3. **Dashboard Outputs**: An 8-page Streamlit application summarizing the intelligence.
4. **Excel Deliverables**: A fully reproducible 9-sheet stakeholder workbook (`excel/PipelineIQ_Analytics.xlsx`).

*All core retention and churn metrics are reproducible via SQL against `saas_intel.db`.*

### Notebooks → Business Questions
| Notebook | Business Question |
|---|---|
| `01_eda_narrative.ipynb` | What patterns and data-quality issues exist? |
| `02_cleaning_decisions.ipynb` | What cleaning decisions are required and why? |
| `03_churn_model_development.ipynb` | What features are associated with churn risk? |
| `04_mmm_development.ipynb` | How do acquisition channels contribute beyond last-click attribution? |
| `05_survival_analysis.ipynb` | How long do customers survive and when does CAC break even? |
| `06_pricing_optimisation.ipynb` | How can price elasticity be modelled? (Illustrative/Simulated) |
| `07_rfm_segmentation.ipynb` | How can customers be segmented using available SaaS proxies? |
| `08_risk_gap_analysis.ipynb` | Where do observed metrics differ from reference benchmarks? |
| `09_experiment_design.ipynb` | Which experiments could validate potential interventions? |
| `10_causal_inference.ipynb` | Which relationships can be evaluated using causal designs? |
| `11_cohort_retention_sql.ipynb` | How do retention and cohort metrics look using SQL? |

### Excel Deliverable
`excel/PipelineIQ_Analytics.xlsx` contains 9 stakeholder-oriented sheets covering:
1. 1_Executive KPIs
2. 2_Tier Breakdown
3. 3_Channel Breakdown
4. 4_Quarterly Trend
5. 5_Cohort Retention
6. 6_Survival Breakeven
7. 7_RFM Segments
8. 8_Pricing Elasticity (Illustrative/Simulated)
9. 9_Gap Analysis

---

## 📖 Data Dictionary

`clean_saas_customers.csv`, 598 rows, one per customer:

| Column | Type | Description |
|---|---|---|
| customer_id | str | Unique identifier |
| signup_month | date | First month of subscription |
| tier | str | Starter / Pro / Enterprise |
| billing_cycle | str | Annual / Monthly |
| channel | str | Acquisition channel |
| mrr | float | Monthly recurring revenue, cleaned to $5 to $500 |
| churned | int | 1 = churned, 0 = active as of Jan 2024 |
| churn_date | date | Month of churn (null if active) |



---

## ⚙️ Tech Stack
- **Dashboard:** Streamlit, Plotly
- **Analysis:** pandas, numpy, scipy, scikit-learn, statsmodels
- **Database:** SQLite via sqlite3
- **Notebooks:** Jupyter
- **Export:** openpyxl

---

## 💻 Quick Start

```bash
git clone <repository-url>
cd pipelineiq
pip install -r requirements.txt

# 1. Rebuild the SQLite database & Excel Report
python src/11_survival_analysis.py
python src/build_excel_report.py

# 2. Run the Dashboard
streamlit run dashboard/app.py
```

---

## 📂 Project Structure

```
pipelineiq/
├── dashboard/
│   └── app.py                          # Streamlit dashboard (8 pages)
├── saas_intel.db                       # SQLite database (built by src scripts)
├── requirements.txt
│
├── data/
│   └── processed/
│       └── clean_saas_customers.csv    # 598 rows, used for db generation
│
├── src/
│   ├── 11_survival_analysis.py         # Builds km_survival, cac_breakeven, and loads data
│   └── build_excel_report.py           # Programmatically regenerates Excel output
│
├── queries/                            # 10 analytical SQL queries
│
├── notebooks/                          # 11 sequential Jupyter notebooks
│
└── excel/
    └── PipelineIQ_Analytics.xlsx        # 9-sheet stakeholder workbook
```

---

## 📈 Dashboard Pages

| Page | What it shows |
|---|---|
| Executive Overview | 4 KPI metrics, priority action items, active MRR by tier and quarter |
| Customer Churn Analysis | Churn by tier, channel, billing cycle, and cohort quarter |
| Cohort Retention | 18-cohort heatmap + average retention curve with P25–P75 band |
| Survival & CAC Breakeven | One panel per tier, survival curve, breakeven month, % surviving to BE |
| Pricing Elasticity | Log-log demand model, tier-level elasticity, optimal price ranges |
| Marketing Attribution (MMM) | Last-click vs RidgeCV attribution, channel churn enrichment |
| Customer Segmentation (RFM) | 5 RFM segments, MRR at risk, segment action matrix |
| Churn Risk Model | Customer-level churn risk scores, model performance, and risk segmentation |
---

## ⚠️ Methodological Limitations

- **Pricing Elasticity Analysis**: The pricing analysis (Notebook 06 and Dashboard Page 5) uses **simulated/illustrative price-conversion data**. It is included to demonstrate the analytical methodology of price elasticity and log-log demand models, but the specific elasticity figures are not historical empirical facts.
- **Churn Rate Definition**: Monthly churn rates use the customer-months method: `SUM(churned) ÷ SUM(dur)`. The dashboard also displays *lifetime churn proportion* separately where appropriate.
- **RFM Segmentation**: Traditional transactional RFM requires login/frequency data. This project uses a SaaS-adapted version using available tenure and MRR proxies.
- **Causal Claims**: Only findings validated with causal inference techniques (Regression Discontinuity, Difference-in-Differences) in Notebook 10 are stated as causal.
