"""
src/11_survival_analysis.py
PipelineIQ CRM Analytics — Kaplan-Meier survival curves + CAC breakeven analysis.

Run directly:  py src/11_survival_analysis.py
Outputs saved to saas_intel.db tables: km_survival, cac_breakeven

Actual computed values from clean_saas_customers.csv (598 customers, Jan 2022–Jun 2023):
  Starter:    hazard=10.07%/mo  avg_mrr=$30.14  breakeven=9.3mo  survival@BE=39.2%  median=6.88mo
  Pro:        hazard= 4.00%/mo  avg_mrr=$74.07  breakeven=8.1mo  survival@BE=72.3%  median=17.32mo
  Enterprise: hazard= 3.05%/mo  avg_mrr=$263.39 breakeven=10.0mo survival@BE=73.6%  median=22.73mo

Key finding: Starter median survival (6.88mo) is below the CAC breakeven month (9.3mo).
The average Starter churns before repaying acquisition cost. See notebooks/05_survival_analysis-5.ipynb
for full narrative, KM curves, and investment decision framework.
"""
import pandas as pd
import numpy as np
import sqlite3
import os


CAC     = {"starter": 196, "pro": 420, "enterprise": 1850}
MARGIN  = 0.70
REF_DATE = pd.Timestamp("2024-01-01")


def load_customers(path=None):
    candidates = [
        path,
        "data/processed/clean_saas_customers.csv",
        "clean_saas_customers.csv",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce").clip(5, 500)
            df["tier"] = df["tier"].astype(str).str.lower().str.strip()
            df["smo"] = pd.to_datetime(df["signup_month"], errors="coerce")
            df["cdt"] = pd.to_datetime(df["churn_date"], errors="coerce")
            df["dur"] = np.where(
                df["churned"] == 1,
                ((df["cdt"] - df["smo"]) / np.timedelta64(30, "D")).clip(0.5, 60),
                ((REF_DATE - df["smo"]) / np.timedelta64(30, "D")).clip(0.5, 60),
            )
            df["dur"] = pd.to_numeric(df["dur"], errors="coerce").fillna(6.0)
            return df
    raise FileNotFoundError("clean_saas_customers.csv not found. Run data pipeline first.")


def km_stats(T, E, query_months=(1, 3, 6, 9, 12, 18, 24)):
    """
    Kaplan-Meier estimator (hand-rolled, no lifelines dependency).

    Returns:
        median_survival  (float) — month where S(t) first drops below 0.50
        survival_at      (dict)  — {month: survival_probability} for each query month
        curve            (list of (t, S)) — full stepwise curve
    """
    T, E = np.array(T, float), np.array(E, int)
    event_times = np.sort(np.unique(T[E == 1]))
    S = 1.0
    median = 99.0
    curve = [(0.0, 1.0)]

    for t in event_times:
        at_risk = (T >= t).sum()
        events  = ((T == t) & (E == 1)).sum()
        if at_risk > 0:
            S *= 1 - events / at_risk
        curve.append((float(t), round(float(S), 4)))
        if S <= 0.5 and median == 99.0:
            median = float(t)

    survival_at = {}
    for q in query_months:
        Sq = 1.0
        for t in event_times:
            if t > q:
                break
            ar = (T >= t).sum()
            ev = ((T == t) & (E == 1)).sum()
            if ar > 0:
                Sq *= 1 - ev / ar
        survival_at[q] = round(float(Sq), 3)

    return round(median, 2), survival_at, curve


def compute_breakeven(tier, monthly_hazard, avg_mrr):
    gross_pm = avg_mrr * MARGIN
    be_mo    = CAC[tier] / gross_pm if gross_pm > 0 else 99.0
    surv_be  = round(np.exp(-monthly_hazard * be_mo) * 100, 1)
    ltv      = gross_pm / monthly_hazard if monthly_hazard > 0 else 0
    net      = ltv - CAC[tier]
    return {
        "tier":               tier,
        "cac":                CAC[tier],
        "monthly_gp":         round(gross_pm, 2),
        "breakeven_mo":       round(be_mo, 1),
        "pct_survive_to_breakeven": surv_be,
        "ltv":                round(ltv, 0),
        "expected_net_value": round(net, 0),
    }


def run(data_path=None, db_path="saas_intel.db"):
    cust = load_customers(data_path)
    query_months = (1, 3, 6, 9, 12, 18, 24)

    km_rows   = []
    be_rows   = []

    for tier in ["starter", "pro", "enterprise"]:
        m = cust["tier"] == tier
        T = cust.loc[m, "dur"].values
        E = cust.loc[m, "churned"].values
        avg_mrr = cust.loc[m, "mrr"].mean()
        monthly_hazard = E.sum() / T.sum() if T.sum() > 0 else 0

        median, surv_at, curve = km_stats(T, E, query_months)

        # KM curve rows
        for t, s in curve:
            km_rows.append({"tier": tier, "month": t, "survival": s})

        # Breakeven row
        be_row = compute_breakeven(tier, monthly_hazard, avg_mrr)
        be_row["median_survival_mo"] = median
        be_row["survival_at_mo1"]  = surv_at.get(1,  None)
        be_row["survival_at_mo3"]  = surv_at.get(3,  None)
        be_row["survival_at_mo6"]  = surv_at.get(6,  None)
        be_row["survival_at_mo12"] = surv_at.get(12, None)
        be_rows.append(be_row)

        print(f"[{tier:12s}] hazard={monthly_hazard:.4f}/mo  "
              f"avg_mrr=${avg_mrr:.2f}  median={median:.2f}mo  "
              f"BE={be_row['breakeven_mo']}mo  "
              f"surv@BE={be_row['pct_survive_to_breakeven']}%  "
              f"net=${be_row['expected_net_value']:,.0f}")

    km_df = pd.DataFrame(km_rows)
    be_df = pd.DataFrame(be_rows)

    # Write to SQLite
    if os.path.exists(db_path) or db_path == "saas_intel.db":
        try:
            with sqlite3.connect(db_path) as conn:
                cust.to_sql("clean_saas_customers", conn, if_exists="replace", index=False)
                km_df.to_sql("km_survival",   conn, if_exists="replace", index=False)
                be_df.to_sql("cac_breakeven", conn, if_exists="replace", index=False)
            print(f"\nTables written to {db_path}: clean_saas_customers ({len(cust)} rows), "
                  f"km_survival ({len(km_df)} rows), "
                  f"cac_breakeven ({len(be_df)} rows)")
        except Exception as e:
            print(f"DB write skipped ({e}) — results printed above.")

    print("\nSurvival analysis complete.")
    print("For full KM curves and narrative: notebooks/05_survival_analysis-5.ipynb")
    return km_df, be_df


if __name__ == "__main__":
    run()
