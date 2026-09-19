-- ============================================================
-- Query 09: Annual vs Monthly Billing — Retention Comparison
-- Purpose : Naive comparison to motivate the causal RD analysis.
--           Shows correlation; RD notebook shows causation.
-- NOTE    : Selection bias present. Annual customers self-select.
--           Do not use this alone to justify billing cycle decisions.
--           See notebooks/10_causal_inference.ipynb for valid estimate.
-- ============================================================
SELECT
    billing_cycle,
    tier,
    COUNT(*)                          AS n_customers,
    ROUND((SUM(churned)*1.0 / SUM(dur)) * 100, 2) AS monthly_churn_rate_pct,
    ROUND(AVG(churned) * 100, 1)    AS lifetime_churn_pct,
    ROUND(AVG(mrr), 2)              AS avg_mrr,
    ROUND(AVG(mrr) * (1 - AVG(churned)) * 12, 2) AS implied_annual_ltv,
    ROUND(
        AVG(CASE WHEN churned=0 THEN mrr ELSE 0 END) /
        NULLIF(AVG(mrr), 0) * 100, 1
    )                                 AS retained_mrr_pct
FROM clean_saas_customers
GROUP BY billing_cycle, tier
ORDER BY tier, billing_cycle;
