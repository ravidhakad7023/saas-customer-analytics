-- ============================================================
-- Query 04: MRR Waterfall Components by Month
-- Purpose : Decompose MRR movement into New, Expansion,
--           Contraction, Churn, and Reactivation.
--           Standard SaaS board reporting metric.
-- ============================================================
WITH monthly_mrr AS (
    SELECT
        strftime('%Y-%m', signup_month) AS month,
        SUM(CASE WHEN churned = 0 THEN mrr ELSE 0 END) AS active_mrr,
        SUM(CASE WHEN churned = 1 THEN mrr ELSE 0 END) AS churned_mrr,
        COUNT(CASE WHEN churned = 0 THEN 1 END)        AS active_customers,
        COUNT(CASE WHEN churned = 1 THEN 1 END)        AS churned_customers,
        COUNT(*)                                        AS total_customers
    FROM clean_saas_customers
    GROUP BY 1
)
SELECT
    month,
    active_mrr,
    churned_mrr,
    active_customers,
    churned_customers,
    ROUND(churned_mrr * 100.0 / NULLIF(active_mrr + churned_mrr, 0), 1) AS mrr_churn_rate_pct,
    ROUND(churned_customers * 100.0 / NULLIF(total_customers, 0), 1)    AS logo_churn_rate_pct
FROM monthly_mrr
ORDER BY month;
