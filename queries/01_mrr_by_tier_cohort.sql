-- ============================================================
-- Query 01: Monthly MRR by Tier and Cohort Month
-- Purpose : Track revenue composition and growth across tiers
--           over time. First query an interviewer will ask for.
-- Usage   : Replace table name if schema differs in production.
-- ============================================================
SELECT
    strftime('%Y-%m', signup_month)   AS cohort_month,
    tier,
    COUNT(*)                          AS customers,
    ROUND(SUM(mrr), 2)               AS total_mrr,
    ROUND(AVG(mrr), 2)               AS avg_mrr,
    ROUND(SUM(churned)*1.0 / SUM(dur), 4) AS monthly_churn_rate,
    ROUND(AVG(churned), 3)           AS lifetime_churn_proportion,
    ROUND(SUM(mrr) * (1 - AVG(churned)), 2) AS retained_mrr
FROM clean_saas_customers
WHERE tier IN ('starter', 'pro', 'enterprise')
GROUP BY 1, 2
ORDER BY 1, 2;
