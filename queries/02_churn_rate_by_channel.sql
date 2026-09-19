-- ============================================================
-- Query 02: Churn Rate and Average MRR by Acquisition Channel
-- Purpose : Identify which channels bring the highest-quality
--           customers (low churn, high MRR). Feeds LTV:CAC.
-- Business: If paid_search has 2x churn of referral at same CAC,
--           referral budget should increase.
-- ============================================================
SELECT
    channel,
    COUNT(*)                        AS total_customers,
    SUM(churned)                    AS churned_count,
    ROUND((SUM(churned)*1.0 / SUM(dur)) * 100, 2) AS monthly_churn_rate_pct,
    ROUND(AVG(churned) * 100, 1)   AS lifetime_churn_pct,
    ROUND(AVG(mrr), 2)             AS avg_mrr,
    ROUND(AVG(mrr) * (1 - AVG(churned)), 2) AS expected_retained_mrr,
    ROUND(
        AVG(mrr) /
        NULLIF(SUM(churned)*1.0 / SUM(dur), 0), 2
    )                               AS implied_ltv_proxy
FROM clean_saas_customers
WHERE channel IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 10
ORDER BY monthly_churn_rate_pct ASC;
