-- ============================================================
-- Query 08: Channel Quality Ranking by Retention-Adjusted MRR
-- Purpose : Rank acquisition channels by the MRR they retain,
--           not just the volume they bring. Feeds budget decisions.
-- ============================================================
SELECT
    channel,
    COUNT(*)                                   AS customers_acquired,
    ROUND(AVG(mrr), 2)                        AS avg_mrr,
    ROUND((SUM(churned)*1.0 / SUM(dur)) * 100, 2) AS monthly_churn_rate_pct,
    ROUND(AVG(churned) * 100, 1)             AS lifetime_churn_pct,
    ROUND(AVG(mrr) * (1 - AVG(churned)) * 12, 2) AS implied_annual_ltv,
    ROUND(
        RANK() OVER (ORDER BY AVG(mrr) * (1 - AVG(churned)) DESC)
    , 0)                                       AS quality_rank
FROM clean_saas_customers
WHERE channel IS NOT NULL
GROUP BY channel
HAVING COUNT(*) >= 10
ORDER BY implied_annual_ltv DESC;
