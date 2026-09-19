-- ============================================================
-- Query 05: RFM Segment Distribution and Financial Summary
-- Purpose : Executive-level view of customer health by segment.
--           Used for CS team prioritisation and board reporting.
-- ============================================================
SELECT
    rfm_segment,
    COUNT(*)                            AS n_customers,
    ROUND(AVG(mrr_rfm), 2)            AS avg_mrr,
    ROUND(SUM(mrr_rfm), 2)            AS total_mrr,
    ROUND(SUM(mrr_rfm) * 100.0 /
        (SELECT SUM(mrr_rfm) FROM rfm_scores), 1) AS mrr_share_pct,
    ROUND(AVG(R_score), 2)            AS avg_recency_score,
    ROUND(AVG(F_score), 2)            AS avg_frequency_score,
    ROUND(AVG(M_score), 2)            AS avg_monetary_score,
    ROUND(AVG(RFM_score), 2)         AS avg_composite_score,
    ROUND(AVG(churned) * 100, 1)     AS lifetime_churn_pct
FROM rfm_scores
GROUP BY rfm_segment
ORDER BY avg_composite_score DESC;
