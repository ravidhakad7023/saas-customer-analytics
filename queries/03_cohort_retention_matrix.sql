-- ============================================================
-- Query 03: Cohort Retention Matrix (Month 0 through Month 12)
-- Purpose : The most common marketing analyst take-home task.
--           Shows what % of each signup cohort is still active
--           at each subsequent month. Foundation of LTV modelling.
-- Note    : This query produces the raw data; pivot in Python/Excel.
-- ============================================================
WITH cohorts AS (
    SELECT
        strftime('%Y-%m', signup_month)   AS cohort_month,
        COUNT(*)                          AS cohort_size
    FROM clean_saas_customers
    GROUP BY 1
),
retained AS (
    SELECT
        strftime('%Y-%m', signup_month)   AS cohort_month,
        CASE
            WHEN churned = 0 THEN 12
            ELSE CAST(
                (julianday(churn_date) - julianday(signup_month)) / 30.0
                AS INTEGER)
        END                               AS months_active,
        COUNT(*)                          AS customers_at_month
    FROM clean_saas_customers
    GROUP BY 1, 2
)
SELECT
    r.cohort_month,
    c.cohort_size,
    r.months_active,
    r.customers_at_month,
    ROUND(
        r.customers_at_month * 100.0 / c.cohort_size, 1
    )                                     AS retention_pct
FROM retained r
JOIN cohorts c ON r.cohort_month = c.cohort_month
WHERE r.months_active BETWEEN 0 AND 12
ORDER BY r.cohort_month, r.months_active;
