-- ============================================================
-- Query 10: Executive KPI Summary — Single Query Board View
-- Purpose : One query that produces the top-line metrics a
--           board or VP needs in a weekly review. Pin this to
--           your BI tool's home dashboard.
-- ============================================================
SELECT
    'Total Customers'      AS metric,
    CAST(COUNT(*) AS TEXT) AS value,
    'count'                AS unit
FROM clean_saas_customers

UNION ALL SELECT 'Active Customers', CAST(SUM(1-churned) AS TEXT), 'count'
FROM clean_saas_customers

UNION ALL SELECT 'Total MRR ($)', CAST(ROUND(SUM(mrr*(1-churned)),0) AS TEXT), 'dollars'
FROM clean_saas_customers

UNION ALL SELECT 'Blended Monthly Churn Rate', CAST(ROUND((SUM(churned)*1.0 / SUM(dur))*100,2)||'%' AS TEXT), 'percent'
FROM clean_saas_customers

UNION ALL SELECT 'Avg MRR per Customer', CAST(ROUND(AVG(mrr),2) AS TEXT), 'dollars'
FROM clean_saas_customers

UNION ALL SELECT 'Enterprise Share of MRR',
    CAST(ROUND(
        SUM(CASE WHEN tier='enterprise' THEN mrr ELSE 0 END)*100.0/SUM(mrr),1
    )||'%' AS TEXT), 'percent'
FROM clean_saas_customers

UNION ALL SELECT 'Annual Billing Rate',
    CAST(ROUND(AVG(CASE WHEN billing_cycle='annual' THEN 1.0 ELSE 0 END)*100,1)||'%' AS TEXT), 'percent'
FROM clean_saas_customers;
