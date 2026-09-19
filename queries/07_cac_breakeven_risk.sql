-- ============================================================
-- Query 07: CAC Breakeven Risk by Tier
-- Purpose : Shows which tiers are destroying capital by failing
--           to survive to their CAC payback month.
--           The most financially consequential query in the project.
-- ============================================================
SELECT
    b.tier,
    b.cac,
    b.monthly_gp,
    b.breakeven_mo,
    b.pct_survive_to_breakeven,
    b.expected_net_value,
    CASE
        WHEN b.expected_net_value < 0    THEN 'DESTROYING CAPITAL'
        WHEN b.expected_net_value < 50   THEN 'MARGINALLY PROFITABLE'
        ELSE                                  'PROFITABLE'
    END AS capital_status,
    b.survival_at_mo6,
    b.survival_at_mo12
FROM cac_breakeven b
ORDER BY b.expected_net_value ASC;
