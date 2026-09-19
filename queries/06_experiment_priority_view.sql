-- ============================================================
-- Query 06: Experiment Feasibility View
-- Purpose : Quick reference for experiment design decisions.
--           Flags which tests are feasible as standard A/B vs
--           require Bayesian or quasi-experimental methods.
-- ============================================================
SELECT
    action,
    test_type,
    base,
    mde,
    n_per_arm,
    total_n,
    min_wks,
    win,
    guard,
    CASE
        WHEN min_wks <= 12  THEN 'Fast — launch immediately'
        WHEN min_wks <= 26  THEN 'Medium — plan for Q2'
        WHEN min_wks <= 52  THEN 'Slow — consider Bayesian alternative'
        ELSE                     'INFEASIBLE as A/B — use quasi-experiment'
    END AS feasibility_flag
FROM experiment_designs
ORDER BY min_wks ASC;
