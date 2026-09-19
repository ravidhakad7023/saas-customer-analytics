# BI Tool Connection Guide
## Connecting Tableau or Power BI to saas_intel.db

### Tableau Desktop
1. Open Tableau Desktop
2. Connect → To a File → Other Databases (JDBC)
   OR: Connect → SQLite (if SQLite connector installed)
3. Navigate to saas_intel.db
4. Available tables: clean_saas_customers, rfm_scores, rfm_segments,
   km_survival, cac_breakeven, experiment_designs, cohort_retention_grid,
   mmm_residual_diag, rd_causal_estimate
5. For cohort heatmap: use cohort_retention_grid table → matrix view

### Power BI Desktop
1. Get Data → More → ODBC
2. DSN: point to saas_intel.db using SQLite ODBC driver
3. Or use: Get Data → Python Script → paste any query from queries/ folder
4. Recommended visuals:
   - cohort_retention_grid → Matrix visual with conditional formatting (red-yellow-green)
   - rfm_segments → Treemap (size=total_mrr, color=churn_rate)
   - km_survival → Line chart (mo_0 through mo_18)
   - cac_breakeven → Clustered bar (expected_net_value by tier)

### Alternative: Metabase (free, open source)
1. docker run -d -p 3000:3000 metabase/metabase
2. Add database: SQLite → upload saas_intel.db
3. All tables auto-discoverable
4. Best free option for demonstrating BI skills without Tableau licence

### Google Looker Studio (free)
1. Upload cohort_retention_grid.csv, rfm_segments.csv to Google Sheets
2. Connect Looker Studio → Google Sheets
3. Build dashboards without any local software
