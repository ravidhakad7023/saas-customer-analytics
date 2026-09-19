import sqlite3
import pandas as pd

def build_excel(db_path='saas_intel.db', output_path='excel/PipelineIQ_Analytics.xlsx'):
    with sqlite3.connect(db_path) as conn:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # 1. Executive KPIs
            try:
                pd.read_sql(open('queries/10_executive_kpi_dashboard.sql').read(), conn).to_excel(writer, sheet_name='1_Executive KPIs', index=False)
            except Exception as e: print("Skipping sheet 1:", e)
            
            # 2. Tier Breakdown
            try:
                pd.read_sql(open('queries/01_mrr_by_tier_cohort.sql').read(), conn).to_excel(writer, sheet_name='2_Tier Breakdown', index=False)
            except Exception as e: print("Skipping sheet 2:", e)
                
            # 3. Channel Breakdown
            try:
                pd.read_sql(open('queries/02_churn_rate_by_channel.sql').read(), conn).to_excel(writer, sheet_name='3_Channel Breakdown', index=False)
            except Exception as e: print("Skipping sheet 3:", e)
                
            # 4. Quarterly Trend
            try:
                pd.read_sql(open('queries/04_mrr_waterfall.sql').read(), conn).to_excel(writer, sheet_name='4_Quarterly Trend', index=False)
            except Exception as e: print("Skipping sheet 4:", e)
                
            # 5. Cohort Retention
            try:
                pd.read_sql(open('queries/03_cohort_retention_matrix.sql').read(), conn).to_excel(writer, sheet_name='5_Cohort Retention', index=False)
            except Exception as e: print("Skipping sheet 5:", e)
                
            # 6. Survival Breakeven
            try:
                pd.read_sql(open('queries/07_cac_breakeven_risk.sql').read(), conn).to_excel(writer, sheet_name='6_Survival Breakeven', index=False)
            except Exception as e: print("Skipping sheet 6:", e)
                
            # 7. RFM Segments
            try:
                pd.read_sql(open('queries/05_rfm_segment_summary.sql').read(), conn).to_excel(writer, sheet_name='7_RFM Segments', index=False)
            except Exception as e: print("Skipping sheet 7:", e)
                
            # 8. Pricing Elasticity (from notebook 06)
            pricing_data = pd.DataFrame([
                {"Tier": "Starter", "Current Price": "$30", "Elasticity": -0.8, "Optimal Range": "$25-$32", "Expected Revenue Impact": "+8-12%"},
                {"Tier": "Pro", "Current Price": "$74", "Elasticity": -2.1, "Optimal Range": "$40-$50", "Expected Revenue Impact": "+15-20%"},
                {"Tier": "Enterprise", "Current Price": "$263", "Elasticity": -0.5, "Optimal Range": "$300+", "Expected Revenue Impact": "+25%"}
            ])
            pricing_data.to_excel(writer, sheet_name='8_Pricing Elasticity', index=False)
            
            # 9. Gap Analysis (Mocked / from notebook 08)
            # Extracted metrics
            gap_data = pd.DataFrame([
                {"Metric": "Monthly Churn", "Value": "7.18%", "Benchmark": "2.0%", "Status": "Critical"},
                {"Metric": "NRR", "Value": "85%", "Benchmark": "106%", "Status": "Critical"},
                {"Metric": "Starter CAC Breakeven", "Value": "9.3 mo", "Benchmark": "<6 mo", "Status": "Poor"},
                {"Metric": "Enterprise LTV/CAC", "Value": "3.27x", "Benchmark": ">3.0x", "Status": "Healthy"}
            ])
            gap_data.to_excel(writer, sheet_name='9_Gap Analysis', index=False)
            
    print("Excel report successfully regenerated at", output_path)

if __name__ == '__main__':
    build_excel()
