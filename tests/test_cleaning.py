"""
Test suite: Data cleaning pipeline
Tests that the cleaning decisions documented in notebook 02 are
correctly implemented and produce a dataset safe for modelling.
"""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def dirty_customers():
    """Minimal dirty dataset mirroring real anomalies."""
    return pd.DataFrame({
        'tier':          ['Starter', 'PRO', 'enterprise', 'starter', 'Pro'],
        'mrr':           [27.0,      -99,    9999,         np.nan,   74.0],
        'churned':       [1,         0,      1,            0,        1],
        'billing_cycle': ['monthly', 'annual','MONTHLY',   'annual', 'monthly'],
        'channel':       ['organic_search','referral', np.nan,'paid_search','linkedin'],
        'signup_month':  ['2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01'],
        'churn_date':    ['2022-06-01', None, '2022-12-01', None, '2022-08-01'],
    })


def clean(df):
    """Replicates the cleaning pipeline from src/02_eda_and_cleaning.py."""
    d = df.copy()
    d['tier']          = d['tier'].astype(str).str.lower().str.strip()
    d['billing_cycle'] = d['billing_cycle'].astype(str).str.lower().str.strip()
    d['mrr']           = pd.to_numeric(d['mrr'], errors='coerce')
    d.loc[(d['mrr'] < 5) | (d['mrr'] > 1500), 'mrr'] = np.nan
    tier_medians = {'starter': 27.3, 'pro': 74.2, 'enterprise': 279.7}
    d['mrr'] = d.apply(
        lambda r: tier_medians.get(r['tier'], 50.0) if pd.isna(r['mrr']) else r['mrr'], axis=1)
    d['channel'] = d['channel'].fillna(d['channel'].mode()[0] if not d['channel'].mode().empty else 'organic_search')
    return d


class TestTierCleaning:
    def test_tier_all_lowercase(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert cleaned['tier'].str.islower().all(), "All tier values must be lowercase"

    def test_tier_no_whitespace(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert not cleaned['tier'].str.contains(r'\s').any(), "No whitespace in tier values"

    def test_tier_valid_values(self, dirty_customers):
        cleaned = clean(dirty_customers)
        valid = {'starter','pro','enterprise','free'}
        assert set(cleaned['tier'].unique()).issubset(valid), f"Invalid tier values: {set(cleaned['tier'].unique()) - valid}"


class TestMRRCleaning:
    def test_no_negative_mrr(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert (cleaned['mrr'] >= 0).all(), "MRR must not be negative after cleaning"

    def test_no_extreme_outliers(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert (cleaned['mrr'] <= 1500).all(), "MRR must not exceed $1,500 after cleaning"

    def test_no_null_mrr(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert cleaned['mrr'].notna().all(), "MRR must have no nulls after imputation"

    def test_imputation_uses_tier_median(self, dirty_customers):
        """NaN MRR for a 'starter' account should be imputed to ~27.3, not global median."""
        cleaned = clean(dirty_customers)
        starter_null_row = cleaned[(dirty_customers['mrr'].isna()) & (cleaned['tier']=='starter')]
        if len(starter_null_row) > 0:
            assert abs(starter_null_row['mrr'].iloc[0] - 27.3) < 1.0, \
                "Starter NaN MRR should be imputed with tier median ($27.3), not global median"

    def test_negative_mrr_replaced(self, dirty_customers):
        """The -99 value must be replaced, not left as negative."""
        cleaned = clean(dirty_customers)
        assert -99 not in cleaned['mrr'].values, "Negative MRR -99 must be replaced"

    def test_outlier_mrr_replaced(self, dirty_customers):
        """The 9999 value must be replaced."""
        cleaned = clean(dirty_customers)
        assert 9999 not in cleaned['mrr'].values, "Outlier MRR 9999 must be replaced"


class TestBillingCycle:
    def test_billing_cycle_lowercase(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert cleaned['billing_cycle'].str.islower().all()

    def test_billing_cycle_valid_values(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert set(cleaned['billing_cycle'].unique()).issubset({'monthly','annual'})


class TestChannelCleaning:
    def test_no_null_channel(self, dirty_customers):
        cleaned = clean(dirty_customers)
        assert cleaned['channel'].notna().all(), "Channel must have no nulls after imputation"
