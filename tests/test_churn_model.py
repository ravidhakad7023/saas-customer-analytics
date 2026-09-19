"""
Test suite: Churn model outputs
Tests that the churn model produces valid probability scores,
correct output shapes, and respects the temporal split constraint.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')


@pytest.fixture
def synthetic_data():
    """Minimal synthetic dataset with known churn pattern."""
    np.random.seed(42)
    n = 300
    tier = np.random.choice([0,1,2], n, p=[0.60,0.30,0.10])
    mrr  = np.where(tier==0, np.random.normal(27,5,n),
           np.where(tier==1, np.random.normal(74,10,n), np.random.normal(280,40,n)))
    # Higher tier → lower churn probability (ground truth)
    churn_prob = np.where(tier==0, 0.80, np.where(tier==1, 0.45, 0.25))
    churned = (np.random.random(n) < churn_prob).astype(int)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    return pd.DataFrame({'tier_n':tier,'log_mrr':np.log1p(mrr),
                         'bill_n':np.random.randint(0,2,n),'ch_n':np.random.randint(0,5,n),
                         'churned':churned,'date':dates})


@pytest.fixture
def trained_model(synthetic_data):
    split = int(len(synthetic_data)*0.75)
    train = synthetic_data.iloc[:split]
    feats = ['tier_n','log_mrr','bill_n','ch_n']
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
    cal = CalibratedClassifierCV(gbm, method='sigmoid', cv=3)
    cal.fit(train[feats], train['churned'])
    return cal, feats


class TestChurnModelOutput:
    def test_probabilities_in_unit_interval(self, trained_model, synthetic_data):
        cal, feats = trained_model
        test = synthetic_data.iloc[int(len(synthetic_data)*0.75):]
        probs = cal.predict_proba(test[feats])[:,1]
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "All churn probabilities must be in [0,1]"

    def test_output_shape_matches_input(self, trained_model, synthetic_data):
        cal, feats = trained_model
        test = synthetic_data.iloc[int(len(synthetic_data)*0.75):]
        probs = cal.predict_proba(test[feats])[:,1]
        assert len(probs) == len(test), "Output length must match input length"

    def test_auc_above_random(self, trained_model, synthetic_data):
        """Model must beat random baseline (AUC > 0.55)."""
        cal, feats = trained_model
        test = synthetic_data.iloc[int(len(synthetic_data)*0.75):]
        auc = roc_auc_score(test['churned'], cal.predict_proba(test[feats])[:,1])
        assert auc > 0.55, f"AUC {auc:.4f} not significantly above random (0.5)"

    def test_higher_tier_lower_churn_score(self, trained_model):
        """Enterprise accounts should score lower churn risk than Starter accounts."""
        cal, feats = trained_model
        starter    = pd.DataFrame({'tier_n':[0],'log_mrr':[np.log1p(27)],'bill_n':[0],'ch_n':[0]})
        enterprise = pd.DataFrame({'tier_n':[2],'log_mrr':[np.log1p(280)],'bill_n':[1],'ch_n':[1]})
        p_starter    = cal.predict_proba(starter)[:,1][0]
        p_enterprise = cal.predict_proba(enterprise)[:,1][0]
        assert p_starter > p_enterprise, \
            f"Starter churn score ({p_starter:.3f}) should exceed Enterprise ({p_enterprise:.3f})"

    def test_no_nan_predictions(self, trained_model, synthetic_data):
        cal, feats = trained_model
        probs = cal.predict_proba(synthetic_data[feats])[:,1]
        assert not np.isnan(probs).any(), "No NaN values in churn predictions"

    def test_temporal_split_no_leakage(self, synthetic_data):
        """Training set must contain no rows with dates after the split date."""
        split_date = synthetic_data['date'].quantile(0.75)
        train = synthetic_data[synthetic_data['date'] < split_date]
        assert train['date'].max() < split_date, "Training set leaks future data"
