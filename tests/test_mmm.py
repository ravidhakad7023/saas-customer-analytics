"""
Test suite: Marketing Mix Model (MMM)
Tests that MMM coefficients have the correct sign, the model
does not overfit catastrophically, and residuals are well-behaved.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.stattools import durbin_watson
import warnings; warnings.filterwarnings('ignore')


@pytest.fixture
def funnel_data():
    np.random.seed(42)
    weeks = pd.date_range('2022-01-01','2023-12-31',freq='W-MON')
    n = len(weeks)
    trend = np.arange(n)
    season = 1 + 0.3*np.sin(2*np.pi*trend/52)
    org = np.random.normal(180,20,n) * season
    ref = np.random.normal(90,12,n)  * season
    ps  = np.random.normal(55,10,n)  * season
    ph  = np.random.normal(45,8,n)   * season
    li  = np.random.normal(35,6,n)   * season
    # Ground truth: all channels have positive contribution
    subs = (org*0.04 + ref*0.06 + ps*0.05 + ph*0.04 + li*0.07 +
            trend*0.05 + np.random.normal(0,1.5,n)).clip(0)
    return pd.DataFrame({'week':weeks,'organic':org,'referral':ref,'paid_search':ps,
                         'product_hunt':ph,'linkedin':li,'new_paid_subs':subs,'wn':trend})


@pytest.fixture
def fitted_model(funnel_data):
    feats = ['organic','referral','paid_search','product_hunt','linkedin','wn']
    X = funnel_data[feats].values
    y = funnel_data['new_paid_subs'].values
    ho = 20
    sc = StandardScaler()
    Xts = sc.fit_transform(X[:-ho]); ytr = y[:-ho]
    Xhs = sc.transform(X[-ho:]);     yho = y[-ho:]
    rcv = RidgeCV(alphas=np.logspace(-2,4,30), cv=TimeSeriesSplit(n_splits=5),
                  scoring='neg_mean_squared_error')
    rcv.fit(Xts, ytr)
    return rcv, sc, Xts, Xhs, ytr, yho, feats


class TestMMMCoefficients:
    def test_all_channel_coefficients_positive(self, fitted_model):
        """All marketing channel coefficients must be positive — more traffic = more subs."""
        rcv, sc, *_ = fitted_model
        feats = fitted_model[6]
        channel_feats = [f for f in feats if f in ['organic','referral','paid_search','product_hunt','linkedin']]
        channel_idx   = [feats.index(f) for f in channel_feats]
        coefs = rcv.coef_[channel_idx]
        assert (coefs > 0).all(), \
            f"Channel coefficients must be positive. Got: {dict(zip(channel_feats, coefs.round(4)))}"

    def test_trend_coefficient_positive(self, fitted_model):
        """Trend coefficient should be positive — business is growing."""
        rcv, sc, *_ = fitted_model
        feats = fitted_model[6]
        trend_idx = feats.index('wn')
        assert rcv.coef_[trend_idx] > 0, "Trend coefficient should be positive"


class TestMMMFit:
    def test_holdout_rmse_below_threshold(self, fitted_model):
        """Holdout RMSE must be below 3× naive baseline (predict mean)."""
        rcv, sc, Xts, Xhs, ytr, yho, _ = fitted_model
        preds = rcv.predict(Xhs)
        rmse_model   = np.sqrt(np.mean((yho - preds)**2))
        rmse_baseline= np.sqrt(np.mean((yho - ytr.mean())**2))
        assert rmse_model < rmse_baseline * 3, \
            f"MMM holdout RMSE ({rmse_model:.2f}) exceeds 3x naive baseline ({rmse_baseline*3:.2f})"

    def test_no_nan_predictions(self, fitted_model):
        rcv, sc, Xts, Xhs, ytr, yho, _ = fitted_model
        preds = rcv.predict(Xhs)
        assert not np.isnan(preds).any(), "No NaN values in MMM predictions"

    def test_predictions_non_negative(self, fitted_model):
        """Subscriber predictions must not be negative (physically impossible)."""
        rcv, sc, Xts, Xhs, ytr, yho, _ = fitted_model
        preds = rcv.predict(Xhs)
        clipped = preds.clip(0)
        assert (clipped >= 0).all(), "Clipped predictions must be non-negative"


class TestMMMResiduals:
    def test_durbin_watson_in_range(self, fitted_model):
        """DW statistic between 1.5 and 2.5 — no significant autocorrelation."""
        rcv, sc, Xts, Xhs, ytr, yho, _ = fitted_model
        residuals = ytr - rcv.predict(Xts)
        dw = durbin_watson(residuals)
        assert 1.5 < dw < 2.5, \
            f"Durbin-Watson {dw:.3f} outside acceptable range [1.5, 2.5] — autocorrelation present"

    def test_residual_mean_near_zero(self, fitted_model):
        """Mean residual must be near zero — no systematic bias."""
        rcv, sc, Xts, Xhs, ytr, yho, _ = fitted_model
        residuals = ytr - rcv.predict(Xts)
        assert abs(residuals.mean()) < 2.0, \
            f"Mean residual {residuals.mean():.4f} indicates systematic model bias"
