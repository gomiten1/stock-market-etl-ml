"""Pruebas de las etiquetas alternativas y del CV purgado por fechas."""
import numpy as np
import pandas as pd
import pytest

from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import (
    build_targets, extra_features, forward_realized_vol, forward_return,
    is_regression, past_realized_vol, relative_forward_return, target_horizon,
)
from src.ml.vol_models import HARModel, naive_forecast, qlike, regression_metrics


def _ohlc(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.01, n)) + 0.001
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(1_000, 10_000, n).astype(float),
    }, index=idx)


# ─── Sin mirar al futuro ─────────────────────────────────────────────────────

@pytest.mark.parametrize("h", [5, 21, 63])
def test_labels_do_not_use_prices_beyond_horizon(h):
    """Cambiar precios después de t+h no altera ninguna etiqueta de t."""
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    t = 100
    base = build_targets(df, bench, horizons=(h,), vol_horizons=(5,))
    shocked = df.copy()
    shocked.iloc[t + h + 1:] *= 3.0
    after = build_targets(shocked, bench, horizons=(h,), vol_horizons=(5,))
    for col in (f"dir_{h}", f"rel_{h}"):
        assert base[col].iloc[t] == after[col].iloc[t]


def test_labels_do_not_use_past_prices():
    """Cambiar precios antes de t no altera la etiqueta de rendimiento de t."""
    df = _ohlc()
    t = 200
    shocked = df.copy()
    shocked.iloc[:t] *= 0.5
    assert forward_return(df["Close"], 5).iloc[t] == pytest.approx(
        forward_return(shocked["Close"], 5).iloc[t])


def test_features_are_causal():
    """Cambiar precios futuros no altera ninguna feature de t."""
    df = _ohlc()
    t = 300
    shocked = df.copy()
    shocked.iloc[t + 1:] *= 3.0
    pd.testing.assert_series_equal(extra_features(df).iloc[t],
                                   extra_features(shocked).iloc[t])


def test_forward_vol_uses_exactly_next_h_days():
    df = _ohlc()
    t, h = 150, 5
    base = forward_realized_vol(df, h).iloc[t]
    shocked = df.copy()
    shocked.iloc[t + h + 1:, :4] *= 2.0     # después de la ventana: no cambia
    assert forward_realized_vol(shocked, h).iloc[t] == pytest.approx(base)
    shocked = df.copy()
    shocked.iloc[t + 2, 1] *= 1.05           # High dentro de la ventana: sí cambia
    assert forward_realized_vol(shocked, h).iloc[t] != pytest.approx(base)


# ─── NaN, nunca 0, al final ──────────────────────────────────────────────────

@pytest.mark.parametrize("h", [5, 21, 63])
def test_tail_is_nan(h):
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    out = build_targets(df, bench, horizons=(h,), vol_horizons=(h,))
    for col in (f"dir_{h}", f"rel_{h}", f"log_rv_fwd_{h}"):
        assert out[col].iloc[-h:].isna().all()
        assert out[col].iloc[:-h].notna().all()


def test_binary_labels_are_0_or_1():
    out = build_targets(_ohlc(), _ohlc(seed=1)["Close"])
    for col in ("dir_5", "rel_5", "dir_63", "rel_63"):
        assert set(out[col].dropna().unique()) <= {0.0, 1.0}


# ─── Rendimiento relativo ────────────────────────────────────────────────────

def test_relative_return_against_itself_is_zero():
    c = _ohlc()["Close"]
    rel = relative_forward_return(c, c, 21).dropna()
    assert np.allclose(rel, 0.0)


def test_relative_return_missing_benchmark_date_is_nan():
    df = _ohlc()
    bench = df["Close"].drop(df.index[50])
    rel = relative_forward_return(df["Close"], bench, 5)
    assert np.isnan(rel.iloc[50])          # falta la fecha t
    assert np.isnan(rel.iloc[45])          # falta la fecha t+h


# ─── Volatilidad ─────────────────────────────────────────────────────────────

def test_vol_is_positive_and_annualized():
    rv = past_realized_vol(_ohlc(), 22).dropna()
    assert (rv > 0).all()
    assert 0.01 < rv.median() < 2.0


def test_vol_falls_back_when_high_equals_low():
    df = _ohlc()
    df.loc[df.index[10:20], ["High", "Low"]] = df.loc[df.index[10:20], ["Close"]].values
    rv = past_realized_vol(df, 5)
    assert rv.iloc[25] > 0 and np.isfinite(rv.iloc[25])


def test_har_recovers_persistent_signal():
    df = _ohlc(1000)
    X = extra_features(df).dropna()
    y = X["log_rv_5d"].shift(-5).dropna()
    X = X.loc[y.index]
    pred = HARModel().fit(X, y.to_numpy()).predict(X)
    m = regression_metrics(y.to_numpy(), pred, naive_forecast(X, 5))
    assert m["skill_vs_naive"] > 0


def test_qlike_is_zero_for_perfect_forecast():
    v = np.log(np.array([0.1, 0.2, 0.3]))
    assert qlike(v, v) == pytest.approx(0.0)
    assert qlike(v, v - 0.5) > 0


def test_target_helpers():
    assert target_horizon("rel_63") == 63
    assert target_horizon("log_rv_fwd_21") == 21
    assert is_regression("log_rv_fwd_5") and not is_regression("dir_5")


# ─── CV purgado por fechas ───────────────────────────────────────────────────

@pytest.mark.parametrize("h,embargo", [(5, 10), (63, 63)])
def test_date_splits_never_share_or_touch_dates(h, embargo):
    """En un panel, train y test no comparten fechas y hay h+embargo de hueco."""
    dates = pd.bdate_range("2010-01-01", periods=1500)
    panel = pd.Series(np.repeat(dates, 30))          # 30 tickers por fecha
    unique = pd.Series(np.arange(len(dates)), index=dates)
    n = 0
    for tr, te in purged_walk_forward_date_splits(panel, n_splits=6,
                                                  horizon=h, embargo=embargo):
        n += 1
        tr_pos = unique[panel.iloc[tr]].to_numpy()
        te_pos = unique[panel.iloc[te]].to_numpy()
        assert not set(tr_pos) & set(te_pos)
        assert tr_pos.max() + h + embargo < te_pos.min()
    assert n == 6


def test_date_splits_accept_unsorted_rows():
    dates = pd.Series(np.repeat(pd.bdate_range("2015-01-01", periods=600), 5))
    shuffled = dates.sample(frac=1.0, random_state=0).reset_index(drop=True)
    for tr, te in purged_walk_forward_date_splits(shuffled, n_splits=3):
        assert shuffled.iloc[tr].max() < shuffled.iloc[te].min()
