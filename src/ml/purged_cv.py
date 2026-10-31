"""
Purged Walk-Forward Cross-Validation con Embargo.

Implementación propia de la metodología de López de Prado
("Advances in Financial Machine Learning", Cap. 7).

Conceptos clave:
  - Purging:   se eliminan del train las observaciones cuyo LABEL
               se solapa temporalmente con el test set.
               Para horizonte H=5 días, las últimas H filas de train
               se eliminan por cada fold.
  - Embargo:   se añade un buffer adicional entre train y test
               para evitar contaminación por autocorrelación.
               Típico: embargo = 0.01 × tamaño del dataset.

Uso:
    from src.ml.purged_cv import purged_walk_forward_splits

    for i, (train_idx, test_idx) in enumerate(
        purged_walk_forward_splits(df, n_splits=6, horizon=5, embargo=10)
    ):
        X_tr = X.iloc[train_idx]
        ...
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def purged_walk_forward_splits(
    df:       pd.DataFrame,
    n_splits: int = 6,
    horizon:  int = 5,
    embargo:  int = 10,
    min_train_size: float = 0.50,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generador de splits (train_idx, test_idx) para walk-forward purged CV.

    Args:
        df:             DataFrame con índice temporal ordenado.
        n_splits:       Número de folds (ventanas de test).
        horizon:        Días del horizonte de predicción (para purging).
        embargo:        Días de buffer entre train y test (embargo).
        min_train_size: Fracción mínima del total usada para el primer train.

    Yields:
        (train_indices, test_indices) como arrays de posición (iloc).
    """
    n = len(df)
    # Tamaño de cada ventana de test
    test_size = int(n * (1 - min_train_size) / n_splits)
    if test_size < 1:
        test_size = max(1, n // (n_splits + 2))

    for fold in range(n_splits):
        # Test: ventana deslizante
        test_end   = n - fold * test_size
        test_start = test_end - test_size
        if test_start < 0:
            break

        # Train: todo lo anterior al test, con purging + embargo
        purge_end  = test_start           # primer día del test
        train_end  = purge_end - embargo  # embargo gap
        if train_end <= 0:
            continue

        # Purging: eliminar las últimas `horizon` filas de train
        # (sus labels se solapan con el test)
        train_end_purged = train_end - horizon
        if train_end_purged <= 10:
            continue

        train_idx = np.arange(0, train_end_purged)
        test_idx  = np.arange(test_start, test_end)

        yield train_idx, test_idx


def purged_walk_forward_date_splits(
    dates:    pd.Series | pd.Index | np.ndarray,
    n_splits: int = 6,
    horizon:  int = 5,
    embargo:  int = 10,
    min_train_size: float = 0.50,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward purged CV para un PANEL (varias filas por fecha).

    `purged_walk_forward_splits` purga y embarga por número de FILAS. En un
    panel con ~30 tickers por fecha, purgar 5 + 10 filas equivale a menos de
    un día, y train y test llegan a compartir fechas. Esta versión trabaja
    sobre fechas únicas: `horizon` y `embargo` se miden en días hábiles.

    Con el test empezando en la fecha de posición `s`, una fila de train en
    la posición j tiene una etiqueta que usa precios hasta j + horizon, así
    que se exige j + horizon < s, más `embargo` días de separación:
        j < s - horizon - embargo

    Args:
        dates: fecha de cada fila (no necesita estar ordenado).
        Resto: igual que `purged_walk_forward_splits`, pero en días.

    Yields:
        (train_idx, test_idx) como posiciones de fila. Fold 0 = el más reciente.
    """
    row_dates = pd.to_datetime(pd.Series(np.asarray(dates)))
    unique = np.sort(row_dates.unique())
    pos = pd.Series(np.arange(len(unique)), index=unique)
    row_pos = pos.reindex(row_dates.values).to_numpy()

    m = len(unique)
    test_size = int(m * (1 - min_train_size) / n_splits)
    if test_size < 1:
        test_size = max(1, m // (n_splits + 2))

    for fold in range(n_splits):
        test_end = m - fold * test_size
        test_start = test_end - test_size
        if test_start < 0:
            break
        train_cut = test_start - horizon - embargo
        if train_cut <= 10:
            continue
        train_idx = np.flatnonzero(row_pos < train_cut)
        test_idx = np.flatnonzero((row_pos >= test_start) & (row_pos < test_end))
        yield train_idx, test_idx


def compute_embargo_size(n: int, pct: float = 0.01) -> int:
    """Tamaño del embargo como porcentaje del dataset (López de Prado recomienda 1%)."""
    return max(1, int(n * pct))


def deflated_sharpe_ratio(
    sharpe_obs:   float,
    n_trials:     int,
    n_obs:        int,
    skewness:     float = 0.0,
    kurtosis:     float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

    Corrige el Sharpe Ratio por la multiplicidad de estrategias evaluadas.
    Si DSR < 0, la estrategia probablemente no tiene edge real.

    Args:
        sharpe_obs:  SR observado (anualizado)
        n_trials:    Número de configuraciones / estrategias probadas
        n_obs:       Número de observaciones en el test
        skewness:    Skewness de los retornos
        kurtosis:    Kurtosis (3 = normal)

    Returns:
        DSR en escala z (0-1 aprox). >0.95 = evidencia fuerte.
    """
    from scipy import stats

    if n_trials <= 1:
        return float(stats.norm.cdf(sharpe_obs))

    # Máximo SR esperado por azar entre n_trials ensayos
    sr_max = (
        (1 - 0.5772156649)  # Euler–Mascheroni
        * np.sqrt(np.log(n_trials))
        + 0.5772156649 / np.sqrt(np.log(n_trials))
    )

    # Varianza del SR estimado
    var_sr = (
        1 / n_obs
        * (1 + 0.5 * sharpe_obs**2
           - skewness * sharpe_obs
           + (kurtosis - 3) / 4 * sharpe_obs**2)
    )

    dsr = stats.norm.cdf(
        (sharpe_obs - sr_max) / np.sqrt(var_sr + 1e-12)
    )
    return float(dsr)


# ─── LSTM vs LGBM comparison (KB-013) ────────────────────────────────────────

_LGBM_BASE = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.03,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=3.0,
    n_jobs=2,
    random_state=42,
    verbosity=-1,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def build_universe_panel(
    universe: dict[str, str],
    feature_cols: list[str],
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Panel cross-seccional con features, rank-normalización y targets.

    Returns:
        DataFrame multi-index por fecha con columnas de features, rn_*, y_rel, ticker.
    """
    from src.data import store
    from src.data.store import MACRO_TICKERS
    from src.ml.features import (
        add_cross_section_features,
        add_macro_features,
        compute_features,
        compute_target,
    )

    macro = {name: store.get(name) for name in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    frames: dict[str, pd.DataFrame] = {}
    for name, sym in universe.items():
        raw = store.load(sym)
        if raw is None or len(raw) < 150:
            continue
        try:
            feat = compute_features(raw)
            feat = add_macro_features(feat, macro)
            feat["ticker"] = name
            feat["Open"] = raw["Open"].reindex(feat.index)
            feat["Close"] = raw["Close"].reindex(feat.index)
            feat["fwd_ret"] = compute_target(raw, horizon=horizon)
            frames[name] = feat
        except Exception:
            continue

    if not frames:
        raise RuntimeError("Sin datos para el universo")

    frames = add_cross_section_features(frames)
    avail = [c for c in feature_cols if c in next(iter(frames.values())).columns]

    rows = []
    for df in frames.values():
        clean = df.dropna(subset=avail + ["fwd_ret"]).copy()
        if len(clean) >= 100:
            rows.append(clean)

    if not rows:
        raise RuntimeError("Panel vacío tras dropna")

    full = pd.concat(rows).sort_index()
    g = full.groupby(full.index)
    for c in avail:
        full[f"rn_{c}"] = g[c].rank(pct=True)
    full["pct_fwd"] = g["fwd_ret"].rank(pct=True)
    full["y_rel"] = (g["fwd_ret"].rank(pct=True) > 0.5).astype(int)
    full["target_up"] = full["y_rel"]

    rn_avail = [f"rn_{c}" for c in avail]
    full = full.dropna(subset=rn_avail + ["pct_fwd", "target_up"])
    return full


def compare_lstm_vs_lgbm(
    ticker: str,
    universe: dict[str, str],
    feature_cols: list[str] | None = None,
    n_splits: int = 6,
    horizon: int = 5,
    embargo: int = 10,
    sequence_length: int = 30,
    lstm_epochs: int = 40,
    lstm_max_samples: int | None = None,
    lstm_num_layers: int = 2,
    lstm_hidden_units: int = 64,
    lstm_batch_size: int = 32,
) -> dict:
    """
    Comparación side-by-side AUC walk-forward: xs_lgbm vs lstm_seq.

    Args:
        ticker:   Ticker BMV a evaluar (ej. WALMEX, ALSEA, GMEXICOB).
        universe: dict nombre → símbolo yfinance del universo cross-seccional.
        feature_cols: columnas de features (default FEATURE_COLS disponibles).

    Returns:
        Dict con folds y agregados por modelo.
    """
    from src.ml.features import FEATURE_COLS
    from src.ml.lstm_seq import LSTMSeqPredictor

    if ticker not in universe:
        raise ValueError(f"Ticker {ticker!r} no está en el universo")

    feat_cols = feature_cols or FEATURE_COLS
    panel = build_universe_panel(universe, feat_cols, horizon=horizon)
    avail = [c for c in feat_cols if c in panel.columns]
    rn_avail = [f"rn_{c}" for c in avail]

    df_reset = panel.reset_index()

    fold_results: list[dict] = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_splits(df_reset, n_splits=n_splits, horizon=horizon, embargo=embargo)
    ):
        train_df = panel.iloc[tr_idx].copy()
        test_df = panel.iloc[te_idx].copy()
        test_ticker = test_df[test_df["ticker"] == ticker]

        if len(test_ticker) < 5 or test_ticker["y_rel"].nunique() < 2:
            continue

        # ── xs_lgbm: regresor sobre pct_fwd, AUC vs y_rel ────────────────────
        xs_auc = 0.5
        xs_error = ""
        try:
            import lightgbm as lgb

            reg = lgb.LGBMRegressor(**_LGBM_BASE)
            reg.fit(train_df[rn_avail].values, train_df["pct_fwd"].values)
            p_te = reg.predict(test_ticker[rn_avail].values)
            xs_auc = _safe_auc(test_ticker["y_rel"].values, p_te)
        except Exception as exc:
            xs_error = str(exc)

        # ── lstm_seq: clasificación P(up) secuencial ───────────────────────
        lstm_auc = 0.5
        lstm_error = ""
        try:
            pred = LSTMSeqPredictor(
                sequence_length=sequence_length,
                num_layers=lstm_num_layers,
                hidden_units=lstm_hidden_units,
                batch_size=lstm_batch_size,
                epochs=lstm_epochs,
            )
            pred.fit(
                train_df, feature_cols=avail, target_col="target_up",
                max_samples=lstm_max_samples,
            )
            if pred.is_fitted:
                eval_df = pd.concat([train_df, test_df]).sort_index()
                lstm_auc = pred.evaluate_auc(
                    eval_df,
                    avail,
                    target_col="target_up",
                    ticker_filter=ticker,
                    eval_dates=test_ticker.index,
                )
            else:
                lstm_error = pred._last_metrics.get("error", "not_fitted")
        except Exception as exc:
            lstm_error = str(exc)

        test_start = str(test_ticker.index.min())[:10]
        test_end = str(test_ticker.index.max())[:10]

        fold_results.append({
            "fold": fold_i + 1,
            "test_start": test_start,
            "test_end": test_end,
            "n_test_ticker": int(len(test_ticker)),
            "n_train": int(len(tr_idx)),
            "xs_lgbm_auc": round(xs_auc, 4),
            "lstm_seq_auc": round(lstm_auc, 4),
            "lstm_error": lstm_error or None,
            "xs_lgbm_error": xs_error or None,
        })

    if not fold_results:
        return {"ticker": ticker, "error": "Sin folds válidos", "folds": []}

    xs_aucs = [f["xs_lgbm_auc"] for f in fold_results]
    lstm_aucs = [f["lstm_seq_auc"] for f in fold_results]

    return {
        "ticker": ticker,
        "meta": {
            "n_splits": n_splits,
            "horizon": horizon,
            "embargo": embargo,
            "sequence_length": sequence_length,
            "n_features": len(avail),
            "universe_size": len(universe),
        },
        "folds": fold_results,
        "aggregate": {
            "xs_lgbm_mean_auc": round(float(np.mean(xs_aucs)), 4),
            "xs_lgbm_std_auc": round(float(np.std(xs_aucs)), 4),
            "lstm_seq_mean_auc": round(float(np.mean(lstm_aucs)), 4),
            "lstm_seq_std_auc": round(float(np.std(lstm_aucs)), 4),
            "lstm_wins": int(sum(l > x for l, x in zip(lstm_aucs, xs_aucs))),
            "n_folds": len(fold_results),
        },
    }


def _eval_xs_lgbm(train_df, test_ticker, rn_avail) -> tuple[float, str]:
    try:
        import lightgbm as lgb
        reg = lgb.LGBMRegressor(**_LGBM_BASE)
        reg.fit(train_df[rn_avail].values, train_df["pct_fwd"].values)
        p_te = reg.predict(test_ticker[rn_avail].values)
        return _safe_auc(test_ticker["y_rel"].values, p_te), ""
    except Exception as exc:
        return 0.5, str(exc)


def _eval_seq_model(
    predictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_ticker: pd.DataFrame,
    avail: list[str],
    ticker: str,
    max_samples: int | None,
) -> tuple[float, str]:
    try:
        predictor.fit(
            train_df, feature_cols=avail, target_col="target_up", max_samples=max_samples,
        )
        if not predictor.is_fitted:
            return 0.5, predictor._last_metrics.get("error", "not_fitted")
        eval_df = pd.concat([train_df, test_df]).sort_index()
        auc = predictor.evaluate_auc(
            eval_df, avail, target_col="target_up",
            ticker_filter=ticker, eval_dates=test_ticker.index,
        )
        return auc, ""
    except Exception as exc:
        return 0.5, str(exc)


def compare_seq_models(
    ticker: str,
    universe: dict[str, str],
    feature_cols: list[str] | None = None,
    n_splits: int = 6,
    horizon: int = 5,
    embargo: int = 10,
    sequence_length: int = 30,
    epochs: int = 40,
    max_samples: int | None = None,
    lstm_num_layers: int = 2,
    lstm_hidden_units: int = 64,
    lstm_batch_size: int = 32,
    tcn_filters: int = 64,
    tcn_blocks: int = 3,
    include_lstm: bool = True,
    include_tcn: bool = True,
) -> dict:
    """
    Walk-forward purged: xs_lgbm vs lstm_seq vs tcn_seq (A/B/C).
    """
    from src.ml.features import FEATURE_COLS
    from src.ml.lstm_seq import LSTMSeqPredictor
    from src.ml.tcn_seq import TCNSeqPredictor

    if ticker not in universe:
        raise ValueError(f"Ticker {ticker!r} no está en el universo")

    feat_cols = feature_cols or FEATURE_COLS
    panel = build_universe_panel(universe, feat_cols, horizon=horizon)
    avail = [c for c in feat_cols if c in panel.columns]
    rn_avail = [f"rn_{c}" for c in avail]
    df_reset = panel.reset_index()
    fold_results: list[dict] = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_splits(df_reset, n_splits=n_splits, horizon=horizon, embargo=embargo)
    ):
        train_df = panel.iloc[tr_idx].copy()
        test_df = panel.iloc[te_idx].copy()
        test_ticker = test_df[test_df["ticker"] == ticker]
        if len(test_ticker) < 5 or test_ticker["y_rel"].nunique() < 2:
            continue

        xs_auc, xs_err = _eval_xs_lgbm(train_df, test_ticker, rn_avail)
        row: dict = {
            "fold": fold_i + 1,
            "test_start": str(test_ticker.index.min())[:10],
            "test_end": str(test_ticker.index.max())[:10],
            "n_test_ticker": int(len(test_ticker)),
            "xs_lgbm_auc": round(xs_auc, 4),
        }

        if include_lstm:
            lstm = LSTMSeqPredictor(
                sequence_length=sequence_length,
                num_layers=lstm_num_layers,
                hidden_units=lstm_hidden_units,
                batch_size=lstm_batch_size,
                epochs=epochs,
            )
            lstm_auc, lstm_err = _eval_seq_model(
                lstm, train_df, test_df, test_ticker, avail, ticker, max_samples,
            )
            row["lstm_seq_auc"] = round(lstm_auc, 4)
            row["lstm_error"] = lstm_err or None

        if include_tcn:
            tcn = TCNSeqPredictor(
                sequence_length=sequence_length,
                filters=tcn_filters,
                num_tcn_blocks=tcn_blocks,
                lstm_units=lstm_hidden_units,
                batch_size=lstm_batch_size,
                epochs=epochs,
            )
            tcn_auc, tcn_err = _eval_seq_model(
                tcn, train_df, test_df, test_ticker, avail, ticker, max_samples,
            )
            row["tcn_seq_auc"] = round(tcn_auc, 4)
            row["tcn_error"] = tcn_err or None

        row["xs_lgbm_error"] = xs_err or None
        fold_results.append(row)

    if not fold_results:
        return {"ticker": ticker, "error": "Sin folds válidos", "folds": []}

    agg: dict = {"n_folds": len(fold_results)}
    xs_aucs = [f["xs_lgbm_auc"] for f in fold_results]
    agg["xs_lgbm_mean_auc"] = round(float(np.mean(xs_aucs)), 4)
    agg["xs_lgbm_std_auc"] = round(float(np.std(xs_aucs)), 4)

    if include_lstm:
        lstm_aucs = [f["lstm_seq_auc"] for f in fold_results]
        agg["lstm_seq_mean_auc"] = round(float(np.mean(lstm_aucs)), 4)
        agg["lstm_seq_std_auc"] = round(float(np.std(lstm_aucs)), 4)
        agg["lstm_wins_vs_lgbm"] = int(sum(l > x for l, x in zip(lstm_aucs, xs_aucs)))

    if include_tcn:
        tcn_aucs = [f["tcn_seq_auc"] for f in fold_results]
        agg["tcn_seq_mean_auc"] = round(float(np.mean(tcn_aucs)), 4)
        agg["tcn_seq_std_auc"] = round(float(np.std(tcn_aucs)), 4)
        agg["tcn_wins_vs_lgbm"] = int(sum(t > x for t, x in zip(tcn_aucs, xs_aucs)))
        if include_lstm:
            lstm_aucs = [f["lstm_seq_auc"] for f in fold_results]
            agg["tcn_wins_vs_lstm"] = int(sum(t > l for t, l in zip(tcn_aucs, lstm_aucs)))

    seq_keys = [k for k in ("lstm_seq_mean_auc", "tcn_seq_mean_auc") if k in agg]
    if seq_keys:
        best = max(seq_keys, key=lambda k: agg[k])
        agg["best_seq_model"] = best.replace("_mean_auc", "")

    return {
        "ticker": ticker,
        "meta": {
            "n_splits": n_splits,
            "horizon": horizon,
            "embargo": embargo,
            "sequence_length": sequence_length,
            "epochs": epochs,
            "n_features": len(avail),
        },
        "folds": fold_results,
        "aggregate": agg,
    }


def grid_lstm_walkforward(
    ticker: str,
    universe: dict[str, str],
    grid: list[dict],
    n_splits: int = 4,
    horizon: int = 5,
    embargo: int = 10,
    max_samples: int | None = 8000,
) -> dict:
    """Grid search LSTM hyperparams con purged walk-forward (solo lstm vs baseline implícito)."""
    results: list[dict] = []
    for cfg in grid:
        res = compare_lstm_vs_lgbm(
            ticker=ticker,
            universe=universe,
            n_splits=n_splits,
            horizon=horizon,
            embargo=embargo,
            sequence_length=cfg.get("sequence_length", 30),
            lstm_epochs=cfg.get("epochs", 40),
            lstm_max_samples=max_samples,
            lstm_num_layers=cfg.get("num_layers", 2),
            lstm_hidden_units=cfg.get("hidden_units", 64),
            lstm_batch_size=cfg.get("batch_size", 32),
        )
        agg = res.get("aggregate", {})
        results.append({
            **cfg,
            "lstm_mean_auc": agg.get("lstm_seq_mean_auc"),
            "lgbm_mean_auc": agg.get("xs_lgbm_mean_auc"),
            "lstm_wins": agg.get("lstm_wins"),
            "n_folds": agg.get("n_folds"),
        })

    valid = [r for r in results if r.get("lstm_mean_auc") is not None]
    best = max(valid, key=lambda r: r["lstm_mean_auc"]) if valid else None
    return {"ticker": ticker, "grid": results, "best": best}

