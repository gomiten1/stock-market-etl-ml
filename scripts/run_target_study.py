"""
Estudio de objetivos: ¿el límite está en el algoritmo o en la pregunta?

Experimento controlado: mismos datos (acciones BMV del store local), mismas
features y misma validación (Purged Walk-Forward CV por fechas). Lo único
que cambia es la etiqueta:

  Clasificación (métrica: AUC; azar = 0.50)
    dir_5    ¿sube en 5 días?                       ← referencia (la actual)
    rel_5    ¿le gana al IPC en 5 días?
    dir_21   ¿sube en 1 mes?        rel_21  ¿le gana al IPC en 1 mes?
    dir_63   ¿sube en 3 meses?      rel_63  ¿le gana al IPC en 3 meses?

  Regresión (métricas: skill contra el ingenuo, QLIKE, Spearman)
    log_rv_fwd_5   volatilidad de la próxima semana
    log_rv_fwd_21  volatilidad del próximo mes

Modelos: regresión logística / Ridge, LightGBM y XGBoost. Para volatilidad,
además, el ingenuo (vol futura = vol pasada) y HAR-RV.

Uso:
    python scripts/run_target_study.py                  # todo (~10-20 min)
    python scripts/run_target_study.py --fast           # 3 folds, menos árboles
    python scripts/run_target_study.py --targets log_rv_fwd_5 rel_21

Salida: data/eval/target_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from src.data.store import BMV_TICKERS, MACRO_TICKERS  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml.features import (  # noqa: E402
    FEATURE_COLS, add_cross_section_features, add_macro_features, compute_features,
)
from src.ml.purged_cv import purged_walk_forward_date_splits  # noqa: E402
from src.ml.targets import (  # noqa: E402
    EXTRA_FEATURE_COLS, build_targets, extra_features, is_regression, target_horizon,
)
from src.ml.vol_models import HARModel, naive_forecast, regression_metrics  # noqa: E402

OUT_PATH = ROOT / "data" / "eval" / "target_study.json"
SEED = 42
N_JOBS = 4  # fijo para que LightGBM y XGBoost sean reproducibles
MIN_EMBARGO = 10

ALL_TARGETS = ["dir_5", "rel_5", "dir_21", "rel_21", "dir_63", "rel_63",
               "log_rv_fwd_5", "log_rv_fwd_21"]
FEATS = FEATURE_COLS + EXTRA_FEATURE_COLS

warnings.filterwarnings("ignore", category=UserWarning)


# ─── Datos ───────────────────────────────────────────────────────────────────

def build_panel() -> pd.DataFrame:
    """Panel fecha × ticker con features (incl. cs_* y HAR) y todas las etiquetas."""
    macro = {n: store_get(n) for n in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}
    if "NAFTRAC" not in macro:
        raise RuntimeError("Falta data/market/NAFTRAC.MX.parquet (benchmark IPC)")
    bench = macro["NAFTRAC"]["Close"]

    frames: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.DataFrame] = {}
    for name in BMV_TICKERS:
        raw = store_get(name)
        if raw is None or len(raw) < 300:
            continue
        feat = compute_features(raw)
        feat = add_macro_features(feat, macro)
        feat = feat.join(extra_features(raw))
        frames[name] = feat
        labels[name] = build_targets(raw, bench)

    # Las cs_* comparan cada ticker contra el universo en la misma fecha.
    frames = add_cross_section_features(frames)

    rows = []
    for name, feat in frames.items():
        df = feat[FEATS].join(labels[name])
        df["ticker"] = name
        df["date"] = df.index
        rows.append(df.dropna(subset=["log_rv_22d"]))
    return pd.concat(rows, ignore_index=True).sort_values("date", kind="stable")


# ─── Modelos ─────────────────────────────────────────────────────────────────

def classifiers(n_trees: int) -> dict:
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
        "lightgbm": LGBMClassifier(
            n_estimators=n_trees, learning_rate=0.03, num_leaves=15,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1),
        "xgboost": XGBClassifier(
            n_estimators=n_trees, learning_rate=0.03, max_depth=4,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist", random_state=SEED,
            n_jobs=N_JOBS, eval_metric="logloss"),
    }


def regressors(n_trees: int) -> dict:
    return {
        "har": HARModel(),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm": LGBMRegressor(
            n_estimators=n_trees, learning_rate=0.03, num_leaves=15,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
            n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1),
        "xgboost": XGBRegressor(
            n_estimators=n_trees, learning_rate=0.03, max_depth=4,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist", random_state=SEED, n_jobs=N_JOBS),
    }


# ─── Evaluación ──────────────────────────────────────────────────────────────

def evaluate_target(panel: pd.DataFrame, target: str, n_splits: int,
                    n_trees: int) -> dict:
    h = target_horizon(target)
    embargo = max(MIN_EMBARGO, h)
    data = panel.dropna(subset=[target]).reset_index(drop=True)
    X = data[FEATS]
    y = data[target].to_numpy()
    reg = is_regression(target)

    folds: list[dict] = []
    splits = purged_walk_forward_date_splits(
        data["date"], n_splits=n_splits, horizon=h, embargo=embargo)
    for k, (tr, te) in enumerate(splits):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
        test_dates = data["date"].iloc[te]
        fold = {
            "fold": k,
            "test_start": str(test_dates.min().date()),
            "test_end": str(test_dates.max().date()),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            # Con etiquetas a h días, solo ~(días de test / h) periodos son
            # independientes: es la cantidad real de evidencia del fold.
            "n_indep_periods": round(test_dates.nunique() / h, 1),
            "models": {},
        }
        if reg:
            y_naive = naive_forecast(X_te, h)
            fold["models"]["naive"] = regression_metrics(y_te, y_naive, y_naive)
            for name, model in regressors(n_trees).items():
                model.fit(X_tr, y_tr)
                fold["models"][name] = regression_metrics(y_te, model.predict(X_te), y_naive)
        else:
            fold["base_rate_test"] = float(y_te.mean())
            for name, model in classifiers(n_trees).items():
                model.fit(X_tr, y_tr.astype(int))
                p = model.predict_proba(X_te)[:, 1]
                fold["models"][name] = {"auc": float(roc_auc_score(y_te.astype(int), p))}
        folds.append(fold)

    return {
        "target": target,
        "task": "regression" if reg else "classification",
        "horizon_days": h,
        "embargo_days": embargo,
        "n_obs": int(len(data)),
        "summary": summarize(folds, reg),
        "folds": folds,
    }


def summarize(folds: list[dict], reg: bool) -> dict:
    metrics = ["skill_vs_naive", "qlike", "spearman", "r2"] if reg else ["auc"]
    out: dict[str, dict] = {}
    for model in folds[0]["models"]:
        out[model] = {}
        for m in metrics:
            vals = np.array([f["models"][model][m] for f in folds], dtype=float)
            out[model][f"{m}_mean"] = round(float(np.nanmean(vals)), 4)
            out[model][f"{m}_std"] = round(float(np.nanstd(vals)), 4)
    return out


def print_result(res: dict) -> None:
    s = res["summary"]
    indep = sum(f["n_indep_periods"] for f in res["folds"])
    print(f"\n── {res['target']}  (horizonte {res['horizon_days']}d, "
          f"embargo {res['embargo_days']}d, {res['n_obs']:,} obs, "
          f"~{indep:.0f} periodos independientes en test)")
    if res["task"] == "classification":
        print(f"   {'modelo':<10} {'AUC':>14}")
        for model, v in s.items():
            print(f"   {model:<10} {v['auc_mean']:.4f} ± {v['auc_std']:.4f}")
    else:
        print(f"   {'modelo':<10} {'skill vs ing.':>14} {'QLIKE':>8} {'Spearman':>9} {'R²':>7}")
        for model, v in s.items():
            print(f"   {model:<10} {v['skill_vs_naive_mean']:>+14.4f} "
                  f"{v['qlike_mean']:>8.4f} {v['spearman_mean']:>9.4f} {v['r2_mean']:>7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de objetivos alternativos")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--fast", action="store_true", help="3 folds y 100 árboles")
    args = parser.parse_args()
    if args.fast:
        args.splits, args.trees = 3, 100

    t0 = time.time()
    print("=" * 66)
    print("  ESTUDIO DE OBJETIVOS  |  Purged Walk-Forward CV por fechas")
    print(f"  {args.splits} folds  |  {args.trees} árboles  |  LR · LightGBM · XGBoost")
    print("=" * 66)
    print("\nConstruyendo panel BMV...")
    panel = build_panel()
    print(f"   {len(panel):,} filas · {panel['ticker'].nunique()} tickers · "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    results = []
    for target in args.targets:
        res = evaluate_target(panel, target, args.splits, args.trees)
        print_result(res)
        results.append(res)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"splits": args.splits, "trees": args.trees, "seed": SEED,
                   "n_jobs": N_JOBS, "features": FEATS},
        "results": results,
    }, indent=2))
    print(f"\nGuardado: {OUT_PATH.relative_to(ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
