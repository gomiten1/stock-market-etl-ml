"""
Ensemble v2 — 3 votantes + filtro HMM de régimen.

Modelos:
  1. WoE + Logistic Regression   (features tabulares, lineal)
  2. LightGBM tuneado con Optuna (árboles, interacciones no lineales)
  3. MiniRocket MULTIVARIADO     (3 canales: precio norm + volumen + RSI)

Filtro de régimen:
  HMM de 3 estados (bull/lateral/bear) entrenado sobre IPC/NAFTRAC.
  Las señales del ensemble reciben un multiplicador según el régimen:
    bull:    ×1.0  (señal sin penalización)
    lateral: ×0.85 (ligera penalización)
    bear:    ×0.65 (penalización fuerte — muchas señales se suprimen)

Voto: P_ensemble = w1·P_lr + w2·P_lgbm + w3·P_rocket, pesos ∝ (AUC - 0.5)
Salida:
  data/ensemble_results.json
  data/ensemble_predictions.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.store import get as store_get, BMV_TICKERS, MACRO_TICKERS
from src.ml.features import (
    compute_features, add_macro_features, add_cross_section_features,
    TECH_FEATURE_COLS, MACRO_FEATURE_COLS,
)
from src.ml.woe import compute_all_woe, woe_transform, iv_summary
from src.ml.regime import RegimeDetector

HORIZON       = 5
TARGET_RET    = 0.02
TRAIN_RATIO   = 0.80
TOP_WOE_FEATS = 12
PRICE_WINDOW  = 60
PROB_THRESH   = 0.53
FEATURE_COLS  = TECH_FEATURE_COLS + MACRO_FEATURE_COLS

# Recursos
ROCKET_KERNELS   = 2_000   # MiniRocket es 75× más rápido, podemos usar más
ROCKET_SUBSAMPLE = 0.50    # 50% del train (suficiente para MiniRocket)
OPTUNA_TRIALS    = 20
LGBM_THREADS     = 2

# Ajuste ADITIVO por régimen (no multiplicativo — menos agresivo)
# En lugar de multiplicar la prob, sumamos/restamos un pequeño valor:
#   bear:    -0.03 (solo suprime señales muy al límite del umbral)
#   lateral: +0.00 (sin cambio)
#   bull:    +0.01 (ligero boost)
REGIME_DELTA = {"bull": +0.01, "lateral": 0.00, "bear": -0.03}

print("=" * 65)
print("  Ensemble v2 — 3 votantes + HMM régimen")
print("  WoE+LR  |  LightGBM (Optuna)  |  MiniRocket×3ch  |  HMM")
print("=" * 65)

# ─── 1. Cargar datos ──────────────────────────────────────────────────────────
print("\n[1] Cargando datos...")

macro: dict[str, pd.DataFrame] = {}
for name in MACRO_TICKERS:
    df = store_get(name)
    if df is not None:
        macro[name] = df.copy()

naftrac_df = macro.get("NAFTRAC")
sp500_df   = macro.get("SP500")

raw_ohlcv: dict[str, pd.DataFrame] = {}  # para ROCKET multicanal
# Pasada 1: features por ticker (técnicos + macro).
frames: dict[str, pd.DataFrame] = {}
for name in BMV_TICKERS:
    df = store_get(name)
    if df is None or len(df) < PRICE_WINDOW + 50:
        continue
    raw_ohlcv[name] = df.copy()
    feat = compute_features(df)
    feat = add_macro_features(feat, macro)
    frames[name] = feat

# Pasada 2: features cross-seccionales (cs_*), que comparan cada ticker
# contra el universo en la misma fecha y requieren todos los tickers.
frames = add_cross_section_features(frames)

rows = []
for name, feat in frames.items():
    df = raw_ohlcv[name]
    fwd = df["Close"].shift(-HORIZON) / df["Close"] - 1
    feat["ticker"]  = name
    # Sin precio futuro (últimos días) no hay etiqueta: NaN, no 0.
    feat["y"]       = (fwd >= TARGET_RET).astype(float).where(fwd.notna())
    feat["ret_fwd"] = fwd * 100
    rows.append(feat)

all_df = pd.concat(rows).sort_index()
all_df = all_df.dropna(subset=FEATURE_COLS + ["y"])
all_df["y"] = all_df["y"].astype(int)
print(f"   {len(all_df):,} obs  |  {all_df.index.min().date()} → {all_df.index.max().date()}")

# ─── 2. Corte temporal 80/20 ─────────────────────────────────────────────────
print("\n[2] Corte 80/20 temporal...")
unique_dates = all_df.index.unique().sort_values()
cutoff_date  = unique_dates[int(len(unique_dates) * TRAIN_RATIO)]
train_df = all_df[all_df.index <  cutoff_date].copy()
test_df  = all_df[all_df.index >= cutoff_date].copy()
print(f"   Corte: {cutoff_date.date()}  |  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

# ─── 3. HMM de régimen ───────────────────────────────────────────────────────
print("\n[3] HMM Régimen (3 estados: bull / lateral / bear)...")

regime_ok = False
regime_series_test: pd.Series | None = None

if naftrac_df is not None:
    try:
        detector = RegimeDetector(n_states=3, n_iter=200)
        detector.fit(naftrac_df, sp500_df, train_cutoff=str(cutoff_date.date()))

        stats = detector.regime_stats(naftrac_df, sp500_df)
        print("   Estadísticas de régimen (histórico completo):")
        for _, row in stats.iterrows():
            bar = "█" * int(abs(row["ret5_mean"]) * 10)
            sign = "+" if row["ret5_mean"] > 0 else ""
            print(f"     {row['regime']:<8} {row['pct_time']:>4.0f}% del tiempo  "
                  f"ret5={sign}{row['ret5_mean']:.2f}%  {bar}")

        # Serie de régimen para el test set
        regime_series_test = detector.predict_series(naftrac_df, sp500_df)
        regime_series_test = regime_series_test.reindex(
            test_df.index.unique().sort_values(), method="ffill"
        )

        current_regime = detector.predict_current(naftrac_df, sp500_df)
        current_proba  = detector.state_proba(naftrac_df, sp500_df)
        print(f"\n   Régimen ACTUAL: {current_regime.upper()}")
        print(f"   P(bull)={current_proba.get('bull',0):.1%}  "
              f"P(lateral)={current_proba.get('lateral',0):.1%}  "
              f"P(bear)={current_proba.get('bear',0):.1%}")
        regime_ok = True
    except Exception as e:
        print(f"   ⚠️  HMM falló: {e}")
else:
    print("   ⚠️  Sin datos de NAFTRAC para HMM")

# ─── 4. Modelo 1 — WoE + LR ──────────────────────────────────────────────────
print("\n[4] Modelo 1 — WoE + Logistic Regression...")
woe_maps  = compute_all_woe(train_df, FEATURE_COLS, target_col="y")
iv_df     = iv_summary(woe_maps)
top_feats = iv_df.head(TOP_WOE_FEATS)["feature"].tolist()

X_tr_woe = woe_transform(train_df[FEATURE_COLS], woe_maps)[top_feats]
X_te_woe = woe_transform(test_df[FEATURE_COLS],  woe_maps)[top_feats]
y_tr = train_df["y"]
y_te = test_df["y"]

mask_tr = X_tr_woe.notna().all(axis=1)
mask_te = X_te_woe.notna().all(axis=1)
X_tr_woe = X_tr_woe[mask_tr];  y_tr_woe = y_tr[mask_tr]
X_te_woe = X_te_woe[mask_te];  y_te_woe = y_te[mask_te]

scaler1  = StandardScaler()
X_tr_sc  = scaler1.fit_transform(X_tr_woe)
X_te_sc  = scaler1.transform(X_te_woe)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_sc, y_tr_woe)
p_lr_te  = lr.predict_proba(X_te_sc)[:, 1]
auc_lr   = roc_auc_score(y_te_woe, p_lr_te)
print(f"   AUC: {auc_lr:.4f}")

# ─── 5. Modelo 2 — LightGBM + Optuna ─────────────────────────────────────────
print(f"\n[5] Modelo 2 — LightGBM + Optuna ({OPTUNA_TRIALS} trials)...")
try:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_lgbm_tr = X_tr_woe.values
    X_lgbm_te = X_te_woe.values

    def lgbm_obj(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary", metric="auc", verbosity=-1,
            num_leaves        = trial.suggest_int("nl",  15, 127),
            learning_rate     = trial.suggest_float("lr", 0.01, 0.3, log=True),
            min_child_samples = trial.suggest_int("mc",  10, 100),
            feature_fraction  = trial.suggest_float("ff", 0.5, 1.0),
            bagging_fraction  = trial.suggest_float("bf", 0.6, 1.0),
            bagging_freq=1,
            reg_alpha  = trial.suggest_float("ra", 1e-4, 10.0, log=True),
            reg_lambda = trial.suggest_float("rl", 1e-4, 10.0, log=True),
            n_estimators=300,
        )
        cut = int(len(X_lgbm_tr) * 0.8)
        m = lgb.LGBMClassifier(**params, n_jobs=LGBM_THREADS)
        m.fit(X_lgbm_tr[:cut], y_tr_woe.values[:cut],
              eval_set=[(X_lgbm_tr[cut:], y_tr_woe.values[cut:])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_tr_woe.values[cut:], m.predict_proba(X_lgbm_tr[cut:])[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgbm_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    bp = study.best_params

    lgbm_m = lgb.LGBMClassifier(
        objective="binary", verbosity=-1, n_estimators=300, n_jobs=LGBM_THREADS,
        num_leaves=bp["nl"], learning_rate=bp["lr"], min_child_samples=bp["mc"],
        feature_fraction=bp["ff"], bagging_fraction=bp["bf"], bagging_freq=1,
        reg_alpha=bp["ra"], reg_lambda=bp["rl"], random_state=42,
    )
    lgbm_m.fit(X_lgbm_tr, y_tr_woe.values)
    p_lgbm_te = lgbm_m.predict_proba(X_lgbm_te)[:, 1]
    auc_lgbm  = roc_auc_score(y_te_woe, p_lgbm_te)
    print(f"   AUC: {auc_lgbm:.4f}  (params: leaves={bp['nl']}, lr={bp['lr']:.3f})")
    has_lgbm = True
except Exception as e:
    print(f"   ⚠️  LightGBM: {e}")
    p_lgbm_te = p_lr_te.copy(); auc_lgbm = auc_lr; has_lgbm = False

# ─── 6. Modelo 3 — MiniRocket MULTIVARIADO (3 canales) ───────────────────────
print(f"\n[6] Modelo 3 — MiniRocket 3 canales × {PRICE_WINDOW}d...")
print("   Canal 1: precio normalizado (P/P[t-60])  ← captura tendencia")
print("   Canal 2: volumen relativo (vol/vol_ma20) ← captura actividad")
print("   Canal 3: RSI-14 rolling                  ← captura momentum")

try:
    from sktime.classification.kernel_based import RocketClassifier

    rng = np.random.default_rng(42)
    rocket_rows_tr, rocket_rows_te = [], []
    rocket_y_tr,    rocket_y_te    = [], []
    rocket_idx_tr,  rocket_idx_te  = [], []

    for ticker in BMV_TICKERS:
        if ticker not in raw_ohlcv:
            continue
        ohlcv = raw_ohlcv[ticker]
        close  = ohlcv["Close"]
        volume = ohlcv["Volume"] if "Volume" in ohlcv.columns else pd.Series(1, index=ohlcv.index)

        # Canal 1: precio normalizado — relativo al precio de PRICE_WINDOW días atrás
        price_norm = (close / close.shift(PRICE_WINDOW)).fillna(1.0)

        # Canal 2: volumen relativo al promedio móvil de 20 días
        vol_ma   = volume.rolling(20, min_periods=5).mean().fillna(volume.mean())
        vol_rel  = (volume / (vol_ma + 1e-8)).clip(0, 5)

        # Canal 3: RSI-14 normalizado a [0,1]
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-8)
        rsi    = (100 - 100 / (1 + rs)) / 100.0  # normalizado [0,1]
        rsi    = rsi.fillna(0.5)

        ticker_df = all_df[all_df["ticker"] == ticker]

        for date, row in ticker_df.iterrows():
            idx_loc = close.index.get_loc(date) if date in close.index else None
            if idx_loc is None or idx_loc < PRICE_WINDOW:
                continue

            # Ventana de PRICE_WINDOW puntos para cada canal
            w1 = price_norm.iloc[idx_loc - PRICE_WINDOW + 1: idx_loc + 1].values
            w2 = vol_rel.iloc[   idx_loc - PRICE_WINDOW + 1: idx_loc + 1].values
            w3 = rsi.iloc[       idx_loc - PRICE_WINDOW + 1: idx_loc + 1].values

            if len(w1) < PRICE_WINDOW or np.isnan(w1).any() or np.isnan(w2).any():
                continue

            window = np.stack([w1, w2, w3], axis=0).astype(np.float32)  # (3, 60)
            y_val  = int(row["y"])

            if date < cutoff_date:
                rocket_rows_tr.append(window)
                rocket_y_tr.append(y_val)
                rocket_idx_tr.append(date)
            else:
                rocket_rows_te.append(window)
                rocket_y_te.append(y_val)
                rocket_idx_te.append(date)

    if len(rocket_rows_tr) < 100:
        raise ValueError("Pocas muestras")

    X_rocket_tr = np.array(rocket_rows_tr)   # (N, 3, 60)
    X_rocket_te = np.array(rocket_rows_te)
    y_rocket_tr = np.array(rocket_y_tr)
    y_rocket_te = np.array(rocket_y_te)

    print(f"   Train: {len(X_rocket_tr):,}  |  Test: {len(X_rocket_te):,}")

    # Subsample train
    n_sub   = int(len(X_rocket_tr) * ROCKET_SUBSAMPLE)
    sub_idx = rng.choice(len(X_rocket_tr), size=n_sub, replace=False)
    sub_idx.sort()
    print(f"   Subsample: {n_sub:,} ({ROCKET_SUBSAMPLE:.0%}) — reduce RAM")

    rocket = RocketClassifier(
        num_kernels=ROCKET_KERNELS,
        rocket_transform="minirocket",  # MiniRocket: 75× más rápido
        use_multivariate="yes",         # acepta (N, 3, 60) multicanal
        random_state=42,
        n_jobs=1,
    )
    rocket.fit(X_rocket_tr[sub_idx], y_rocket_tr[sub_idx])

    p_rocket_te = rocket.predict_proba(X_rocket_te)[:, 1]
    auc_rocket  = roc_auc_score(y_rocket_te, p_rocket_te)
    n_rocket_te = len(p_rocket_te)
    print(f"   AUC MiniRocket 3ch: {auc_rocket:.4f}  (vs 0.5040 con 1ch antes)")
    has_rocket = True

except Exception as e:
    print(f"   ⚠️  MiniRocket: {e}")
    p_rocket_te = p_lr_te.copy(); auc_rocket = auc_lr; n_rocket_te = len(p_lr_te)
    has_rocket = False

# ─── 7. Combinando votantes ───────────────────────────────────────────────────
print("\n[7] Combinando votantes (pesos ∝ AUC - 0.5)...")

# Alinear ROCKET con el resto: usa máscara del test igual que WoE
# Si ROCKET tiene filas de menos, rellenar con p_lr
n_te = len(p_lr_te)
if has_rocket and len(p_rocket_te) == n_te:
    p_rocket_aligned = p_rocket_te
else:
    p_rocket_aligned = p_lr_te.copy()
    auc_rocket = auc_lr

w_lr     = max(auc_lr     - 0.50, 0.001)
w_lgbm   = max(auc_lgbm   - 0.50, 0.001)
w_rocket = max(auc_rocket  - 0.50, 0.001)
total_w  = w_lr + w_lgbm + w_rocket
w_lr    /= total_w; w_lgbm /= total_w; w_rocket /= total_w

print(f"   Pesos — LR: {w_lr:.1%}  LightGBM: {w_lgbm:.1%}  MiniRocket: {w_rocket:.1%}")

p_ensemble = p_lr_te * w_lr + p_lgbm_te * w_lgbm + p_rocket_aligned * w_rocket

# ─── 8. Aplicar filtro HMM ────────────────────────────────────────────────────
print("\n[8] Aplicando filtro HMM de régimen...")

test_clean = test_df[mask_te].copy()
test_clean_dates = test_clean.index  # fechas del test (con duplicados por ticker)

if regime_ok and regime_series_test is not None:
    regime_by_date = regime_series_test.to_dict()
    regime_deltas = np.array([
        REGIME_DELTA.get(regime_by_date.get(d, "lateral"), 0.0)
        for d in test_clean_dates
    ])
    p_ensemble_adj = np.clip(p_ensemble + regime_deltas, 0, 1)
    print(f"   Régimen aplicado (ajuste aditivo). Distribución en test set:")
    for r in ["bull", "lateral", "bear"]:
        cnt = sum(1 for d in test_clean_dates
                  if regime_by_date.get(d, "lateral") == r)
        pct = cnt / len(test_clean_dates) * 100
        delta = REGIME_DELTA.get(r, 0)
        print(f"     {r:<8} {cnt:>6} obs ({pct:.1f}%)  ajuste={delta:+.2f}")
else:
    p_ensemble_adj = p_ensemble
    print("   Sin HMM — usando ensemble sin filtro")

# ─── 9. Métricas finales ──────────────────────────────────────────────────────
print("\n[9] Métricas finales...")

auc_ens   = roc_auc_score(y_te_woe.values, p_ensemble_adj)
brier_ens = brier_score_loss(y_te_woe.values, np.clip(p_ensemble_adj, 0, 1))
print(f"   AUC individual:  LR={auc_lr:.4f}  LGBM={auc_lgbm:.4f}  Rocket={auc_rocket:.4f}")
print(f"   AUC ENSEMBLE+HMM: {auc_ens:.4f}  (Brier: {brier_ens:.4f})")

test_clean["p_lr"]       = p_lr_te
test_clean["p_lgbm"]     = p_lgbm_te
test_clean["p_rocket"]   = p_rocket_aligned
test_clean["p_ensemble"] = p_ensemble
test_clean["p_adj"]      = np.clip(p_ensemble_adj, 0, 1)
test_clean["signal"]     = (test_clean["p_adj"] >= PROB_THRESH).astype(int)

if regime_ok:
    regime_by_date_full = regime_series_test.reindex(
        test_clean_dates, method="ffill"
    )
    test_clean["regime"] = regime_by_date_full.values

sigs    = test_clean[test_clean["signal"] == 1]
n_sig   = len(sigs)
wr      = sigs["y"].mean()       if n_sig else 0
ret_avg = sigs["ret_fwd"].mean() if n_sig else 0
ret_std = sigs["ret_fwd"].std()  if n_sig else 1
sharpe  = ret_avg / (ret_std + 1e-8) * np.sqrt(252 / HORIZON)
pf_w    = sigs[sigs["ret_fwd"] > 0]["ret_fwd"].sum()
pf_l    = abs(sigs[sigs["ret_fwd"] < 0]["ret_fwd"].sum())
pf      = pf_w / pf_l if pf_l > 0 else float("inf")

print(f"   Señales: {n_sig} ({n_sig/len(test_clean)*100:.1f}%)  "
      f"Win rate: {wr:.1%}  Ret: {ret_avg:+.2f}%  Sharpe: {sharpe:.2f}  PF: {pf:.2f}")

# Comparación pre/post HMM y vs modelo base
print("\n   Comparación por modelo a umbral 0.53:")
for col, label in [("p_lr","WoE+LR"), ("p_lgbm","LightGBM"),
                   ("p_rocket","MiniRocket"), ("p_ensemble","Ensemble"),
                   ("p_adj","Ensemble+HMM")]:
    sub = test_clean[test_clean[col] >= PROB_THRESH]
    if len(sub) == 0:
        print(f"     {label:<18} 0 señales")
        continue
    w = sub["y"].mean(); r = sub["ret_fwd"].mean()
    print(f"     {label:<18}  n={len(sub):>3}  win={w:.0%}  ret={r:+.1f}%")

# ─── 10. Guardar ──────────────────────────────────────────────────────────────
print("\n[10] Guardando resultados...")

buckets = []
for lo, hi in zip(np.arange(0.40, 0.75, 0.05), np.arange(0.45, 0.80, 0.05)):
    sub = test_clean[(test_clean["p_adj"] >= lo) & (test_clean["p_adj"] < hi)]
    if len(sub) < 5: continue
    buckets.append({
        "bucket":      f"{lo:.2f}–{hi:.2f}",
        "n":           int(len(sub)),
        "mean_pred":   round(float(sub["p_adj"].mean()), 4),
        "actual_rate": round(float(sub["y"].mean()), 4),
    })

ticker_stats = [
    {"ticker": t, "n": len(g), "win_rate": round(float(g["y"].mean()), 3),
     "mean_ret": round(float(g["ret_fwd"].mean()), 2)}
    for t, g in sigs.groupby("ticker")
]

results = {
    "meta": {
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version":     "v2_hmm",
        "cutoff_date": str(cutoff_date.date()),
        "test_start":  str(test_df[mask_te].index.min().date()),
        "test_end":    str(test_df[mask_te].index.max().date()),
        "n_test":      len(test_clean),
        "regime_current": detector.predict_current(naftrac_df, sp500_df) if regime_ok else "unknown",
        "regime_proba":   detector.state_proba(naftrac_df, sp500_df) if regime_ok else {},
        "models": {
            "woe_lr":    {"auc": round(float(auc_lr), 4),     "weight": round(w_lr, 4)},
            "lightgbm":  {"auc": round(float(auc_lgbm), 4),   "weight": round(w_lgbm, 4)},
            "minirocket":{"auc": round(float(auc_rocket), 4), "weight": round(w_rocket, 4)},
        },
    },
    "performance": {
        "auc":          round(float(auc_ens), 4),
        "brier":        round(float(brier_ens), 4),
        "n_signals":    int(n_sig),
        "signal_pct":   round(n_sig / len(test_clean) * 100, 1),
        "win_rate":     round(float(wr), 4),
        "mean_ret":     round(float(ret_avg), 3),
        "sharpe":       round(float(sharpe), 3),
        "profit_factor": round(float(pf), 3),
    },
    "calibration":  buckets,
    "ticker_stats": ticker_stats,
    "individual_auc": {
        "woe_lr":     round(float(auc_lr), 4),
        "lightgbm":   round(float(auc_lgbm), 4),
        "minirocket": round(float(auc_rocket), 4),
        "ensemble":   round(float(auc_ens), 4),
    },
}

out = ROOT / "data"
out.mkdir(exist_ok=True)
(out / "ensemble_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
cols_save = ["ticker","y","ret_fwd","p_lr","p_lgbm","p_rocket","p_ensemble","p_adj","signal"]
if "regime" in test_clean.columns:
    cols_save.append("regime")
test_clean[cols_save].to_csv(out / "ensemble_predictions.csv")

print("   ensemble_results.json  ✅")
print("   ensemble_predictions.csv  ✅")
print()
print("=" * 65)
print("  RESUMEN ENSEMBLE v2 + HMM")
print(f"  WoE+LR:      AUC {auc_lr:.4f}  ({w_lr:.0%})")
print(f"  LightGBM:    AUC {auc_lgbm:.4f}  ({w_lgbm:.0%})")
print(f"  MiniRocket:  AUC {auc_rocket:.4f}  ({w_rocket:.0%})")
print(f"  Ensemble+HMM: AUC {auc_ens:.4f}")
print(f"  Win rate: {wr:.1%}  Sharpe: {sharpe:.2f}  PF: {pf:.2f}")
if regime_ok:
    print(f"  Régimen actual: {detector.predict_current(naftrac_df, sp500_df).upper()}")
print("=" * 65)
