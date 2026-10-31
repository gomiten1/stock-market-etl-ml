"""
WoE analysis sobre datos históricos BMV.

Flujo:
  1. Descarga 2.5 años de datos para los 19 tickers
  2. Calcula los 25 features (features.py)
  3. Define target: ret_5d > 0  (señal alcista a 5 días)
  4. Corre WoE + IV por feature
  5. Guarda resultados en data/woe_results.json

Ejecución:
    python scripts/run_woe.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features import (
    compute_features, FEATURE_COLS, add_macro_features, add_cross_section_features,
)
from src.ml.woe import compute_all_woe, iv_summary
from src.data.store import get as store_get

TICKERS: dict[str, str] = {
    "WALMEX":   "WALMEX.MX",   "FEMSAUBD": "FEMSAUBD.MX",
    "GMEXICOB": "GMEXICOB.MX", "CEMEXCPO": "CEMEXCPO.MX",
    "GFNORTEO": "GFNORTEO.MX", "BIMBOA":   "BIMBOA.MX",
    "ALSEA":    "ALSEA.MX",    "KIMBERA":  "KIMBERA.MX",
    "PINFRA":   "PINFRA.MX",   "ORBIA":    "ORBIA.MX",
    "GRUMAB":   "GRUMAB.MX",   "ASURB":    "ASURB.MX",
    "OMAB":     "OMAB.MX",     "GAPB":     "GAPB.MX",
    "GCARSOA1": "GCARSOA1.MX", "MEGACPO":  "MEGACPO.MX",
    "LABB":     "LABB.MX",     "Q":        "Q.MX",
    "VOLARA":   "VOLARA.MX",
}

HORIZON = 5
WARMUP  = 60


def build_dataset() -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=int(2.5 * 365))).strftime("%Y-%m-%d")

    macro = {
        "USDMXN":  store_get("USDMXN"),
        "NAFTRAC": store_get("NAFTRAC"),
        "WTI":     store_get("WTI"),
        "SP500":   store_get("SP500"),
    }

    # Pasada 1: features por ticker (técnicos + macro).
    # Las features cs_* comparan cada ticker contra el universo en la misma
    # fecha, así que no se pueden calcular aquí: necesitan todos los tickers.
    frames: dict[str, pd.DataFrame] = {}
    raws: dict[str, pd.DataFrame] = {}

    for ticker in TICKERS:
        try:
            raw = store_get(ticker)
            if raw is None or len(raw) < WARMUP + HORIZON + 10:
                print(f"  {ticker:<12} ... SKIP")
                continue
            raw = raw[raw.index >= start]
            df = compute_features(raw)
            df = add_macro_features(df, macro)
            frames[ticker] = df
            raws[ticker] = raw
        except Exception as e:
            print(f"  {ticker:<12} ... ERROR: {e}")

    # Pasada 2: features cross-seccionales con el universo completo.
    frames = add_cross_section_features(frames)

    rows: list[pd.DataFrame] = []
    for ticker, df in frames.items():
        print(f"  {ticker:<12} ...", end=" ", flush=True)
        try:
            raw = raws[ticker]
            entry  = raw["Open"].shift(-1)
            exit_  = raw["Close"].shift(-(HORIZON + 1))
            df["target_ret"] = (exit_ - entry) / (entry + 1e-8) * 100
            # Sin precio futuro (últimos días) no hay etiqueta: NaN, no 0.
            df["y"] = (df["target_ret"] > 0).astype(float).where(df["target_ret"].notna())

            df["ticker"] = ticker
            df = df.dropna(subset=["y"] + FEATURE_COLS[:10])
            df = df.iloc[WARMUP:]
            df["y"] = df["y"].astype(int)

            rows.append(df[["ticker", "y", "target_ret"] + FEATURE_COLS])
            print(f"{len(df):,} obs")

        except Exception as e:
            print(f"ERROR: {e}")

    if not rows:
        raise RuntimeError("Sin datos.")

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print(f"\n{'='*58}")
    print(f"  WoE ANALYSIS BMV  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Target: retorno positivo a {HORIZON} días (entry = open t+1)")
    print(f"{'='*58}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Construyendo dataset...\n")
    ds = build_dataset()

    print(f"\nTotal observaciones : {len(ds):,}")
    print(f"Tickers             : {ds['ticker'].nunique()}")
    print(f"Eventos (Y=1)       : {ds['y'].sum():,}  ({ds['y'].mean()*100:.1f}%)")
    print(f"No-eventos (Y=0)    : {(ds['y']==0).sum():,}")

    # ── WoE ───────────────────────────────────────────────────────────────────
    print("\nCalculando WoE + IV...\n")
    available = [f for f in FEATURE_COLS if f in ds.columns]
    woe_map   = compute_all_woe(ds, available, target_col="y", n_bins=8)
    summary   = iv_summary(woe_map)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"  {'Feature':<22} {'IV':>8}  {'Fuerza':<12} {'Monot.':>6}  {'Bins':>4}")
    print(f"  {'─'*58}")
    for _, row in summary.iterrows():
        mono = "si" if row["monotone"] else "no"
        bar  = "█" * min(20, int(row["iv"] * 200))
        print(f"  {row['feature']:<22} {row['iv']:>8.5f}  {row['strength']:<12} "
              f"{mono:>6}  {row['n_bins']:>4}   {bar}")

    # ── WoE por bins del mejor feature ───────────────────────────────────────
    top_feat = summary.iloc[0]["feature"]
    top_woe  = woe_map[top_feat]
    print(f"\n  Bins WoE — {top_feat}  (IV={top_woe.iv:.5f}):")
    print(f"  {'Rango':<22} {'N':>6} {'EventRate':>10} {'WoE':>8} {'IV contrib':>10}")
    print(f"  {'─'*60}")
    for b in top_woe.bins:
        bar = "█" * max(0, min(15, int(b.woe * 10)))
        print(f"  {b.bin_label:<22} {b.n:>6} {b.event_rate:>9.1%} {b.woe:>+8.3f} "
              f"{b.iv_contrib:>10.5f}  {bar}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Serializar resultados
    out: dict = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "horizon": HORIZON,
            "n_obs": int(len(ds)),
            "n_tickers": int(ds["ticker"].nunique()),
            "event_rate": round(float(ds["y"].mean()), 4),
        },
        "iv_summary": summary.to_dict(orient="records"),
        "bins_by_feature": {},
    }
    for feat, fw in woe_map.items():
        out["bins_by_feature"][feat] = [
            {
                "label":      b.bin_label,
                "lo":         b.lo,
                "hi":         b.hi,
                "n":          b.n,
                "event_rate": round(b.event_rate, 4),
                "woe":        round(b.woe, 4),
                "iv_contrib": round(b.iv_contrib, 6),
            }
            for b in fw.bins
        ]

    (data_dir / "woe_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    summary.to_csv(data_dir / "woe_iv_summary.csv", index=False)

    print(f"\n  Guardado: data/woe_results.json")
    print(f"           data/woe_iv_summary.csv\n")


if __name__ == "__main__":
    main()
