"""
Etiquetas alternativas de predicción.

Todos los modelos del repo predicen la dirección del precio a 5 días y
ninguno supera AUC ~0.55. Este módulo construye otras preguntas con los
mismos precios, para probar si el límite está en el algoritmo o en lo que
se le pide predecir:

  - Volatilidad futura:    qué tanto se moverá la acción (no hacia dónde).
  - Rendimiento relativo:  si la acción le gana al IPC (NAFTRAC).
  - Horizonte más largo:   dirección y rendimiento relativo a 21 y 63 días.

Convenciones
------------
- Las features en la fecha t usan precios hasta el cierre de t.
- Una etiqueta en la fecha t usa precios de t+1 a t+h (rendimiento de
  cierre a cierre). Para purgar el CV, la etiqueta de t "vive" hasta t+h.
- Si no hay suficiente futuro para calcular una etiqueta (últimos días de
  la serie), el valor es NaN y nunca 0.
- Las etiquetas binarias son float (1.0 / 0.0 / NaN) para conservar NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_EPS = 1e-4


# ─── Rendimientos ────────────────────────────────────────────────────────────

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Rendimiento simple de t a t+horizon. NaN en los últimos `horizon` días."""
    return close.shift(-horizon) / close - 1.0


def relative_forward_return(close: pd.Series, bench_close: pd.Series,
                            horizon: int) -> pd.Series:
    """
    Rendimiento a `horizon` días de la acción menos el del benchmark (IPC).

    El benchmark se alinea por fecha al índice de la acción. Si al benchmark
    le falta la fecha t o la fecha t+horizon, el resultado es NaN.
    """
    bench = bench_close.reindex(close.index)
    return forward_return(close, horizon) - forward_return(bench, horizon)


def binary_label(values: pd.Series) -> pd.Series:
    """1.0 si > 0, 0.0 si <= 0, NaN si el valor es NaN."""
    return (values > 0).astype(float).where(values.notna())


# ─── Volatilidad realizada ───────────────────────────────────────────────────

def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Varianza diaria de Parkinson: ln(High/Low)^2 / (4 ln 2).

    Usa el rango del día, no solo el cierre, así que estima la volatilidad
    con menos ruido que el rendimiento al cuadrado.
    """
    hl = np.log(high / low)
    return hl ** 2 / (4.0 * np.log(2.0))


def _daily_variance(df: pd.DataFrame) -> pd.Series:
    """
    Varianza diaria: Parkinson si hay High/Low válidos, si no r^2 de cierre.

    Algunos días de Yahoo traen High == Low (sin rango); ahí se usa r^2.
    """
    r2 = np.log(df["Close"]).diff() ** 2
    if {"High", "Low"}.issubset(df.columns):
        pk = parkinson_variance(df["High"], df["Low"])
        valid = (df["High"] > df["Low"]) & pk.notna() & np.isfinite(pk)
        return pk.where(valid, r2)
    return r2


def past_realized_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Volatilidad anualizada de los últimos `window` días, hasta t inclusive."""
    var = _daily_variance(df)
    return np.sqrt(var.rolling(window).mean() * TRADING_DAYS)


def forward_realized_vol(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Volatilidad anualizada de t+1 a t+horizon. NaN en los últimos días.

    En t usa la varianza de los días t+1, ..., t+horizon.
    """
    var = _daily_variance(df)
    fwd_mean = var.rolling(horizon).mean().shift(-horizon)
    return np.sqrt(fwd_mean * TRADING_DAYS)


# ─── Features extra (causales) ───────────────────────────────────────────────

def momentum_12_1(close: pd.Series) -> pd.Series:
    """
    Momentum 12-1: rendimiento de t-252 a t-21 (sin el último mes).
    Factor estándar de la literatura (Jegadeesh y Titman, 1993).
    """
    return close.shift(21) / close.shift(TRADING_DAYS) - 1.0


HAR_COLS = ["log_rv_1d", "log_rv_5d", "log_rv_22d"]
EXTRA_FEATURE_COLS = ["mom_12_1"] + HAR_COLS


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features que este estudio agrega a FEATURE_COLS (todas causales)."""
    return pd.DataFrame({
        "mom_12_1": momentum_12_1(df["Close"]) * 100,
        "log_rv_1d": np.log(past_realized_vol(df, 1) + _EPS),
        "log_rv_5d": np.log(past_realized_vol(df, 5) + _EPS),
        "log_rv_22d": np.log(past_realized_vol(df, 22) + _EPS),
    }, index=df.index)


# ─── Construcción de todas las etiquetas ─────────────────────────────────────

def build_targets(
    df: pd.DataFrame,
    bench_close: pd.Series,
    horizons: tuple[int, ...] = (5, 21, 63),
    vol_horizons: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """
    Todas las etiquetas de un ticker, alineadas a su índice.

    Columnas:
      dir_{h}        1 si el rendimiento a h días es > 0
      rel_{h}        1 si le gana al IPC a h días
      log_rv_fwd_{v} log de la volatilidad realizada de t+1 a t+v
    """
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"dir_{h}"] = binary_label(forward_return(close, h))
        out[f"rel_{h}"] = binary_label(relative_forward_return(close, bench_close, h))
    for v in vol_horizons:
        out[f"log_rv_fwd_{v}"] = np.log(forward_realized_vol(df, v) + _EPS)
    return out


def target_horizon(target: str) -> int:
    """Días de futuro que usa una etiqueta (para purgar y embargar el CV)."""
    return int(target.rsplit("_", 1)[1])


def is_regression(target: str) -> bool:
    return target.startswith("log_rv_fwd_")
