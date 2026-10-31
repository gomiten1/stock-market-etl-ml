"""
Modelos de referencia y métricas para predecir volatilidad.

La volatilidad es muy persistente: la de mañana se parece mucho a la de hoy.
Por eso un R² alto no prueba nada por sí solo. Todo modelo de volatilidad
debe compararse contra:

  - Ingenuo:  la volatilidad futura es igual a la de los últimos días.
  - HAR-RV:   regresión lineal sobre la volatilidad de 1 día, 1 semana y
              1 mes (Corsi, 2009). Es el estándar académico y difícil de
              superar.

Las predicciones y la etiqueta están en log de volatilidad anualizada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src.ml.targets import HAR_COLS


def naive_forecast(X: pd.DataFrame, horizon: int) -> np.ndarray:
    """La volatilidad futura es la de la ventana pasada más parecida al horizonte."""
    col = "log_rv_5d" if horizon <= 5 else "log_rv_22d"
    return X[col].to_numpy()


class HARModel:
    """HAR-RV: log_rv_fwd = b0 + b1·log_rv_1d + b2·log_rv_5d + b3·log_rv_22d."""

    def __init__(self) -> None:
        self._lr = LinearRegression()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HARModel":
        self._lr.fit(X[HAR_COLS].to_numpy(), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._lr.predict(X[HAR_COLS].to_numpy())


def qlike(log_vol_true: np.ndarray, log_vol_pred: np.ndarray) -> float:
    """
    Pérdida QLIKE sobre varianzas: mean(σ²/h − log(σ²/h) − 1).

    Es la métrica estándar para pronósticos de volatilidad: castiga más
    subestimar el riesgo que sobreestimarlo. 0 = perfecto; menor es mejor.
    """
    var_true = np.exp(2 * log_vol_true)
    var_pred = np.exp(2 * log_vol_pred)
    ratio = var_true / var_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_naive: np.ndarray) -> dict[str, float]:
    """
    R² fuera de muestra, skill contra el ingenuo, Spearman y QLIKE.

    skill_vs_naive = 1 − MSE(modelo) / MSE(ingenuo). Positivo = mejor que
    el ingenuo; es la cifra que importa.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mse_naive = float(np.mean((y_true - y_naive) ** 2))
    var = float(np.var(y_true))
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "r2": 1 - mse / var if var > 0 else float("nan"),
        "skill_vs_naive": 1 - mse / mse_naive if mse_naive > 0 else float("nan"),
        "spearman": float(rho),
        "qlike": qlike(y_true, y_pred),
    }
