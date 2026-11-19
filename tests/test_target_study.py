import sys
from types import ModuleType

try:
    import fcntl  # noqa: F401
except ImportError:
    sys.modules["fcntl"] = ModuleType("fcntl")

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_target_study import QuantileClipper


def test_clipper_fit_only_on_train_and_clips_test():
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(2000, 3))
    X_test = np.array([
        [500.0, -500.0, 1000.0],
        [-999.0, 999.0, -800.0],
    ])

    clipper = QuantileClipper().fit(X_train)

    expected_lo = np.nanpercentile(X_train, 0.1, axis=0)
    expected_hi = np.nanpercentile(X_train, 99.9, axis=0)
    np.testing.assert_allclose(clipper.lo_, expected_lo)
    np.testing.assert_allclose(clipper.hi_, expected_hi)

    X_test_clipped = clipper.transform(X_test)
    assert np.all(X_test_clipped >= clipper.lo_)
    assert np.all(X_test_clipped <= clipper.hi_)


def test_clipper_leaves_in_range_values_unchanged():
    rng = np.random.default_rng(7)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(2000, 2))
    clipper = QuantileClipper().fit(X_train)

    X_inside = np.array([
        [0.0, 0.5],
        [-0.5, 1.0],
        [0.2, -0.3],
    ])
    X_transformed = clipper.transform(X_inside)
    np.testing.assert_allclose(X_transformed, X_inside)


def test_clipper_preserves_nans_and_finite_bounds():
    rng = np.random.default_rng(123)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(1000, 2))
    X_train[0, 0] = np.nan
    X_train[10, 1] = np.nan

    clipper = QuantileClipper().fit(X_train)
    assert np.all(np.isfinite(clipper.lo_))
    assert np.all(np.isfinite(clipper.hi_))

    X_test = np.array([
        [np.nan, 0.0],
        [100.0, np.nan],
    ])
    X_out = clipper.transform(X_test)
    assert np.isnan(X_out[0, 0])
    assert np.isnan(X_out[1, 1])
    assert np.isfinite(X_out[0, 1])
    assert np.isfinite(X_out[1, 0])


def test_ridge_pipeline_with_vs_without_clipper():
    rng = np.random.default_rng(99)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(2000, 3))
    y_train = 2.0 * X_train[:, 0] - 1.5 * X_train[:, 1] + rng.normal(0.0, 0.1, size=2000)

    # Outlier extremo de 1000 desviaciones estándar en el conjunto de prueba
    X_test_outlier = np.array([[1000.0, 0.0, 0.0]])

    pipe_sin_clipper = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))
    pipe_con_clipper = make_pipeline(QuantileClipper(), SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))

    pipe_sin_clipper.fit(X_train, y_train)
    pipe_con_clipper.fit(X_train, y_train)

    pred_sin = abs(pipe_sin_clipper.predict(X_test_outlier)[0])
    pred_con = abs(pipe_con_clipper.predict(X_test_outlier)[0])

    assert pred_sin > 500.0
    assert pred_con < 20.0