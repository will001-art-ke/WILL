import numpy as np
import pandas as pd

from will_utils.preprocessing import (
    encode_categorical_columns,
    encode_labels,
    scale_features,
)


def test_encode_labels_roundtrip():
    encoded, encoder = encode_labels(["b", "a", "b"])
    assert list(encoded) == [1, 0, 1]
    assert list(encoder.inverse_transform(encoded)) == ["b", "a", "b"]


def test_encode_categorical_columns_defaults_to_object_columns():
    df = pd.DataFrame({"num": [1, 2], "cat": ["x", "y"]})
    encoded, encoders = encode_categorical_columns(df)
    assert list(encoders) == ["cat"]
    assert list(encoded["cat"]) == [0, 1]
    assert list(encoded["num"]) == [1, 2]
    assert list(df["cat"]) == ["x", "y"]


def test_scale_features_fits_on_train_only():
    X_train = pd.DataFrame({"a": [0.0, 10.0]})
    X_test = pd.DataFrame({"a": [5.0]})
    train_scaled, test_scaled, _ = scale_features(X_train, X_test, method="minmax")
    assert np.allclose(train_scaled.ravel(), [0.0, 1.0])
    assert np.allclose(test_scaled.ravel(), [0.5])


def test_scale_features_without_test_split():
    _, test_scaled, scaler = scale_features(pd.DataFrame({"a": [1.0, 3.0]}))
    assert test_scaled is None
    assert np.isclose(scaler.mean_[0], 2.0)


def test_scale_features_rejects_unknown_method():
    try:
        scale_features(pd.DataFrame({"a": [1.0]}), method="robust")
    except ValueError as exc:
        assert "Unknown scaler" in str(exc)
    else:
        raise AssertionError("expected ValueError")
