import pandas as pd
import pytest

from will_utils.data import (
    describe_dataframe,
    load_csv_from_dir,
    missing_value_summary,
    rename_columns,
    split_and_report,
)


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2, None, 4], "b": list("wxyz")})


def test_load_csv_from_dir_reads_single_file(tmp_path, frame):
    frame.to_csv(tmp_path / "data.csv", index=False)
    assert list(load_csv_from_dir(str(tmp_path)).columns) == ["a", "b"]


def test_load_csv_from_dir_picks_first_of_many(tmp_path, capsys):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"b": [1]}).to_csv(tmp_path / "b.csv", index=False)
    assert list(load_csv_from_dir(str(tmp_path)).columns) == ["a"]
    assert "Multiple CSV files found" in capsys.readouterr().out


def test_load_csv_from_dir_without_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_from_dir(str(tmp_path))


def test_missing_value_summary(frame):
    summary = missing_value_summary(frame)
    assert summary.loc["a", "missing_count"] == 1
    assert summary.loc["a", "missing_pct"] == 25.0
    assert summary.index[0] == "a"


def test_describe_dataframe_prints_sections(frame, capsys):
    describe_dataframe(frame)
    out = capsys.readouterr().out
    for section in ("--- Head ---", "--- DataFrame Info ---", "--- Missing Values ---"):
        assert section in out


def test_split_and_report_stratifies():
    X = pd.DataFrame({"f": range(20)})
    y = pd.Series([0, 1] * 10)
    X_train, X_test, y_train, y_test = split_and_report(X, y, stratify=True)
    assert len(X_train) == 16 and len(X_test) == 4
    assert set(y_test) == {0, 1}
    assert len(y_train) == 16


def test_rename_columns_reports_unknown(frame, capsys):
    renamed = rename_columns(frame, {"a": "alpha", "zz": "nope"})
    assert "alpha" in renamed.columns
    assert "'zz'" in capsys.readouterr().out
