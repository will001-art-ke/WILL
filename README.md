# WILL

Data science and machine learning notebooks, plus `will_utils`, a small package
holding the helpers that the notebooks used to copy and paste between each other.

## will_utils

```bash
pip install -r requirements.txt
```

```python
from will_utils import (
    describe_dataframe, load_csv_from_dir, split_and_report,
    evaluate_classifier, evaluate_regressor, feature_importance_frame,
    plot_confusion_matrix, plot_feature_importance, plot_training_history,
)

df = load_csv_from_dir("/content/data")
describe_dataframe(df)

X_train, X_test, y_train, y_test = split_and_report(X, y, stratify=True)
evaluate_classifier(y_test, model.predict(X_test), target_names=encoder.classes_)
plot_confusion_matrix(y_test, model.predict(X_test), labels=encoder.classes_)
```

| Module | Contents |
| --- | --- |
| `will_utils.data` | `load_csv_from_dir`, `describe_dataframe`, `missing_value_summary`, `split_and_report`, `rename_columns` |
| `will_utils.preprocessing` | `encode_labels`, `encode_categorical_columns`, `scale_features` |
| `will_utils.evaluation` | `evaluate_classifier`, `evaluate_regressor`, `feature_importance_frame`, `coefficient_importance_frame`, `compare_models` |
| `will_utils.plotting` | `styled_figure`, `plot_confusion_matrix`, `plot_feature_importance`, `plot_training_history`, `plot_decision_tree`, `plot_value_counts`, `plot_correlation_heatmap` |
| `will_utils.algorithms` | `bubble_sort`, `count_sort`, `radix_sort`, `linear_search`, `binary_search`, `report_search` |

In Colab, clone the repo (or `%cd` into it) so that `will_utils` is importable:

```python
!git clone https://github.com/will001-art-ke/WILL.git
%cd WILL
!pip install -q -r requirements.txt
```

## Tests

```bash
pip install pytest
python -m pytest
```
