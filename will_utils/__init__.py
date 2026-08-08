"""Shared utilities extracted from the notebooks in this repository.

Usage inside a notebook (Colab included)::

    !pip install -q -r requirements.txt  # or: pip install pandas scikit-learn seaborn
    from will_utils import describe_dataframe, evaluate_classifier, plot_confusion_matrix
"""

from will_utils.algorithms import (
    binary_search,
    bubble_sort,
    count_sort,
    linear_search,
    radix_sort,
    report_search,
)
from will_utils.data import (
    describe_dataframe,
    load_csv_from_dir,
    missing_value_summary,
    rename_columns,
    split_and_report,
)
from will_utils.evaluation import (
    coefficient_importance_frame,
    compare_models,
    evaluate_classifier,
    evaluate_regressor,
    feature_importance_frame,
)
from will_utils.plotting import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_decision_tree,
    plot_feature_importance,
    plot_training_history,
    plot_value_counts,
    show,
    styled_figure,
)
from will_utils.preprocessing import (
    encode_categorical_columns,
    encode_labels,
    scale_features,
)

__all__ = [
    "binary_search",
    "bubble_sort",
    "coefficient_importance_frame",
    "compare_models",
    "count_sort",
    "describe_dataframe",
    "encode_categorical_columns",
    "encode_labels",
    "evaluate_classifier",
    "evaluate_regressor",
    "feature_importance_frame",
    "linear_search",
    "load_csv_from_dir",
    "missing_value_summary",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_decision_tree",
    "plot_feature_importance",
    "plot_training_history",
    "plot_value_counts",
    "radix_sort",
    "rename_columns",
    "report_search",
    "scale_features",
    "show",
    "split_and_report",
    "styled_figure",
]
