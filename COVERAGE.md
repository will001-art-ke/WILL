# Test coverage analysis

## Starting point

The repository contains 26 Colab notebooks and no importable Python package, no
test suite and no test configuration. Line coverage tools (`coverage.py`) cannot
be pointed at `.ipynb` files, and running a notebook top to bottom is not a test:
most cells download datasets, train models or open Matplotlib windows.

Coverage is therefore measured per callable: a function or class defined in a
notebook counts as covered when the test suite loads and exercises it.
`tools/notebook_coverage.py` computes that metric.

Before this change: **0 of 131 callables (0%)** across all notebooks.

## Which modules had the least coverage

Everything was at 0%, so the notebooks were ranked by how much untested logic
they contain and by how testable that logic is without heavy dependencies
(TensorFlow, PyTorch, Earth Engine, live Bluesky/Yahoo APIs). Tests were added
for the pure, deterministic helpers, prioritising the algorithmic notebooks
whose code is the most reusable and the most likely to be wrong silently.

## What is covered now

| Notebook | Callables under test |
| --- | --- |
| `sorts.ipynb` | `count_sort`, `countingSort_for_radix`, `radixSort`, `bubble_sort` |
| `bubble_search_algos.ipynb` | `linear_search`, `bubble_sort` |
| `RREF_&_NEWTON_RAPHSON_TECH.ipynb` | `rref`, `newton_raphson`, `newton_raphson_visual` |
| `Fixed_Point_Iteration_num_technique.ipynb` | `g`, `gx`, `gy` |
| `fourier_series_and_transform10.ipynb` | `square_wave`, `square_wave_period`, `calculate_fourier_coefficients`, `reconstruct_fourier_series` |
| `sentiment_analysis.ipynb` | `preprocess_text`, `lemmatize_tokens` |
| `real_time_topic_analysis_for_X_...ipynb` | `clean_text`, `merge_and_format_entities` |
| `simulated_weather_kenyan_ML_model.ipynb` | `assign_condition`, `visvalingam_whyatt`, `visvalingam_whyatt_simplify` |
| `stock_exchange_data_by_omoke.ipynb` | `create_sequences` |
| `web_analysis_ml,nn.ipynb` | `categorize_traffic` |
| `clustering_algrithms_on_kenyan_medical_facilities.ipynb` | `calculate_score_no_noise` |
| `mpesa_transaction_by_omoke.ipynb` | `evaluate_regression` |
| `ASSOCIATION_RULE_MINING_..._by_Omoke.ipynb` | `monitor_algorithm` |
| `last_supervised_learning_ml.ipynb` | `get_confidence_interval` |
| `GIS_analysis.ipynb` | `calculate_ndvi` |

After this change: **30 of 131 callables (23%)**, 128 passing tests and one
`xfail` documenting a real bug (see below).

## Bug found by the new tests

`rref` in `RREF_&_NEWTON_RAPHSON_TECH.ipynb` increments the pivot column and
moves to the next row when a column has no pivot, instead of retrying the same
row against the next column. For `[[0, 1, 2], [0, 2, 5]]` it returns
`[[0, 0, -0.5], [0, 1, 2.5]]`, which is not in reduced row echelon form. The
behaviour is pinned by a strict `xfail` test rather than silently accepted; the
fix belongs in the notebook.

## Still uncovered

The remaining callables are Keras/PyTorch model builders, Matplotlib and
`ipywidgets` dashboards, animation callbacks and functions that call live
services (Earth Engine, Bluesky, `yfinance`). Testing them needs either those
dependencies installed or a refactor that separates computation from I/O and
plotting; the two titanic notebooks and the CNN/MNIST notebooks fall entirely in
that category.

## Running

```bash
pip install -r requirements-dev.txt
pytest                                  # run the suite
python tools/notebook_coverage.py            # coverage table
python tools/notebook_coverage.py --uncovered  # list untested callables
```
