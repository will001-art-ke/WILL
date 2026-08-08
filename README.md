# WILL

Colab notebooks covering numerical methods, classical algorithms and machine
learning experiments.

## Tests

The notebook helpers are unit tested by extracting their definitions with
`tests/notebook_loader.py`, so the notebooks themselves stay unchanged.

```bash
pip install -r requirements-dev.txt
pytest
python tools/notebook_coverage.py   # per-notebook coverage of notebook callables
```

See [COVERAGE.md](COVERAGE.md) for the coverage analysis and what is still
untested.
