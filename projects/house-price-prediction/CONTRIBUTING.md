# Contributing

1. Create a branch from `main`.
2. Install `requirements-dev.txt` in a clean environment.
3. Keep preprocessing inside the scikit-learn pipeline so validation data cannot influence fitted transformations.
4. Run `python -m src.train` and `python -m pytest` before opening a pull request.
5. Update `MODEL_CARD.md` and `models/metrics.json` when the data, features, or estimator changes.

Do not commit private, licensed, or personally identifiable data. Document the source and license of any replacement dataset.
