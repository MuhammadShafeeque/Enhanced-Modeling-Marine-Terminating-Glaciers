# Instructions for publishing to PyPI

## Prerequisites

Make sure you have the necessary tools installed:

```bash
pip install build twine
```

## Building the package

1. Clean previous builds:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

2. Build the package:
   ```bash
   python -m build
   ```

3. Check the distribution:
   ```bash
   twine check dist/*
   ```

## Publishing to TestPyPI (optional)

1. Upload to TestPyPI:
   ```bash
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

2. Install from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ oggm-marine-terminating
   ```

## Publishing to PyPI

1. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

2. Install from PyPI:
   ```bash
   pip install oggm-marine-terminating
   ```

## GitHub Release

1. Create a new release on GitHub
2. Tag the release with the version number (e.g., v0.1.0)
3. The GitHub Actions workflow will automatically build and publish to PyPI if configured correctly

## Updating the package

1. Update the version number in:
   - `setup.py`
   - `pyproject.toml`
   - `CHANGELOG.md`

2. Commit and push changes
3. Create a new release on GitHub
4. The CI/CD pipeline will handle the rest
