# Digits Dataset Source and License

## Dataset: digits_8x8.csv

This is the scikit-learn `load_digits()` 8x8 handwritten digit dataset. It contains grayscale images of handwritten digits, flattened to 64 integer pixel features with labels in the range 0-9.

- **Format:** CSV with header `label,pixel0,...,pixel63`
- **Samples:** 1,797 total
- **Features:** 64 integer pixel values per sample
- **Classes:** 10 labels, digits 0-9
- **Source:** `sklearn.datasets.data.digits.csv.gz` from scikit-learn
- **License:** BSD 3-Clause; see `LICENSE`

## Notes

The original scikit-learn dataset is derived from the UCI Optical Recognition of Handwritten Digits data set. This repository stores a compact CSV copy so tests, demos, and documentation can run without network access.
