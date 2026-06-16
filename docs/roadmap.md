# Roadmap

## Completed v2 direction

- Composable `FeatureMap` and `Solver` architecture.
- Batch ridge solve with tunable regularization.
- Online RLS with ReOS-ELM, FOS-ELM, and OS-CELM toggles.
- Real RBF feature maps.
- Learned ELM-AE feature extraction, ML-ELM, and H-OS-ELM.
- CUDA primitive layer for shared GPU operations.
- Dataset IO, preprocessing, and drift streams.
- Narrative documentation suite.

## Remaining work

| Step | Work |
|---|---|
| U11 | MkDocs Material site, Doxygen API reference, strict docs build |
| U12 | README badges, generated benchmark table, demo assets |
| U13 | GitHub Actions CI, GHCR publishing, versioned releases |

## Future ideas

- Optional Python bindings after a spec update.
- More datasets and larger benchmark snapshots.
- Self-hosted GPU CI for CUDA runtime tests.
- Model export format for demo reproducibility.
- More detailed drift-stream scenarios beyond the bundled synthetic stream.
