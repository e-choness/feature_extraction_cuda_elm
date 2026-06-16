# Glossary

| Term | Meaning |
|---|---|
| ELM | Extreme Learning Machine, a single hidden-layer network with random features and analytic output solve |
| OS-ELM | Online Sequential ELM, an ELM updated chunk by chunk |
| ReOS-ELM | Regularized OS-ELM with covariance regularization |
| FOS-ELM | Forgetting OS-ELM with a forgetting factor |
| OS-CELM | Constrained OS-ELM with class-distance constraint |
| ELM-AE | ELM autoencoder used as a learned feature map |
| ML-ELM | Multilayer ELM using stacked ELM-AE features and a batch ridge head |
| H-OS-ELM | Hierarchical OS-ELM using stacked ELM-AE features and an online RLS head |
| FeatureMap | Interface for transforming input matrices into feature matrices |
| Solver | Strategy for learning output weights from features and targets |
| Backend | CPU or GPU execution target |
| Ridge alpha | Positive regularization added to least-squares normal systems |
| Forgetting factor | RLS multiplier that down-weights stale online data |
| RBF | Radial Basis Function feature using Euclidean distance to centers |
