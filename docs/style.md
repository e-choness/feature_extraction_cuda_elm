# Style

## Formatting

`clang-format` is the formatting source of truth.

```bash
docker compose run --rm dev ./scripts/style_check.sh
```

The style check runs `clang-format --dry-run --Werror`, configures CMake, builds the project, and runs `clang-tidy` on C++ source files.

## Naming

| Kind | Style | Example |
|---|---|---|
| Namespace | `lower_snake_case` | `feature_elm` |
| File | `lower_snake_case` | `feature_map.hpp` |
| Type | `PascalCase` | `BatchRidgeSolver` |
| Function/method | `camelCase` | `transform()` |
| Variable | `camelCase` | `numSamples` |
| Data member | `camelCase_` | `ridgeAlpha_` |
| Constant | `kCamelCase` | `kSigmoid` |
| Macro | `ALL_CAPS` | `FEATURE_ELM_CORE_ELM_HPP_` |

## Documentation style

- Prefer relative links from the page that contains them.
- Keep code examples close to the current public API.
- Include a Mermaid diagram when a page explains dataflow.
- Document defaults and off-switches for every extension.
- Do not describe fake algorithms. If an implementation changes, update the spec and docs together.

## No unchecked CUDA errors

CUDA, cuBLAS, cuSOLVER, and kernel launches must use consistent error checks. Do not add raw CUDA calls without RAII or helper checks.
