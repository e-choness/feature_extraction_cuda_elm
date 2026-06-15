#ifndef FEATURE_ELM_CORE_ACTIVATION_KIND_HELPERS_HPP_
#define FEATURE_ELM_CORE_ACTIVATION_KIND_HELPERS_HPP_

#include "core/feature_map.hpp"

namespace feature_elm {

inline ActivationKind activationKind(ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::kSigmoid:
      return ActivationKind::kSigmoid;
    case ActivationFunction::kTanh:
      return ActivationKind::kTanh;
    case ActivationFunction::kRelu:
      return ActivationKind::kRelu;
  }
  return ActivationKind::kSigmoid;
}

}  // namespace feature_elm

#endif  // FEATURE_ELM_CORE_ACTIVATION_KIND_HELPERS_HPP_