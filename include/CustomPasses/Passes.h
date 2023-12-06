
#ifndef MLIR_DIALECT_CUSTOM_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_CUSTOM_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createForEachThreadLowering();


} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_
