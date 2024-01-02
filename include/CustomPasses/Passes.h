
#ifndef MLSCEDULER_CUSTOM_PASSES_PASSES_H_
#define MLSCEDULER_CUSTOM_PASSES_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createForEachThreadLowering();


} // namespace mlir

#endif // MLSCEDULER_CUSTOM_PASSES_PASSES_H_
