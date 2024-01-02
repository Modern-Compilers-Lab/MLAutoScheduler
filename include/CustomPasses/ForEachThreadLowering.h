
#ifndef MLSCEDULER_CUSTOM_PASSES_FOR_EACH_THREAD_LOWERING_H_
#define MLSCEDULER_CUSTOM_PASSES_FOR_EACH_THREAD_LOWERING_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
namespace mlir {
namespace func {
class FuncOp;
}
namespace scf {
void populateLowerForEachThreadPatterns(RewritePatternSet& patterns);
} // namespace scf
} // namespace mlir

#endif // MLSCEDULER_CUSTOM_PASSES_FOR_EACH_THREAD_LOWERING_H_