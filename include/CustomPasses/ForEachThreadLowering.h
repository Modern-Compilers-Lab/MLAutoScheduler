#pragma once
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