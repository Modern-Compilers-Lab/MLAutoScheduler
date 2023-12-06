#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "ForEachThreadLowering.h"
using namespace mlir;
namespace {



class ForEachToParallelRewritePattern final
  : public OpRewritePattern<scf::ForallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(scf::ForallOp op, PatternRewriter& rewriter) const override {
    // We don't know how to handle loop carried values.
    // The assumption is that the lowering is running after bufferization, so
    // there should not be any loop carried tensor values.
    if (op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(
          op, "scf.parallel with outputs not supported");

    auto loc = op.getLoc();

    // 0. Create the new ParallelOp
    auto newLoop = rewriter.create<scf::ParallelOp>(
        loc,
        op.getLowerBound(rewriter),
        op.getUpperBound(rewriter),
        op.getStep(rewriter),
        [&](OpBuilder&, Location, ValueRange ivs) {
          for (auto [oldIV, newIV] : llvm::zip(op.getInductionVars(), ivs)) {
            replaceAllUsesInRegionWith(oldIV, newIV, op.getRegion());
          }
        });

    // 1. Delete old op terminator
    rewriter.eraseOp(&op.getBody()->back());

    // 2. Splice the body of the original op into the body of the new op
    auto block = newLoop.getBody();
    block->getOperations().splice(
        block->begin(), op.getBody()->getOperations());

    // 3. Erase the old operation
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

} // namespace


namespace mlir {
namespace scf {

#define GEN_PASS_DEF_FOREACHTHREADLOWERING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
#include "CustomPasses/Passes.h.inc"


class LowerForEachThread final
  : public impl::ForEachThreadLoweringBase<LowerForEachThread> {
public:
  void runOnOperation() override {
    auto ctx = &getContext();
    auto funcOp = getOperation();

    RewritePatternSet patterns(ctx);
    populateLowerForEachThreadPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

void populateLowerForEachThreadPatterns(RewritePatternSet& patterns) {
  patterns.add<ForEachToParallelRewritePattern>(patterns.getContext());
}

} // namespace scf
std::unique_ptr<Pass> createForEachThreadLowering() {
  return std::make_unique<scf::LowerForEachThread>();
}

} // namespace mlir



