 
#ifndef MLIR_DIALECT_SCF_PASSES_CUSTOM
#define MLIR_DIALECT_SCF_PASSES_CUSTOM

include "mlir/Pass/PassBase.td"

def ForEachThreadLowering : Pass<"scf-foreach-thread-lowering", "func::FuncOp"> {
  let summary = "Lower scf.foreach_thread operations to scf.parallel loops";

    let dependentDialects = [
   "scf::SCFDialect",
    "arith::ArithDialect"
 ];
}

#endif 