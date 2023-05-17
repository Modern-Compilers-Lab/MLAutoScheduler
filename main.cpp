#include <iostream>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "Node.h"
#include "EvaluationByExecution.h"
#include "TilingTransformation.h"
#include "MLIRCodeIR.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <optional>

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "/home/nassimiheb/MLIR/llvm-project/mlir/examples/toy/Ch8/MyInterpter.cpp"
#include "/home/nassimiheb/MLIR/llvm-project/mlir/test/lib/Dialect/Transform/TestTransformDialectInterpreter.cpp"
#include "/home/nassimiheb/MLIR/llvm-project/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterPassBase.cpp"


#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
using namespace mlir;

int main(int argc, char** argv) {


llvm::StringRef inputFilename = "/home/nassimiheb/conv2d_benchmark.mlir";
// llvm::StringRef transformString = "transform.sequence failures(propagate) {^bb0(%arg1: !pdl.operation): \n %0 = transform.structured.match ops{[\"linalg.conv_2d_nhwc_hwcf\"]} in %arg1 : (!pdl.operation) -> !pdl.operation \n %1, %loops:4 = transform.structured.tile %0 [32, 32, 32, 32] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)}";
MLIRCodeIR codeIr;

mlir::test::registerTestTransformDialectInterpreterPass();

// Register any command line options.
mlir::registerAsmPrinterCLOptions();
mlir::registerMLIRContextCLOptions();
mlir::registerPassManagerCLOptions();

mlir::MLIRContext context;
//mlir::registerAllPasses();

DialectRegistry registry;
registerAllDialects(registry);


mlir::registerAllToLLVMIRTranslations(registry);
//mlir::registerAllToLLVMIRTranslations(registry);

registry.insert<AffineDialect, linalg::LinalgDialect, arith::ArithDialect, func::FuncDialect,memref::MemRefDialect,transform::TransformDialect>();
context.appendDialectRegistry(registry);

mlir::OwningOpRef<Operation*> module1 = (mlir::OwningOpRef<Operation*>)codeIr.parseInputFile(inputFilename, context);

// mlir::OwningOpRef<Operation*> module = parseSourceString(transformString, &context);

Node* node = new Node(&codeIr);

Tiling* tr = new Tiling(32) ;
std::list<Transformation> TransformationList;
TransformationList.push_back(*tr);
Node* ChildNode = new Node (TransformationList,&codeIr,&(*tr));

ChildNode->applyTransformation();
std::list<Transformation*>  list = tr->createCandidates((ChildNode->getTransformedCodeIr()));


// Tiling* tr1 = new Tiling(128) ;
// std::list<Transformation> TransformationList1;
// TransformationList1.push_back(*tr1);
//Node* ChildNode1 = new Node (TransformationList1,codeIr,&(*tr1));

//ChildNode1->applyTransformation();

EvaluationByExecution evaluator;
//double time = evaluator.evaluateTransformation(argc, argv, registry, node);
}


