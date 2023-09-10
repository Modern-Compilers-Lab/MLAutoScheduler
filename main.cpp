#include <iostream>
#include <cstdio>
#include <cstring>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "InterchangeTransformation.h"
#include "ParallelizationTransformation.h"
#include "VectorizationTransformation.h"

#include "MLIRCodeIR.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <optional>

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

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

int main(int argc, char **argv)
{
   if (argc < 2) {
        std::cerr << "Usage: arguments error" << std::endl;
        return 1; // Indicate an error
  }
  llvm::StringRef inputFilename = argv[1];
  MLIRCodeIR codeIr;

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;
  // mlir::registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);

  mlir::registerAllToLLVMIRTranslations(registry);

  registry.insert<affine::AffineDialect, scf::SCFDialect,
                  linalg::LinalgDialect,
                  arith::ArithDialect,
                  func::FuncDialect,
                  memref::MemRefDialect,
                  transform::TransformDialect,
                  bufferization::BufferizationDialect,
                  tensor::TensorDialect,
                  vector::VectorDialect,
                  shape::ShapeDialect>();
  context.appendDialectRegistry(registry);
  context.loadDialect<scf::SCFDialect>();
  context.loadDialect<vector::VectorDialect>();

  mlir::OwningOpRef<Operation *> module1 =
      (mlir::OwningOpRef<Operation *>)codeIr.parseInputFile(inputFilename, context);

  // mlir::OwningOpRef<Operation*> module = parseSourceString(transformString, &context);

  (*module1)->dump();
  Node *root = new Node(&codeIr);
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  std::optional<ArrayAttr> mapping;
  SmallVector<OpFoldResult> mixedTileSizes;
  IRRewriter rewriter(&context);

  mlir::PassManager pm((module1).get()->getName());

  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

   optPM.addPass(mlir::createLinalgGeneralizationPass());

  if (!mlir::failed(pm.run(*(module1))))
    // module1->dump();
  std::cout << "##################################################\n";

  EvaluationByExecution evaluator;
  SmallVector<Node *, 2> Interlist = Interchange::createInterchangeCandidates(root, &context /*(ChildNode->getTransformedCodeIr())*/);
  // SmallVector<Node* , 2>   vectlist = Parallelization::createParallelizationCandidates(root, &context /*(ChildNode->getTransformedCodeIr())*/);

  root->setChildrenNodes(Interlist);
  std::cout << "Size " << Interlist.size() << std::endl;

  std::string RootEvel = evaluator.evaluateTransformation(root);

  root->setEvaluation(RootEvel);

  for (auto ChildNode : Interlist)
  {
      SmallVector<Node* , 2>   list1 = Tiling::createTilingCandidates(ChildNode, &context/*(ChildNode->getTransformedCodeIr())*/);
    // SmallVector<Node *, 2> list = Parallelization::createVectorizationCandidates(ChildNode, &context /*(ChildNode->getTransformedCodeIr())*/);

    // std::cout<<list1.size()<<std::endl;
    ChildNode->setChildrenNodes(list1);
    std::string evel = evaluator.evaluateTransformation(ChildNode);

    ChildNode->setEvaluation(evel);

    for (auto node1 : list1)
    {
      std::string evel = evaluator.evaluateTransformation(node1);
      node1->setEvaluation(evel);
    }
  }

  std::ostringstream outputStringStream;
  outputStringStream << "{ \"name\" : \"conv2d\" , \"evaluations\": [\n";

  root->printSchedule(outputStringStream);
  outputStringStream << "]\n}]}";

  std::string outputString = outputStringStream.str();
  std::ofstream outputFile("/home/nassimiheb/conv2d_benchmark_exhustiveEval.json");
  if (!outputFile.is_open())
  {
    std::cout << "Failed to open file: " << std::endl;
  }
  outputFile << outputString;
  outputFile.close();

  std::cout << "End of exploration!";

  
}