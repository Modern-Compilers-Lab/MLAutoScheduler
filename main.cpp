#include <iostream>
#include <cstdio>
#include <cstring>
#include <string.h>
#include <regex>

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


// function to split strings based on delimeter
std::vector<std::string> split(const std::string str,
                               const std::string regex_str) {
    std::regex regexz(regex_str);
    return {std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
            std::sregex_token_iterator()};
}

// function to generate candidate, apply transformation and evaluate
Node* generateSingleCandidate(int64_t tileSize1,int64_t tileSize2, int64_t tileSize3, MLIRCodeIR* originalCode, mlir::MLIRContext* context) {
    EvaluationByExecution evaluator;
    // Clone the original MLIR code
    MLIRCodeIR* clonedCode = (MLIRCodeIR*)originalCode->cloneIr();

    // Create a new node with the cloned code
    Node* candidateNode = new Node(clonedCode);

    // Create a transformation with the specified tile size
    linalg::LinalgTilingOptions options;
    options.setTileSizes({tileSize1, tileSize2, tileSize3});
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);
    Tiling* tiling = new Tiling(nullptr, options, {tileSize1, tileSize2, tileSize3}, context);

    // Set the transformation for the candidate node
    candidateNode->setTransformation(tiling);
    candidateNode->addTransformation(tiling);
    
    int OpIndex = 0;
    //apply transformation
    Operation* ClonedTarget = ((mlir::OwningOpRef<Operation*>*)(*((MLIRCodeIR *)candidateNode->getTransformedCodeIr())).getIr())->get();

    int ClonedOpIndex = 0;
    ClonedTarget->walk([&](Operation *op) {
      // op->dump();
          if (linalg::LinalgOp ClonedTileableOp 
                            = dyn_cast<linalg::LinalgOp>(op)) {
              if (ClonedOpIndex == OpIndex){
                      IRRewriter rewriter(context);
                    FailureOr<linalg::TiledLinalgOp> maybeTiled = // apply transformation
                            linalg::tileLinalgOp(rewriter, ClonedTileableOp, tiling->getOptions());
              }
            ClonedOpIndex++;
            }     

        // op->dump();
    });  
    OpIndex++;

    // compute the evaluation and set it 
    std::string eval = evaluator.evaluateTransformation(candidateNode);
    candidateNode->setEvaluation(eval);
    // is the transformation applied?
    return candidateNode;
}

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

  // (*module1)->dump();


  // START OF HILL CLIMBING ALGORITHM
  int64_t desiredTileSize1 = 32;
  int64_t desiredTileSize2 = 32;
  int64_t desiredTileSize3 = 32;


  Node *root = new Node(&codeIr); //rootnode
  MLIRCodeIR *CodeIr = (MLIRCodeIR *)root->getTransformedCodeIr();
  root = generateSingleCandidate(desiredTileSize1,desiredTileSize2, desiredTileSize3, CodeIr,&context);

  Node* head = root;
  int maxLoopRun = 20;
  int stepSize = 8;

  while(maxLoopRun > 0) {
  
  Node* neighbor1 = generateSingleCandidate(desiredTileSize1 - stepSize, desiredTileSize2, desiredTileSize3, CodeIr,&context);
  Node* neighbor2 = generateSingleCandidate(desiredTileSize1 + stepSize, desiredTileSize2, desiredTileSize3, CodeIr,&context);

  Node* neighbor3 = generateSingleCandidate(desiredTileSize1, desiredTileSize2 - stepSize, desiredTileSize3, CodeIr,&context);
  Node* neighbor4 = generateSingleCandidate(desiredTileSize1, desiredTileSize2 + stepSize, desiredTileSize3, CodeIr,&context);

  Node* neighbor5 = generateSingleCandidate(desiredTileSize1, desiredTileSize2, desiredTileSize3 - stepSize, CodeIr,&context);
  Node* neighbor6 = generateSingleCandidate(desiredTileSize1, desiredTileSize2, desiredTileSize3 + stepSize, CodeIr,&context);


  root->createChild(neighbor1);
  root->createChild(neighbor2);
  root->createChild(neighbor3);
  root->createChild(neighbor4);
  root->createChild(neighbor5);
  root->createChild(neighbor6);

  // find the best among neighbor children generated
  SmallVector<Node *, 2> children = root->getChildrenNodes();

  double bestChildTime = 0.0 ;
  Node* bestChildNeighbour;

  for (Node* child: children){
    std::string child_evaluation_str = split(child->getEvaluation(), " ")[0]; // "0.007746 GFLOPS7746167";
    double child_evaluation_time = std::atof(child_evaluation_str.c_str());

    if (child_evaluation_time > bestChildTime){
      bestChildNeighbour = child;
      bestChildTime = child_evaluation_time;
    }
  }

  std::string root_evaluation_str = split(root->getEvaluation(), " ")[0];
  double root_evaluation_time = std::atof(root_evaluation_str.c_str());

  // only proceed if change leads to better execution time
  if (root_evaluation_time > bestChildTime){
    break;
  }
  else{
    
      std::string transformation = bestChildNeighbour->getTransformation()->printTransformation();   

      // grab the parameters from the transformation =  T( 24, 16, 16 )
      auto tokens = split(transformation, " "); // T and 24, 16, 16 )
      auto parameter_one = split(tokens[1], ",")[0]; //"24"
      auto parameter_two = split(tokens[2], ",")[0]; //"16"
      auto parameter_three = split(tokens[3], ",")[0]; //"16"

      desiredTileSize1 = std::atol(parameter_one.c_str());
      desiredTileSize2 = std::atol(parameter_two.c_str());
      desiredTileSize3 = std::atol(parameter_three.c_str());

      root = bestChildNeighbour; // use the parameters for the root 
  }
  maxLoopRun -= 1;
}


std::cout << "End of exploration!"<<std::endl;

std::ostringstream outputStringStream2;
head->printSchedule(outputStringStream2);
std::string outputString2 = outputStringStream2.str();

std::cout<<outputString2<<std::endl;



std::cout<<std::endl;
std::cout<<"Best Tile Size: T("<<desiredTileSize1 <<", "<<desiredTileSize2 <<", "<<desiredTileSize3 <<")"<<std::endl;



std::ofstream outputFile("/Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/matmul_benchmark_eval.json");
if (!outputFile.is_open())
{
  std::cout << "Failed to open file: " << std::endl;
}
outputFile << outputString2;
outputFile.close();

return 0;
};
  
















  // // Now you can use candidateNode for further evaluation or processing

  // // Clean up resources (assuming ownership of the objects)
  // delete candidateNode;
  // delete originalCode;


  // SmallVector<Operation *> tileOps;
  // SmallVector<Operation *> tiledOps;

  // std::optional<ArrayAttr> mapping;
  // SmallVector<OpFoldResult> mixedTileSizes;
  // IRRewriter rewriter(&context);

  // mlir::PassManager pm((module1).get()->getName());

  // // Apply any generic pass manager command line options and run the pipeline.
  // applyPassManagerCLOptions(pm);
  // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

  //  optPM.addPass(mlir::createLinalgGeneralizationPass());

  // if (!mlir::failed(pm.run(*(module1))))
  //   // module1->dump();
  // std::cout << "##################################################\n";

  // EvaluationByExecution evaluator;
  // SmallVector<Node *, 2> Interlist = Interchange::createInterchangeCandidates(root, &context /*(ChildNode->getTransformedCodeIr())*/);
  // // SmallVector<Node* , 2>   vectlist = Parallelization::createParallelizationCandidates(root, &context /*(ChildNode->getTransformedCodeIr())*/);

  // root->setChildrenNodes(Interlist);
  // std::cout << "Size " << Interlist.size() << std::endl;

  // std::string RootEvel = evaluator.evaluateTransformation(root); // evaluator

  // root->setEvaluation(RootEvel);

  // for (auto ChildNode : Interlist)
  // {
  //     SmallVector<Node* , 2>   list1 = Tiling::createTilingCandidates(ChildNode, &context/*(ChildNode->getTransformedCodeIr())*/);
  //   // SmallVector<Node *, 2> list = Parallelization::createVectorizationCandidates(ChildNode, &context /*(ChildNode->getTransformedCodeIr())*/);

  //   // std::cout<<list1.size()<<std::endl;
  //   ChildNode->setChildrenNodes(list1);
  //   std::string evel = evaluator.evaluateTransformation(ChildNode);

  //   ChildNode->setEvaluation(evel);

  //   for (auto node1 : list1)
  //   {
  //     std::string evel = evaluator.evaluateTransformation(node1);
  //     node1->setEvaluation(evel);
  //   }
  // }

  // std::ostringstream outputStringStream;
  // outputStringStream << "{ \"name\" : \"conv2d\" , \"evaluations\": [\n";

  // root->printSchedule(outputStringStream);
  // outputStringStream << "]\n}]}";

  // std::string outputString = outputStringStream.str();

  ///Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/main.cpp


