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



#include <iostream>
#include <chrono>
#include <ctime>    

using namespace mlir;


// function to split strings based on delimeter
std::vector<std::string> split(const std::string str,
                               const std::string regex_str) {
    std::regex regexz(regex_str);
    return {std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
            std::sregex_token_iterator()};
}

// function to generate candidate, apply transformation and evaluate
Node* generateSingleCandidate_3D(int64_t tileSize1,int64_t tileSize2, int64_t tileSize3, MLIRCodeIR* originalCode, mlir::MLIRContext* context) {
 

    EvaluationByExecution evaluator;
    // Clone the original MLIR code
    MLIRCodeIR* clonedCode = (MLIRCodeIR*)originalCode->cloneIr();

    // Create a new node with the cloned code
    Node* candidateNode = new Node(clonedCode);

    if (tileSize1 < 2 || tileSize2 < 2 || tileSize3 < 2){
      return candidateNode;
    }

    // Create a transformation with the specified tile size
    linalg::LinalgTilingOptions options;
    options.setTileSizes({tileSize1, tileSize2, tileSize3});
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);
    Tiling* tiling = new Tiling(nullptr, options, {tileSize1, tileSize2, tileSize3}, context);

    // Set the transformation for the candidate node
    candidateNode->setTransformation(tiling);
    candidateNode->addTransformation(tiling);
    
    //apply transformation
    Operation* ClonedTarget = ((mlir::OwningOpRef<Operation*>*)(*((MLIRCodeIR *)candidateNode->getTransformedCodeIr())).getIr())->get();

    ClonedTarget->walk([&](Operation *op) {
      // op->dump();
          if (linalg::LinalgOp ClonedTileableOp 
                            = dyn_cast<linalg::LinalgOp>(op)) {
              if ((op->getName().getStringRef()).str() != "linalg.fill"){
                      IRRewriter rewriter(context);
                    FailureOr<linalg::TiledLinalgOp> maybeTiled = // apply transformation
                            linalg::tileLinalgOp(rewriter, ClonedTileableOp, tiling->getOptions());
              }
            }     
        // op->dump();
    });  

    // compute the evaluation and set it 
    std::string eval = evaluator.evaluateTransformation(candidateNode);
    candidateNode->setEvaluation(eval);
    // is the transformation applied?
    return candidateNode;
}

// function to generate candidate, apply transformation and evaluate
Node* generateSingleCandidate_2D(int64_t tileSize1,int64_t tileSize2, MLIRCodeIR* originalCode, mlir::MLIRContext* context) {
    EvaluationByExecution evaluator;
    // Clone the original MLIR code
    MLIRCodeIR* clonedCode = (MLIRCodeIR*)originalCode->cloneIr();

    // Create a new node with the cloned code
    Node* candidateNode = new Node(clonedCode);

    if (tileSize1 < 2 || tileSize2 < 2){
      return candidateNode;
    }


    // Create a transformation with the specified tile size
    linalg::LinalgTilingOptions options;
    options.setTileSizes({tileSize1, tileSize2});
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);
    Tiling* tiling = new Tiling(nullptr, options, {tileSize1, tileSize2}, context);

    // Set the transformation for the candidate node
    candidateNode->setTransformation(tiling);
    candidateNode->addTransformation(tiling);
    
    //apply transformation
    Operation* ClonedTarget = ((mlir::OwningOpRef<Operation*>*)(*((MLIRCodeIR *)candidateNode->getTransformedCodeIr())).getIr())->get();

    ClonedTarget->walk([&](Operation *op) {
      // op->dump();
          if (linalg::LinalgOp ClonedTileableOp 
                            = dyn_cast<linalg::LinalgOp>(op)) {
              if ((op->getName().getStringRef()).str() != "linalg.fill"){
                      IRRewriter rewriter(context);
                    FailureOr<linalg::TiledLinalgOp> maybeTiled = // apply transformation
                            linalg::tileLinalgOp(rewriter, ClonedTileableOp, tiling->getOptions());
              }
            }     
        // op->dump();
    });  

    // compute the evaluation and set it 
    std::string eval = evaluator.evaluateTransformation(candidateNode);
    candidateNode->setEvaluation(eval);
    // is the transformation applied?
    return candidateNode;
}

int main(int argc, char **argv)
{
   if (argc < 3) {
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

  // UTILITY TIME


              
  // START OF HILL CLIMBING ALGORITHM

  std::string benchmark = split(argv[1], "/")[2];


  std::string filesource = "/Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/";

  std:: string performanceResultPath = filesource + "performanceResults.txt";
  std::ofstream performanceResults(performanceResultPath, std::ios_base::app);
  if (!performanceResults.is_open())
  {
    std::cout << "Failed to open file: " << std::endl;
  }



  int64_t desiredTileSize1 = 32;
  int64_t desiredTileSize2 = 32;
  int64_t desiredTileSize3 = 32;

  //2D
  int64_t desiredTileSize1_2D = 8;
  int64_t desiredTileSize2_2D = 8;

  EvaluationByExecution evaluator;

  Node *root = new Node(&codeIr); //rootnode
  Node * root2D = new Node(& codeIr);

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)root->getTransformedCodeIr();
  MLIRCodeIR *CodeIr2D = (MLIRCodeIR *)root2D->getTransformedCodeIr();

  // for evaluation -
  MLIRCodeIR* clonedCode = (MLIRCodeIR*)CodeIr->cloneIr();
  Node* rootNode = new Node(clonedCode);
  
  MLIRCodeIR* clonedCode2D = (MLIRCodeIR*)CodeIr2D->cloneIr();
  Node* rootNode2D = new Node(clonedCode2D);


  std::string bigRooteval = evaluator.evaluateTransformation(rootNode);
  std::string bigRooteval2D = evaluator.evaluateTransformation(rootNode2D);

  performanceResults << benchmark << "3D" <<","<<bigRooteval<<std::endl;
  performanceResults << benchmark << "2D" <<","<<bigRooteval2D<<std::endl;

  Node* firstChild = generateSingleCandidate_3D(desiredTileSize1, desiredTileSize2, desiredTileSize3, CodeIr,&context);
  root->createChild(firstChild);

  Node* firstChild2D = generateSingleCandidate_2D(desiredTileSize1_2D, desiredTileSize2_2D, CodeIr2D,&context);
  root2D->createChild(firstChild2D);

  performanceResults << benchmark << "3D" <<","<<firstChild->getEvaluation()<<std::endl;
  performanceResults << benchmark << "2D" <<","<<firstChild2D->getEvaluation()<<std::endl;

  Node* head = root;
  Node* head2D = root2D;

  int maxLoopRun = 100;
  int maxLoopRun2D = 100;

  int stepSize = 8;


// UTILITY TIME
  auto start = std::chrono::system_clock::now();
//Run Tests for 3D Tiling
  std::string bestChildEvaluation3D = "";
  while(maxLoopRun > 0) {
  
  Node* neighbor1 = generateSingleCandidate_3D(desiredTileSize1 - stepSize, desiredTileSize2, desiredTileSize3, CodeIr,&context);
  Node* neighbor2 = generateSingleCandidate_3D(desiredTileSize1 + stepSize, desiredTileSize2, desiredTileSize3, CodeIr,&context);

  Node* neighbor3 = generateSingleCandidate_3D(desiredTileSize1, desiredTileSize2 - stepSize, desiredTileSize3, CodeIr,&context);
  Node* neighbor4 = generateSingleCandidate_3D(desiredTileSize1, desiredTileSize2 + stepSize, desiredTileSize3, CodeIr,&context);

  Node* neighbor5 = generateSingleCandidate_3D(desiredTileSize1, desiredTileSize2, desiredTileSize3 - stepSize, CodeIr,&context);
  Node* neighbor6 = generateSingleCandidate_3D(desiredTileSize1, desiredTileSize2, desiredTileSize3 + stepSize, CodeIr,&context);


  firstChild->createChild(neighbor1);
  firstChild->createChild(neighbor2);
  firstChild->createChild(neighbor3);
  firstChild->createChild(neighbor4);
  firstChild->createChild(neighbor5);
  firstChild->createChild(neighbor6);

  // find the best among neighbor children generated
  SmallVector<Node *, 2> children = firstChild->getChildrenNodes();

  double bestChildTime = 0.0 ;
  Node* bestChildNeighbour;

  for (Node* child: children){
    std::string child_evaluation_str = split(child->getEvaluation(), " ")[0]; // "0.007746 GFLOPS7746167";
    double child_evaluation_time = std::atof(child_evaluation_str.c_str());

    if (child_evaluation_time > bestChildTime){
      bestChildNeighbour = child;
      bestChildTime = child_evaluation_time;
      bestChildEvaluation3D = child->getEvaluation();
    }
  }

  std::string root_evaluation_str = split(firstChild->getEvaluation(), " ")[0];
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

      firstChild = bestChildNeighbour; // use the parameters for the root 
  }
  maxLoopRun -= 1;
}

performanceResults << benchmark << "3D" <<","<<bestChildEvaluation3D<<std::endl;

// Some computation here
auto end = std::chrono::system_clock::now();

std::chrono::duration<double> elapsed_seconds = end-start;
std::time_t end_time = std::chrono::system_clock::to_time_t(end);

std::cout << "finished computation 3D " << std::ctime(&end_time)
          << "elapsed time: " << elapsed_seconds.count() << "s"
          << std::endl;

// double ExecutionTime3D = elapsed_seconds.count();

// Run Tests for 2D Tiling
  std::string bestChildEvaluation2D = "";
  auto st = std::chrono::system_clock::now();
  while(maxLoopRun2D > 0) {
  
  Node* neighbor1 = generateSingleCandidate_2D(desiredTileSize1_2D - stepSize, desiredTileSize2_2D, CodeIr2D,&context);
  Node* neighbor2 = generateSingleCandidate_2D(desiredTileSize1_2D + stepSize, desiredTileSize2_2D, CodeIr2D,&context);

  Node* neighbor3 = generateSingleCandidate_2D(desiredTileSize1_2D, desiredTileSize2_2D - stepSize, CodeIr2D,&context);
  Node* neighbor4 = generateSingleCandidate_2D(desiredTileSize1_2D, desiredTileSize2_2D + stepSize, CodeIr2D,&context);

  firstChild2D->createChild(neighbor1);
  firstChild2D->createChild(neighbor2);
  firstChild2D->createChild(neighbor3);
  firstChild2D->createChild(neighbor4);

  // find the best among neighbor children generated
  SmallVector<Node *, 2> children = firstChild2D->getChildrenNodes();

  double bestChildTime = 0.0 ;
  Node* bestChildNeighbour;

  for (Node* child: children){
    std::string child_evaluation_str = split(child->getEvaluation(), " ")[0]; // "0.007746 GFLOPS7746167";
    double child_evaluation_time = std::atof(child_evaluation_str.c_str());

    if (child_evaluation_time > bestChildTime){
      bestChildNeighbour = child;
      bestChildTime = child_evaluation_time;
      bestChildEvaluation2D = child->getEvaluation();
    }
  }

  std::string root_evaluation_str = split(firstChild2D->getEvaluation(), " ")[0];
  double root_evaluation_time = std::atof(root_evaluation_str.c_str());

  //only proceed if change leads to better execution time
  if (root_evaluation_time > bestChildTime){
    break;
  }
  else{
    
      std::string transformation = bestChildNeighbour->getTransformation()->printTransformation();   

      // grab the parameters from the transformation =  T( 24, 16, 16 )
      auto tokens = split(transformation, " "); // T and 24, 16, 16 )
      auto parameter_one = split(tokens[1], ",")[0]; //"24"
      auto parameter_two = split(tokens[2], ",")[0]; //"16"
      // auto parameter_three = split(tokens[3], ",")[0]; //"16"

      desiredTileSize1_2D = std::atol(parameter_one.c_str());
      desiredTileSize2_2D = std::atol(parameter_two.c_str());
      // desiredTileSize3 = std::atol(parameter_three.c_str());

      firstChild2D = bestChildNeighbour; // use the parameters for the root 
  }
  maxLoopRun2D -= 1;
}

performanceResults << benchmark << "2D" <<","<<bestChildEvaluation2D<<std::endl;
// Some computation here
auto end2 = std::chrono::system_clock::now();

std::chrono::duration<double> elapsed_s = end2-st;

std::time_t end_t = std::chrono::system_clock::to_time_t(end2);

std::cout << "finished computation 2D " << std::ctime(&end_t)
          << "elapsed time: " << elapsed_s.count() << "s"
          << std::endl;

// double ExecutionTime2D = elapsed_seconds.count();

root->setEvaluation(bigRooteval);
std::cout << "End of exploration for 3D!"<<std::endl;

root2D->setEvaluation(bigRooteval2D);
std::cout << "End of exploration for 2D!"<<std::endl;

std::cout<<std::endl;
std::cout<<"Best Tile Size2D: T("<<desiredTileSize1_2D <<", "<<desiredTileSize2_2D <<")"<<std::endl;

std::cout<<std::endl;
std::cout<<"Best Tile Size3D: T("<<desiredTileSize1 <<", "<<desiredTileSize2 <<", "<<desiredTileSize3 <<")"<<std::endl;

performanceResults << benchmark <<"3D," << "BestTileSize" <<","<<desiredTileSize1<<", "<<desiredTileSize2<<", "<<desiredTileSize3<<std::endl;
performanceResults << benchmark << "2D," << "BestTileSize" <<","<<desiredTileSize1_2D<<", "<<desiredTileSize2_2D<<std::endl;

std::ostringstream outputStringStream2;
head->printSchedule(outputStringStream2);
outputStringStream2 << "]\n";
outputStringStream2 <<"Best Tile Size 3D: T("<<desiredTileSize1 <<", "<<desiredTileSize2 <<", "<<desiredTileSize3 <<")}";
std::string outputString2 = outputStringStream2.str();


std::ostringstream outputStringStream3;
head2D->printSchedule(outputStringStream3);
outputStringStream3 << "]\n";
outputStringStream3 <<"Best Tile Size 2D: T(:"<<desiredTileSize1_2D <<", "<<desiredTileSize2_2D <<")}";
std::string outputString3 = outputStringStream3.str();
// std::cout<<outputString2<<std::endl;

std::string filename = "_eval.json";

std::string filepath2D =  filesource + benchmark + argv[2] + filename;
std::string filepath3D =  filesource + benchmark + argv[3] + filename;


std::ofstream outputFile2D(filepath2D);
if (!outputFile2D.is_open())
{
  std::cout << "Failed to open file: " << std::endl;
}

std::ofstream outputFile3D(filepath3D);
if (!outputFile3D.is_open())
{
  std::cout << "Failed to open file: " << std::endl;
}

outputFile2D << outputString2;
outputFile3D << outputString3;

outputFile2D.close();
outputFile3D.close();
performanceResults.close();
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


