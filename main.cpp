#include <iostream>
#include <cstdio>
#include <cstring>

// Include MLIR-related headers
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

// Include LLVM and other necessary headers
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

// Include MLIR passes and transformations
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Include custom headers
#include "Node.h"
#include "EvaluationByExecution.h"
#include "TilingTransformation.h"
#include "InterchangeTransformation.h"
#include "ParallelizationTransformation.h"
#include "VectorizationTransformation.h"
#include "MLIRCodeIR.h"
#include "BeamSearch.h"
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
  // Check if the correct number of command-line arguments is provided
  if (argc < 2)
  {
    std::cerr << "Usage: arguments error" << std::endl;
    return 1; // Indicate an error
  }

  // Extract the input filename and function name from command-line arguments
  llvm::StringRef inputFilename = argv[1];
  std::string inputFilenameString = argv[1];
  std::string extractedSubstring = inputFilenameString.substr(inputFilenameString.find_last_of('/') + 1);
  size_t dotIndex = extractedSubstring.find('.');
  std::string functionName = extractedSubstring.substr(0, dotIndex);

  // Create an instance of the MLIRCodeIR class
  MLIRCodeIR codeIr;
  // mlir::test::registerTestTransformDialectInterpreterPass();
  //   Register MLIR command-line options
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Create an MLIR context
  mlir::MLIRContext context;

  // Create a dialect registry and register necessary dialects
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

  // Append the dialect registry to the MLIR context
  context.appendDialectRegistry(registry);
  context.loadDialect<scf::SCFDialect>();
  context.loadDialect<vector::VectorDialect>();

  // Parse the input file and obtain an MLIR module
  mlir::OwningOpRef<Operation *> module1 =
      (mlir::OwningOpRef<Operation *>)codeIr.parseInputFile(inputFilename, context);

  // Dump the contents of the parsed module
  (*module1)->dump();

  // Create a root Node for transformations
  Node *root = new Node(&codeIr);
  EvaluationByExecution evaluator =  EvaluationByExecution(functionName+"_logs_best_exhustive.txt");

  // Evaluate the root transformation
  //std::string RootEvel = evaluator.evaluateTransformation(root);
  //root->setEvaluation(RootEvel);
  /*BeamSearch* searcher = new BeamSearch(3, &context, functionName);
  Node * res = searcher->runSearchMethod(root);*/

  // Initialize an evaluator for transformation evaluations
  // EvaluationByExecution evaluator =  EvaluationByExecution(functionName+"_logs_best.txt");

  //SmallVector<Node *, 2> ParaList = Parallelization::createParallelizationCandidates(root, &context);
  
  // Interlist.push_back(root);
  //std::cout << "size = " << ParaList.size() << std::endl;
  // Set children nodes for the root Node

  //root->setChildrenNodes(ParaList);

  // Evaluate the root transformation
  std::string RootEvel = evaluator.evaluateTransformation(root);
  root->setEvaluation(RootEvel);

  // Loop through the children nodes of the root
  /*for (auto ChildNode : ParaList)
  {
    std::string evel = evaluator.evaluateTransformation(ChildNode);
    ChildNode->setEvaluation(evel);*/
      
    SmallVector<Node *, 2> list1 = Tiling::createTilingCandidates(root, &context);
    
    /*MLIRCodeIR *ToCloneCodeIr = (MLIRCodeIR *)ChildNode->getTransformedCodeIr();
    MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)ToCloneCodeIr->cloneIr();
    Node* ClonedNode = new Node (ClonedCode);        

    std::vector<Transformation*> TransList= ChildNode->getTransformationList();
    ClonedNode->setTransformationList(TransList);
     list1.insert(list1.begin(), ClonedNode);*/

    //list1.push_back(ClonedNode);
    root->setChildrenNodes(list1);

    // Loop through the children nodes of the current child node
    for (auto node1 : list1)
    {
      
      std::string evel = evaluator.evaluateTransformation(node1);
      node1->setEvaluation(evel);

      /*MLIRCodeIR *ToCloneCodeIrInter = (MLIRCodeIR *)ChildNode->getTransformedCodeIr();
      MLIRCodeIR* ClonedCodeInter =  (MLIRCodeIR*)ToCloneCodeIrInter->cloneIr();
      Node* ClonedNodeInter = new Node (ClonedCodeInter);        

      std::vector<Transformation*> TransList= ChildNode->getTransformationList();
      ClonedNodeInter->setTransformationList(TransList);*/
    
      SmallVector<Node *, 2> list = Interchange::createInterchangeCandidates(node1, &context);
     
      node1->setChildrenNodes(list);

      // Loop through the children nodes of the current node1
      for (auto node2 : list)
      {
        std::string evel1 = evaluator.evaluateTransformation(node2);
        node2->setEvaluation(evel1);

        SmallVector<Node *, 2> list_vect = Vectorization::createVectorizationCandidates(node2, &context);
        node2->setChildrenNodes(list_vect);
        
        for (auto node3 : list_vect)
        {
          std::string evel2 = evaluator.evaluateTransformation(node3);
          node3->setEvaluation(evel2);
        }
      }
    }
  //}

  // Prepare the output JSON string
  std::ostringstream outputStringStream;
  outputStringStream << "{ \"name\" : \"" + functionName + "\" , \"evaluations\": [\n";

  // Print the schedule information to the output string
  root->printSchedule(outputStringStream);
  outputStringStream << "]\n}]}";

  // Convert the output string to JSON and write it to a file
  std::string outputString = outputStringStream.str();
  std::ofstream outputFile("./benchmark_exhustiveEval_" + functionName + ".json");
  if (!outputFile.is_open())
  {
    std::cout << "Failed to open file: " << std::endl;
  }
  outputFile << outputString;
  outputFile.close();

  // Display a message indicating the end of exploration
  std::cout << "End of exploration!" << std::endl;
}
