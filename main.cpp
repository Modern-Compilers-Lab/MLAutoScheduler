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


#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "/home/nassimiheb/MLIR/llvm-project/mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp"
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


registry.insert<AffineDialect, scf::SCFDialect, 
                linalg::LinalgDialect, 
                arith::ArithDialect, 
                func::FuncDialect,
                memref::MemRefDialect,
                transform::TransformDialect,
                bufferization::BufferizationDialect,
                tensor::TensorDialect>();
context.appendDialectRegistry(registry);
context.loadDialect<SCFDialect>();

mlir::OwningOpRef<Operation*> module1 = 
  (mlir::OwningOpRef<Operation*>)codeIr.parseInputFile(inputFilename, context);

// mlir::OwningOpRef<Operation*> module = parseSourceString(transformString, &context);

(*module1)->dump();
Node* root = new Node(&codeIr);
SmallVector<Operation *> tileOps;
SmallVector<Operation *> tiledOps;
// // ############# 3nd Try
// // Transform all targets one by one.
//   //for (Operation *target : targets) {
// std::optional<ArrayAttr> mapping;
// // SmallVector<OpFoldResult> mixedTileSizes;
// // Operation *target = (module1).get();
// Operation *target = (module1).get();
// IRRewriter rewriter(&context);
// std::cout<<"tileableOp\n";
// auto tileableOp = dyn_cast<TilingInterface>(target);
// FailureOr<linalg::ForallTilingResult> TilingResult = tileToForallOpImpl(rewriter, tileableOp, 6,
//                             /*nominalTileSizes=*/std::nullopt, mapping,
//                             /*omitTileOffsetBoundsCheck=*/false);
// // ############# 3nd Try

std::optional<ArrayAttr> mapping;
SmallVector<OpFoldResult> mixedTileSizes;
IRRewriter rewriter(&context);

mlir::PassManager pm((module1).get()->getName());

//Apply any generic pass manager command line options and run the pipeline.
applyPassManagerCLOptions(pm);
mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>(); 

optPM.addPass(mlir::createLinalgGeneralizationPass());

if (!mlir::failed(pm.run(*(module1))))

std::cout<<"##################################################\n";

EvaluationByExecution evaluator;

SmallVector<Node* , 2>   list = Interchange::createInterchangeCandidates(root, &context/*(ChildNode->getTransformedCodeIr())*/);

root->setChildrenNodes(list);
std::cout  << "Size "<<list.size()<<std::endl;

double RootEvel = evaluator.evaluateTransformation(root);
 
root->setEvaluation(RootEvel);

for (auto ChildNode : list){

  SmallVector<Node* , 2>   list1 = Tiling::createTilingCandidates(ChildNode, &context/*(ChildNode->getTransformedCodeIr())*/);
 
  std::cout<<list.size();
  ChildNode->setChildrenNodes(list1);
  double evel = evaluator.evaluateTransformation(ChildNode);
 
  ChildNode->setEvaluation(evel);

  for (auto node1: list1){
    double evel = evaluator.evaluateTransformation(node1);
    node1->setEvaluation(evel); 
  }
}

std::ostringstream outputStringStream;
outputStringStream << "{ \"name\" : \"conv2d\" , \"evaluations\": [\n";

root->printSchedule(outputStringStream);
outputStringStream << "]\n}";

std::string outputString = outputStringStream.str();
std::ofstream outputFile("/home/nassimiheb/conv2d_benchmark_exhustiveEval.json");
    if (!outputFile.is_open()) {
        std::cout << "Failed to open file: " << std::endl;
      
    }
  outputFile << outputString;
outputFile.close();

std::cout <<"End of exploration!";

// /double time = evaluator.evaluateTransformation(argc, argv, registry, list[0]);
// for (int i =0 ; i <=10; i++){
// std::cout<<"########################Interchange##########################\n";
// Operation* op0 = ((mlir::OwningOpRef<Operation*>*)(*list[0]->getTransformedCodeIr()).getIr())->get();
// }

// Operation* op = ((mlir::OwningOpRef<Operation*>*)(*list[1]->getTransformedCodeIr()).getIr())->get();
// op->dump();
// SmallVector<Node* , 2>   list1 = Tiling::createTilingCandidates((MLIRCodeIR*)(list[0]->getTransformedCodeIr()), &context);
// std::cout<<list1.size();
// std::cout<<"########################Tiling##########################\n";
// Operation* opfinal = ((mlir::OwningOpRef<Operation*>*)(*list1[28]->getTransformedCodeIr()).getIr())->get();
// opfinal->dump();

// target->walk([&](Operation *op) {
//       if (auto tileableOp = dyn_cast<GenericOp>(op)) {
         // #############Tiling####################
        // // // 'op' implements the TilingInterface, you can work with it.
        // // // Perform tiling-related operations here.
        // // // ...
        // std::cout<<"##################################################\n"; 
        // std::cout<<"FOUND\n";
        // llvm::outs() << "Found operation that implements TilingInterface:\n";
        // op->print(llvm::outs());
        // llvm::outs() << "\n";
        // // if (!(op->getNumResults() != 1 || !op->getResult(0).getType().isIndex())) {
        // //     mixedTileSizes.push_back(op->getResult(0));
        // //     std::cout<<"ENtery\n";
        // //     }
        // LinalgTilingOptions options;
        // SmallVector<int64_t, 4> tileSizes;
  
        // // Fill the vector with values
        // tileSizes.push_back(32);
        // tileSizes.push_back(32);
        // tileSizes.push_back(64);
        // tileSizes.push_back(32);
        // tileSizes.push_back(128);
        // tileSizes.push_back(256);
        // options.setTileSizes(tileSizes);
        // options.setLoopType(LinalgTilingLoopType::Loops);
        // FailureOr<linalg::TiledLinalgOp> maybeTiled = linalg::tileLinalgOp(
        //   rewriter, tileableOp, options);
        // ###################################################################
        // SmallVector<Range> iterationDomain = tileableOp.getIterationDomain(rewriter);
        // size_t numLoops = iterationDomain.size();
        // ##############################INterchange#####################################
      // if (LinalgOp linalgOperation = dyn_cast<GenericOp>(op)) {
      //       int64_t numLoops = linalgOperation.getNumLoops();
      //        std::vector<unsigned> values(numLoops);
      //       for (int64_t i = 0; i < numLoops; ++i) {
      //         values[i] = i;  // Replace with your desired values or logic
      //       }  
      //       int64_t  temp = values[0];
      //       values[0] = values[1];
      //       values[1] = temp;
      //       ArrayRef<unsigned> interchangeVector(values);
      //       FailureOr<GenericOp> interOp = interchangeGenericOp(rewriter,tileableOp, interchangeVector);
      // }
      // (*module1)->dump();
      //  // ###################################################################
        
        //FailureOr<GenericOp> genOp = generalizeNamedOp(rewriter,
        //                                      tileableOp);

        //std::cout<op->getResult(0).getType();
        
        //
        // rewriter.setInsertionPoint(tileableOp);
        // std::cout<<"tilingResult\n"; 
        // FailureOr<linalg::ForallTilingResult> tilingResult;
        // std::cout<<"tileToForallOpUsingTileSizes\n"; 
        // tilingResult = linalg::tileToForallOpUsingTileSizes(
        // rewriter, tileableOp, mixedTileSizes, mapping);
        // std::cout<<"rewriter.replaceOp\n"; 
        // if (!failed(tilingResult))
        // rewriter.replaceOp(tileableOp, tilingResult->tileOp->getResults());

        // tileOps.push_back(tilingResult->tileOp);
        // tiledOps.push_back(tilingResult->tiledOp);

        
    //   }
    // });
// ###################################################################
  //}
// ############# First Try
// //Tiling* tr = new Tiling(32) ;


// (*module1)->dump();
// Tiling* tr1 = new Tiling(128) ;
// std::list<Transformation> TransformationList1;
// TransformationList1.push_back(*tr1);
//Node* ChildNode1 = new Node (TransformationList1,codeIr,&(*tr1));

//ChildNode1->applyTransformation();


// std::cout<<"INSIDE\n";
//     list[8]->applyTransformation();
//     Operation* op = ((mlir::OwningOpRef<Operation*>*)(*list[8]->getTransformedCodeIr()).getIr())->get();
//     op->dump();
//     double time = evaluator.evaluateTransformation(argc, argv, registry, list[8]);

}