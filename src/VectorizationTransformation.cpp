//===------------ VectorizationTransformation.cpp VectorizationTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the VectorizationTransformation class, which
/// contains the declartion of the Vectorization transformation
///
//===----------------------------------------------------------------------===//
#include "VectorizationTransformation.h"
#include "/home/nassimiheb/MLIR/llvm-project/build/tools/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "/home/nassimiheb/MLIR/llvm-project/mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp"
#pragma once
using namespace mlir;

Vectorization::Vectorization(mlir::linalg::LinalgOp *op,
                             mlir::MLIRContext *context)
{

  this->op = op;
  this->context = context;
}

std::string Vectorization::printTransformation()
{

  std::string result = "V( ";
  // Iterate over the elements of the vector and append them to the string
  // for (size_t i = 0; i < (tileSizes).size(); ++i) {
  //     result += std::to_string((tileSizes)[i]);

  //     if (i != (tileSizes).size() - 1) {
  //         result += ", ";
  //     }
  // }
  result += " )";

  return result;
}
void Vectorization::applyTransformation(CodeIR CodeIr)
{

}

SmallVector<Node *, 2> Vectorization::createVectorizationCandidates(Node *node,
                                                                    mlir::MLIRContext *context)
{

  // int64_t maxNumberLoops = 3;
  // std::vector<int64_t> possibleTileSizes = {0, 32, 64, 128};

  // SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;
  // SmallVector<int64_t, 4> concatenatedCombinations;

  SmallVector<SmallVector<Node *, 2>> ChildNodesList;

  SmallVector<linalg::LinalgOp, 2> LinalgOps;
  SmallVector<MLIRCodeIR *, 2> CodeIRs;

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  Operation *target = ((mlir::OwningOpRef<Operation *> *)(*CodeIr)
                           .getIr())
                          ->get();

  target->walk([&](Operation *op){
    (op)->dump();
    if (auto genricOp = dyn_cast<linalg::LinalgOp>(op)) {
        SmallVector<Node* , 2> ChildNodes;

        //for (const auto& candidate : tileCombinations){

          MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
          Node* ChildNode = new Node (ClonedCode);
          

          std::vector<Transformation*> TransList= node->getTransformationList();
          ChildNode->setTransformationList(TransList);

          Vectorization *vectorization  = 
            new Vectorization(&genricOp,
                            //candidate,
                            context);

          ChildNode->setTransformation(vectorization); 
          
          ChildNode->addTransformation(vectorization);
          
          ChildNodes.push_back(ChildNode);
        //}
        ChildNodesList.push_back(ChildNodes);
      } 
  });
  int OpIndex = 0;
  for (auto ChildNodes : ChildNodesList)
  {
    for (auto node : ChildNodes)
    {
      Operation *ClonedTarget = ((mlir::OwningOpRef<Operation *> *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                     .getIr())
                                    ->get();
      Vectorization *vectorization = (Vectorization *)node->getTransformation();

      int ClonedOpIndex = 0;
      ClonedTarget->walk([&](Operation *op)
                         {

              if (auto genricOp = dyn_cast<linalg::LinalgOp>(op)) {
                  if (ClonedOpIndex == OpIndex){
                    IRRewriter rewriter(context);
                    OpBuilder builder(context);
      
                    llvm::ArrayRef<int64_t> emptyArrayRef;

                    llvm::ArrayRef<bool> boolArrayRef;

                    mlir::linalg::vectorize(rewriter, genricOp, /*vectorSizesinputVectorSizes=*/emptyArrayRef,
                     boolArrayRef/*scalableVecDims={}*/, false);
                    //genricOp->dump();
                  }
                ClonedOpIndex++;
                } });
    }
    OpIndex++;
  }

  SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }
  std::cout << "ERROR\n";
  return ResChildNodes;
}