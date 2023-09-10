//===------------ ParallelizationTransformation.cpp ParallelizationTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the ParallelizationTransformation class, which
/// contains the declartion of the Parallelization transformation
///
//===----------------------------------------------------------------------===//
#include "ParallelizationTransformation.h"
#pragma once
using namespace mlir;
void generateCombinations(const SmallVector<int64_t, 4> &tileSizes,
                          int64_t maxNumberLoops,
                          int64_t currentLoop,
                          SmallVector<int64_t, 4> &currentCombination,
                          std::vector<SmallVector<int64_t, 4>> &combinations);

SmallVector<SmallVector<int64_t, 4>, 4>
generateTileForOpCombinations(int64_t maxNumberLoops,
                              const std::vector<int64_t> &possibleTileSizes);

Parallelization::Parallelization(mlir::TilingInterface *op,
                                 llvm::SmallVector<int64_t, 4> tileSizes,
                                 mlir::MLIRContext *context)
{

  this->op = op;
  this->context = context;
  this->tileSizes = tileSizes;
}

llvm::SmallVector<int64_t, 4> Parallelization::getTileSizes()
{
  return this->tileSizes;
}
std::string Parallelization::printTransformation()
{

  std::string result = "T( ";
  // Iterate over the elements of the vector and append them to the string
  for (size_t i = 0; i < (tileSizes).size(); ++i)
  {
    result += std::to_string((tileSizes)[i]);

    if (i != (tileSizes).size() - 1)
    {
      result += ", ";
    }
  }
  result += " )";

  return result;
}
void Parallelization::applyTransformation(CodeIR CodeIr)
{
}

SmallVector<Node *, 2> Parallelization::createParallelizationCandidates(Node *node,
                                                                        mlir::MLIRContext *context)
{

  int64_t maxNumberLoops = 3;
  std::vector<int64_t> possibleTileSizes = {32, 64, 128};

  SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;
  SmallVector<int64_t, 4> concatenatedCombinations;

  SmallVector<SmallVector<Node *, 2>> ChildNodesList;

  for (int NumberLoops = 1; NumberLoops <= maxNumberLoops; ++NumberLoops)
  {
    SmallVector<SmallVector<int64_t, 4>, 4> newCombinations =
        generateTileForOpCombinations(NumberLoops, possibleTileSizes);
    tileCombinations.insert(tileCombinations.end(), newCombinations.begin(), newCombinations.end());
  }

  // SmallVector<SmallVector<int64_t, 4>, 4> SelectedTileCombinations;
  // std::sample(
  //     tileCombinations.begin(),
  //     tileCombinations.end(),
  //     std::back_inserter(SelectedTileCombinations),
  //     tileCombinations.size(),
  //     std::mt19937{std::random_device{}()}
  // );

  SmallVector<linalg::LinalgOp, 2> LinalgOps;
  SmallVector<MLIRCodeIR *, 2> CodeIRs;

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  Operation *target = ((mlir::OwningOpRef<Operation *> *)(*CodeIr)
                           .getIr())
                          ->get();

  target->walk([&](Operation *op)
               {

      if (mlir::TilingInterface tileableOp = dyn_cast<mlir::TilingInterface>(op)) {
          std::cout<<"FOUND\n";
          SmallVector<Node* , 2> ChildNodes;

          for (const auto& candidate : tileCombinations){

            MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
            Node* ChildNode = new Node (ClonedCode);
            
            std::vector<Transformation*> TransList= node->getTransformationList();
            ChildNode->setTransformationList(TransList);

            Parallelization *parallelization  = 
              new Parallelization(&tileableOp,
                              candidate,
                              context);

            ChildNode->setTransformation(parallelization); 
            
            ChildNode->addTransformation(parallelization);
            
            ChildNodes.push_back(ChildNode);
          }
          ChildNodesList.push_back(ChildNodes);

        } });
  int OpIndex = 0;
  for (auto ChildNodes : ChildNodesList)
  {
    for (auto node : ChildNodes)
    {
      Operation *ClonedTarget = ((mlir::OwningOpRef<Operation *> *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                     .getIr())
                                    ->get();
      Parallelization *parallelization = (Parallelization *)node->getTransformation();

      int ClonedOpIndex = 0;
      ClonedTarget->walk([&](Operation *op){
        if (mlir::TilingInterface ClonedTileableOp 
                          = dyn_cast<mlir::TilingInterface>(op)) {
            if (ClonedOpIndex == OpIndex){
              IRRewriter rewriter(context);
              OpBuilder builder(context);
            
              std::optional<ArrayAttr> mapping;
              SmallVector<OpFoldResult, 4> opFoldResults;
              for (int64_t value : parallelization->getTileSizes()) {
                    opFoldResults.push_back(builder.getIndexAttr(value));
              }
              rewriter.setInsertionPoint(ClonedTileableOp);
              ArrayRef<OpFoldResult> tileSizes =  llvm::makeArrayRef(opFoldResults);
              FailureOr<linalg::ForallTilingResult> tilingResult = 
                          linalg::tileToForallOpUsingTileSizes(rewriter,ClonedTileableOp,tileSizes,mapping);

              if (!failed(tilingResult))
                  rewriter.replaceOp(ClonedTileableOp, tilingResult->tileOp->getResults());      
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

  return ResChildNodes;
}

void generateForOpCombinations(const SmallVector<int64_t, 4> &tileSizes,
                               int64_t maxNumberLoops,
                               int64_t currentLoop,
                               SmallVector<int64_t, 4> &currentCombination,
                               std::vector<SmallVector<int64_t, 4>> &combinations)
{
  if (currentLoop >= maxNumberLoops)
  {
    combinations.push_back(currentCombination);
    return;
  }

  for (int64_t tileSize : tileSizes)
  {
    currentCombination[currentLoop] = tileSize;
    generateForOpCombinations(tileSizes,
                              maxNumberLoops,
                              currentLoop + 1,
                              currentCombination,
                              combinations);
  }
}

SmallVector<SmallVector<int64_t, 4>, 4>
generateTileForOpCombinations(int64_t maxNumberLoops,
                              const std::vector<int64_t> &possibleTileSizes)
{
  SmallVector<int64_t, 4> tileSizes;
  std::copy(possibleTileSizes.begin(),
            possibleTileSizes.end(),
            std::back_inserter(tileSizes));

  SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<SmallVector<int64_t, 4>> combinations;

  generateForOpCombinations(tileSizes,
                            maxNumberLoops,
                            0,
                            currentCombination,
                            combinations);

  return SmallVector<SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                 combinations.end());
}
