//===------------ TilingTransformation.cpp TilingTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the TilingTransformation class, which
/// contains the declartion of the tiling transformation
///
//===----------------------------------------------------------------------===//
#include "TilingTransformation.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/Value.h"

#pragma once
using namespace mlir;
void generateCombinations(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                          int64_t maxNumberLoops,
                          int64_t currentLoop,
                          SmallVector<int64_t, 4> &currentCombination,
                          std::vector<SmallVector<int64_t, 4>> &combinations);

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForOpCombinations(int64_t maxNumberLoops,
                              const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &possibleTileSizes,
                              const llvm::SmallVector<int64_t> &upperBounds);
Tiling::Tiling(mlir::TilingInterface *op,
               scf::SCFTilingOptions &options,
               llvm::SmallVector<int64_t, 4> tileSizes,
               mlir::MLIRContext *context)
{

  this->op = op;
  this->options = options;
  this->context = context;
  this->tileSizes = tileSizes;
}

scf::SCFTilingOptions Tiling::getOptions()
{
  return this->options;
}
std::string Tiling::printTransformation()
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
void Tiling::applyTransformation(CodeIR CodeIr)
{

  IRRewriter rewriter(this->context);
  // linalg::LinalgOp oper = (this->op);
  //  std::cout<<"BLOCKINSIDE BUG\n";

  // OpBuilder::InsertionGuard g(rewriter);

  // auto blk = ((this->op).getNumLoops());

  // rewriter.setInsertionPoint((this->op));
  // //   SmallVector<OpFoldResult> tileSizeVector =
  // //   getAsOpFoldResult((this->options).tileSizeComputationFunction(rewriter, (this->op)));
  // std::cout<<"BLOCKINSIDE\n";

  // FailureOr<linalg::TiledLinalgOp> maybeTiled = linalg::tileLinalgOp(
  //     rewriter, oper, this->options);

  // SmallVector<AffineForOp, 6> tiledNest;
  // if (failed(tilePerfectlyNested(this->band, this->tileSize, &tiledNest))) {
  //         std::cout<<"FAILED\n";
  // }

  // mlir::OwningOpRef<Operation*>* module =
  //     (mlir::OwningOpRef<Operation*>*)CodeIr.getIr();
  // mlir::PassManager pm((*module).get()->getName());

  // // // Apply any generic pass manager command line options and run the pipeline.
  // applyPassManagerCLOptions(pm);
  // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

  // // pm.addPass(mlir::createInlinerPass());
  // optPM.addPass(mlir::createConvertLinalgToAffineLoopsPass ());
  // //optPM.addPass(mlir::createLoopTilingPass ());
  // // for (Pass &pass : optPM.getPasses()) {
  // //     pass.initializeOptions("tile-size="+std::to_string(tileSize));
  // //     //pass.printAsTextualPipeline(output);
  // // }
  // // optPM.addPass(mlir::createConvertLinalgToLoopsPass());
  // pm.addPass(mlir::createInlinerPass());
  // optPM.addPass(mlir::createLinalgBufferizePass());
  // pm.addPass(mlir::createConvertLinalgToLLVMPass ());
  // optPM.addPass(createSCFForLoopCanonicalizationPass());
  // optPM.addPass(mlir::createConvertLinalgToLoopsPass());
  // optPM.addPass(mlir::createConvertLinalgToAffineLoopsPass ());
  // optPM.addPass(mlir::createConvertSCFToCFPass());
  // optPM.addPass(memref::createExpandStridedMetadataPass ());
  // optPM.addPass(mlir::createLowerAffinePass ());

  // optPM.addPass( mlir::createArithToLLVMConversionPass());
  // optPM.addPass(mlir::createConvertSCFToCFPass());

  // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  // pm.addPass(mlir::createConvertFuncToLLVMPass());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass ());

  // if (!mlir::failed(pm.run(*(*module))));

  // (**module)->dump();

  // CodeIr.setIr(module);
}

SmallVector<Node *, 2> Tiling::createTilingCandidates(Node *node,
                                                      mlir::MLIRContext *context)
{

  int64_t maxNumberLoops = 3;

  // std::vector<int64_t> possibleTileSizes = {128, 32, 64};

  SmallVector<int64_t, 4> concatenatedCombinations;

  SmallVector<SmallVector<Node *, 2>> ChildNodesList;

  SmallVector<SmallVector<int64_t, 4>, 4> SelectedTileCombinations;

  SmallVector<linalg::LinalgOp, 2> LinalgOps;
  SmallVector<MLIRCodeIR *, 2> CodeIRs;

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  Operation *target = ((mlir::OwningOpRef<Operation *> *)(*CodeIr)
                           .getIr())
                          ->get();
  SmallVector<Node *, 2> ChildNodes;

  // tileCombinations = generateTileCombinations(loops.size(),
  //                                    possibleTileSizes);
  target->walk([&](Operation *op)
               {
    if ((op->getName().getStringRef()).str() != "linalg.fill" ){
             // std::cout << "op = "<<(op->getName().getStringRef()).str()<<std::endl;
      if (mlir::TilingInterface tileableOp = dyn_cast<mlir::TilingInterface>(op)) {

          SmallVector<Node* , 2> ChildNodes;
          SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;
          SmallVector<utils::IteratorType> loops = tileableOp.getLoopIteratorTypes();
          std::cout << loops.size() <<std::endl;
          /*if(loops.size()==3){
           llvm::SmallVector<int64_t, 4> elementsToInsert = {4, 2, 15};

          // Insert the elements into tileCombinations
          tileCombinations.push_back(elementsToInsert);}*/

          OpBuilder builder(context);
          SmallVector<Range> iterationDomain = tileableOp.getIterationDomain(builder);

          llvm::SmallVector<int64_t> upperBounds;
          for (const auto &range : llvm::enumerate(iterationDomain))
          {
            llvm::SmallVector<Value> dynamicVec;
            llvm::SmallVector<int64_t> staticVec;
            if (auto val = getConstantIntValue(range.value().size)){
              dispatchIndexOpFoldResult(range.value().size,
                                      dynamicVec,
                                      staticVec);
              upperBounds.append(staticVec.begin(), staticVec.end());
            }else{
              upperBounds.push_back(-1);
            }
           
          }
          SmallVector<SmallVector<int64_t, 4>, 4> possibleTileSizes;
          for (int64_t value : upperBounds) {
            llvm::SmallVector<int64_t, 4> dividers;
            if (value == -1){
               dividers.push_back(1);
            }
            for (int64_t i = 2; i <= value; ++i) {
                if (value % i == 0) {
                    dividers.push_back(i);
                }
            }
            possibleTileSizes.push_back(dividers);
          }
          for (int NumberLoops = 2; NumberLoops <= upperBounds.size(); ++NumberLoops)
          {
            SmallVector<SmallVector<int64_t, 4>, 4> newCombinations =
                generateTileForOpCombinations(NumberLoops, possibleTileSizes, upperBounds);
            tileCombinations.insert(tileCombinations.end(), newCombinations.begin(), newCombinations.end());
          }
          /*tileCombinations.erase(
              std::remove_if(
                  tileCombinations.begin(),
                  tileCombinations.end(),
                  [](const llvm::SmallVector<int64_t, 4> &innerVec) {
                      return innerVec.size() == 1;
                  }
              ),
              tileCombinations.end()
          );*/
           /*std::cout << "Upper BOUNDS\n";
          for (const int64_t value : upperBounds) {
            std::cout << value << " ";
            std::cout << std::endl;
          }
          std::cout << "Deviders\n";
          for (const auto &outerVector : possibleTileSizes) {
            for (const int64_t value : outerVector) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
          }
          std::cout << "Tiling Sizes\n";
          for (const auto &outerVector : tileCombinations) {
            for (const int64_t value : outerVector) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
          }
          std::cout << "End Tiling Sizes\n";*/
          
         //tileCombinations.erase(tileCombinations.begin());
         
          /*tileCombinations = generateTileCombinations(loops.size(),
                                      possibleTileSizes);
          std::sample(
            tileCombinations.begin(),
            tileCombinations.end(),
            std::back_inserter(SelectedTileCombinations),
            tileCombinations.size()/6,
            std::mt19937{std::random_device{}()});*/
          for (const auto& candidate : tileCombinations){

 
            MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
            Node* ChildNode = new Node (ClonedCode);
            

            std::vector<Transformation*> TransList= node->getTransformationList();
            ChildNode->setTransformationList(TransList);

            scf::SCFTilingOptions options;
            //scf::SCFTilingOptions scfoptions;
            //std::cout <<"SIZE ====" << loops.size();
            options.setTileSizes(candidate);
            //options.setLoopType(linalg::LinalgTilingLoopType::Loops);
            //options.setTilingOptions(scfoptions);
            Tiling *tiling  = 
              new Tiling(&tileableOp,
                              options, 
                              candidate,
                              context);

            ChildNode->setTransformation(tiling); 
            
            ChildNode->addTransformation(tiling);
            
            ChildNodes.push_back(ChildNode);
          }
          ChildNodesList.push_back(ChildNodes);

        } } });

  /*for (const auto& candidate : tileCombinations){


    MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
    Node* ChildNode = new Node (ClonedCode);


    std::vector<Transformation*> TransList= node->getTransformationList();
    ChildNode->setTransformationList(TransList);

    scf::SCFTileAndFuseOptions options;
    scf::SCFTilingOptions scfoptions;
    //std::cout <<"SIZE ====" << loops.size();
    scfoptions.setTileSizes(candidate);
    //options.setLoopType(linalg::LinalgTilingLoopType::Loops);
    options.setTilingOptions(scfoptions);
    Tiling *tiling  =
      new Tiling(&tileableOp1,
                      options,
                      candidate,
                      context);

    ChildNode->setTransformation(tiling);

    ChildNode->addTransformation(tiling);

    ChildNodes.push_back(ChildNode);
    }
    ChildNodesList.push_back(ChildNodes);*/

  int OpIndex = 0;
  for (auto ChildNodes : ChildNodesList)
  {
    for (auto node : ChildNodes)
    {
      Operation *ClonedTarget = ((mlir::OwningOpRef<Operation *> *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                     .getIr())
                                    ->get();
      Tiling *tiling = (Tiling *)node->getTransformation();

      int ClonedOpIndex = 0;
      ClonedTarget->walk([&](Operation *op)
                         {
                          
                       /*if (mlir::TilingInterface ClonedTileableOp 
                                =dyn_cast<mlir::TilingInterface>(op)) { 
                        IRRewriter rewriter(context);
                        Diagnostic diag(op->getLoc(),DiagnosticSeverity::Remark);
                        ClonedTarget->walk([&](Operation *op1)
                         {  
                          if ((op1->getName().getStringRef()).str() == "scf.forall") {
                                  std::cout<<"for.all\n";
                                SmallVector<Operation *>res =   tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(rewriter,diag,op,op1);

                                 }
                         });
                        
                       
                        }*/
              if (mlir::TilingInterface ClonedTileableOp 
                                =dyn_cast<mlir::TilingInterface>(op)) {
                  if ((op->getName().getStringRef()).str() != "linalg.fill" ){
                      IRRewriter rewriter(context);
                      FailureOr<scf::SCFTilingResult> maybeTiled = 
                              scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, tiling->getOptions());
                      //FailureOr<scf::SCFTileAndFuseResult> maybeTiled =
                         //mlir::scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(rewriter,ClonedTileableOp,tiling->getOptions());
                      if (!failed(maybeTiled))
                              rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults()); 
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

  // // // Print the generated tile combinations
  // // for (const auto& combination : SelectedTileCombinations) {
  // //     for (int64_t tileSize : combination) {
  // //         std::cout << tileSize << " ";
  // //     }
  // //     std::cout << std::endl;
  // // }

  // // MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
  // // Node* ChildNode = new Node (ClonedCode);
  // // ChildNodes.push_back(ChildNode);
  // Operation* target =
  //     ((mlir::OwningOpRef<Operation*>*)(*CodeIr).getIr())->get();
  // int counter = 0;
  // target->walk([&](Operation *op) {

  //     if (linalg::LinalgOp tileableOp = dyn_cast<linalg::LinalgOp>(op)) {
  //         // #############Tiling####################
  //         // // // 'op' implements the TilingInterface, you can work with it.
  //         // // // Perform tiling-related operations here.
  //         // // // ...
  //         // std::cout<<"##################################################\n";
  //         // std::cout<<"FOUND\n";
  //         // llvm::outs() << "Found operation that implements TilingInterface:\n";
  //         // op->print(llvm::outs());
  //         // llvm::outs() << "\n";

  //         for (SmallVector<int64_t, 4> TileCombination: SelectedTileCombinations){
  //             MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
  //             Node* ChildNode = new Node (ClonedCode);
  //             std::vector<Transformation*> TransList= node->getTransformationList();
  //             ChildNode->setTransformationList(TransList);
  //             scf::SCFTilingOptions options;

  //             options.setTileSizes(TileCombination);
  //             options.setLoopType(linalg::LinalgTilingLoopType::Loops);

  //             Operation* ClonedOp =
  //                 ((mlir::OwningOpRef<Operation*>*)(*ClonedCode).getIr())->get();
  //             int ClonedCounter = 0;

  //             ClonedOp->walk([&](Operation *op) {
  //                 if (linalg::LinalgOp ClonedTileableOp = dyn_cast<linalg::LinalgOp>(op)) {
  //                     if (ClonedCounter == counter){
  //                         //std::cout<<"BLOCK\n";
  //                         //auto blk = (ClonedTileableOp.getNumLoops());
  //                         //std::cout<<blk;
  //                         IRRewriter rewriter(context);
  //                         FailureOr<linalg::TiledLinalgOp> maybeTiled =
  //                             linalg::tileLinalgOp(rewriter, op, options);

  //                         Tiling *tiling = new Tiling(&ClonedTileableOp,
  //                                                     options,
  //                                                     TileCombination,
  //                                                     context);

  //                         ChildNode->addTransformation(tiling);

  //                         ChildNode->setTransformation(tiling);
  //                         op->erase();
  //                         ChildNodes.push_back(ChildNode);
  //                     }
  //                 ClonedCounter++;
  //                 }
  //             });
  //         }
  //         counter++;
  //     }

  // });
  // return ChildNodes;
}
// Function to generate tiling sizes that are multiples of the upperBounds.
void generateForOpCombinations(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                               int64_t maxNumberLoops,
                               int64_t currentLoop,
                               llvm::SmallVector<int64_t, 4> &currentCombination,
                               std::vector<llvm::SmallVector<int64_t, 4>> &combinations,
                               const llvm::SmallVector<int64_t> &upperBounds)
{
  if (currentLoop >= maxNumberLoops)
  {
    combinations.push_back(currentCombination);
    return;
  }
  llvm::SmallVector<int64_t, 4> currentTileSizes = tileSizes[currentLoop];

  for (int64_t tileSize : currentTileSizes)
  {
    // Check if the current tileSize is a multiple of the corresponding upperBound.
    if (upperBounds[currentLoop] % tileSize == 0)
    {

      currentCombination[currentLoop] = tileSize;
      generateForOpCombinations(tileSizes,
                                maxNumberLoops,
                                currentLoop + 1,
                                currentCombination,
                                combinations,
                                upperBounds);
    }
  }
}

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForOpCombinations(int64_t maxNumberLoops,
                              const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &possibleTileSizes,
                              const llvm::SmallVector<int64_t> &upperBounds)
{

  llvm::SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<llvm::SmallVector<int64_t, 4>> combinations;

  generateForOpCombinations(possibleTileSizes,
                            maxNumberLoops,
                            0,
                            currentCombination,
                            combinations,
                            upperBounds);

  return llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                             combinations.end());
}
/*void generateCombinations(const SmallVector<int64_t, 4> &tileSizes,
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
    generateCombinations(tileSizes,
                         maxNumberLoops,
                         currentLoop + 1,
                         currentCombination,
                         combinations);
  }
}

SmallVector<SmallVector<int64_t, 4>, 4>
generateTileCombinations(int64_t maxNumberLoops,
                         const std::vector<int64_t> &possibleTileSizes)
{
  SmallVector<int64_t, 4> tileSizes;
  std::copy(possibleTileSizes.begin(),
            possibleTileSizes.end(),
            std::back_inserter(tileSizes));

  SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<SmallVector<int64_t, 4>> combinations;

  generateCombinations(tileSizes,
                       maxNumberLoops,
                       0,
                       currentCombination,
                       combinations);

  return SmallVector<SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                 combinations.end());
}*/

// SmallVector<Node* , 2> Tiling::createTilingCandidates(MLIRCodeIR *CodeIr){

//     std::vector<Transformation*>  TransformationList;
//     // Bands of loops to tile.
//     std::vector<SmallVector<AffineForOp, 6>> bands;

//     std::vector<int> tileSizeoptions{ 4, 16, 32, 64 };

//     Operation* op = ((mlir::OwningOpRef<Operation*>*)(*CodeIr).getIr())->get();

//     SmallVector<AffineForOp , 2> forOps;
//     SmallVector<MLIRCodeIR* , 2> CodeIRs;
//     op->walk([&](AffineForOp forOp) {
//         MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
//         CodeIRs.push_back(ClonedCode);
//         forOps.push_back(forOp);

//         });
//     SmallVector<AffineForOp , 2> forOpsToExplore;
//     SmallVector<Node* , 2> ChildNodes;
//     for (int i = 0; i < CodeIRs.size(); ++i) {
//         auto& CloneCodeIr = CodeIRs[i];
//         for (int size: tileSizeoptions){
//             Node* ChildNode = new Node (CloneCodeIr);
//             ChildNodes.push_back(ChildNode);
//         }

//         Operation* op = ((mlir::OwningOpRef<Operation*>*)(*CloneCodeIr).getIr())->get();

//         SmallVector<AffineForOp , 2> forOps;
//         op->walk([&](AffineForOp forOp) { forOps.push_back(forOp); });
//         forOpsToExplore.push_back(forOps[i]);
//     }

//     for (auto forOp : forOpsToExplore) {
//         SmallVector<AffineForOp, 6> nest;

//         // Block &body = forOp.getRegion().front();
//         // body.dump();
//         // Get the maximal perfect nest.
//         getPerfectlyNestedLoops(nest, forOp);
//         bands.push_back(nest);
//     }

//     for (int i = 0; i < bands.size(); ++i) {
//         auto& band = bands[i];
//         // if (!checkTilingLegality(band)) {
//         // band.front().emitRemark("tiling code is illegal due to dependences");
//         // continue;
//         // }

//         // Set up tile sizes; fill missing tile sizes at the end with default tile
//         // size or tileSize if one was provided.

//         for (int j = 0; j < tileSizeoptions.size(); ++j){
//             SmallVector<unsigned, 6> tileSizes;
//             getTileSizes(band, &tileSizes);
//             std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
//             if (!fp) {
//                 std::cout<<"footprint \n";
//                 // Fill with default tile sizes if footprint is unknown.
//                 std::fill(tileSizes.begin(), tileSizes.end(),32);
//             }
//             Tiling *tiling = new Tiling(&tileSizes, &band);
//             ChildNodes[i+j]->setTransformation(tiling);
//         }

//         //TransformationList.push_back(tiling);

//         // for (const auto& size : tileSizes) {
//         //     std::cout << size << " ";
//         // }
//         // std::cout << std::endl;
//     }
//     return ChildNodes;
// }
// void getTileSizes(ArrayRef<AffineForOp> band,
//                               SmallVectorImpl<unsigned> *tileSizes) {
//   if (band.empty())
//     return;

//   // Use tileSizes and fill them with default tile size if it's short.
//   if (!tileSizes->empty()) {
//     std::cout<<"inside \n";
//     tileSizes->assign(tileSizes->begin(), tileSizes->end());
//     tileSizes->resize(band.size(), 4);
//     return;
//   }
//   tileSizes->resize(band.size());
// }