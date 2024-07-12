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

using namespace mlir;

Tiling::Tiling(mlir::TilingInterface *op,
               int OperationStage,
               scf::SCFTilingOptions &options,
               llvm::SmallVector<int64_t, 4> tileSizes,
               mlir::MLIRContext *context)
{

  this->op = op;
  this->OperationStage = OperationStage;
  this->options = options;
  this->context = context;
  this->tileSizes = tileSizes;
}

scf::SCFTilingOptions Tiling::getOptions()
{
  return this->options;
}
llvm::SmallVector<int64_t, 4> Tiling::getTilingSizes()
{
  return tileSizes;
}
std::string Tiling::getType()
{
  return "Tiling";
}
int Tiling::getOperationStage(){
  return this->OperationStage;
}
void Tiling::setOperationStage(int stage){
  this->OperationStage = stage;
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

  result += "I( ";
  // Iterate over the elements of the vector and append them to the string
  for (size_t i = 0; i < (this->options.interchangeVector).size(); ++i)
  {
    result += std::to_string((this->options.interchangeVector)[i]);
    if (i != (this->options.interchangeVector).size() - 1)
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
                                                      mlir::MLIRContext *context,
                                                      int CurrentStage,
                                                      std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>>   LinalgOpStages)
{

  // int64_t maxNumberLoops = 3;

  // std::vector<int64_t> possibleTileSizes = {128, 32, 64};

  SmallVector<int64_t, 4> concatenatedCombinations;

  // SmallVector<SmallVector<Node *, 2>> ChildNodesList;
  SmallVector<Node *, 2> ChildNodes;

  SmallVector<SmallVector<int64_t, 4>, 4> SelectedTileCombinations;

  SmallVector<linalg::LinalgOp, 2> LinalgOps;
  SmallVector<MLIRCodeIR *, 2> CodeIRs;

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  Operation *target = ((Operation *)(*CodeIr)
                           .getIr());
  Operation *op = LinalgOpStages["operation"+std::to_string(CurrentStage)].first;
  /*target->walk([&](Operation *op)
               {
    if ((op->getName().getStringRef()).str() != "linalg.fill" ){*/
  if (mlir::TilingInterface tileableOp = dyn_cast<mlir::TilingInterface>(op))
  {

    // SmallVector<Node* , 2> ChildNodes;
    SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;
    SmallVector<utils::IteratorType> loops = tileableOp.getLoopIteratorTypes();

    OpBuilder builder(context);
    SmallVector<Range> iterationDomain = tileableOp.getIterationDomain(builder);

    /*for (int NumberLoops = 2; NumberLoops <= iterationDomain.size(); ++NumberLoops)
    {*/
    SmallVector<SmallVector<int64_t, 4>, 4> newCombinations =
        generateTileForOpCombinations(/*NumberLoops*/ iterationDomain.size(), iterationDomain);
    tileCombinations.insert(tileCombinations.end(), newCombinations.begin(), newCombinations.end());
    //}

    std::vector<std::vector<unsigned>> values =
        generateCandidates(loops.size(), 5);
    //SelectedTileCombinations.push_back({1, 8, 8, 32, 5, 5, 3});
    std::sample(
      tileCombinations.begin(),
      tileCombinations.end(),
      std::back_inserter(SelectedTileCombinations),
      1,
      std::mt19937{std::random_device{}()});
    for (const auto &candidate : SelectedTileCombinations)
    {
      for (const auto &interchange : values)
      {
        MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
        Node *ChildNode = new Node(ClonedCode, node->getCurrentStage());

        std::vector<Transformation *> TransList = node->getTransformationList();
        ChildNode->setTransformationList(TransList);

        scf::SCFTilingOptions options;
        SmallVector<OpFoldResult> mixedSizes = getMixedSizes(candidate, context);
        options.setTileSizes(mixedSizes);
        llvm::SmallVector<int64_t, 4> targetSmallVector;

        targetSmallVector.reserve(interchange.size()); // Optional optimization if you know the size beforehand

        // Copy elements from std::vector to SmallVector
        for (const auto &element : interchange)
        {
          targetSmallVector.push_back((element));
        }

        options.setInterchange(targetSmallVector);

        Tiling *tiling =
            new Tiling(&tileableOp,
                       CurrentStage,
                       options,
                       candidate,
                       context);

        ChildNode->setTransformation(tiling);

        ChildNode->addTransformation(tiling);

        ChildNodes.push_back(ChildNode);
      }
    }
    // ChildNodesList.push_back(ChildNodes);
  }
  //}
  // });

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
  // for (auto ChildNodes : ChildNodesList)
  // {
  for (auto node : ChildNodes)
  {
    Operation *ClonedTarget = ((Operation *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                   .getIr());
    Tiling *tiling = (Tiling *)node->getTransformation();
    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOps = getLinalgOps(ClonedTarget);

    mlir::Operation *linalgOp = linalgOps["operation"+std::to_string(CurrentStage)].first;
    if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(linalgOp))
    {
      /*if ((op->getName().getStringRef()).str() != "linalg.fill")
      {*/
        IRRewriter rewriter(context);
        FailureOr<scf::SCFTilingResult> maybeTiled =
            scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, tiling->getOptions());
        // FailureOr<scf::SCFTileAndFuseResult> maybeTiled =
        // mlir::scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(rewriter,ClonedTileableOp,tiling->getOptions());
        node->setCurrentStage(node->getCurrentStage() - 1);
        if (!failed(maybeTiled))
          rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
      //}
    }
    int ClonedOpIndex = 0;
    /*ClonedTarget->walk([&](Operation *op)
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
                       });}*/

    /*if (mlir::TilingInterface ClonedTileableOp
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
      } });*/
  }
  OpIndex++;
  // }

  /*SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }

  return ResChildNodes;*/
  return ChildNodes;

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