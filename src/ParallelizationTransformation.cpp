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

#include "/scratch/ia2280/LLVM/llvm-project/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp"
#include "/scratch/ia2280/LLVM/llvm-project/mlir/test/lib/Dialect/Linalg/TestLinalgFusionTransforms.cpp"

using namespace mlir;

/*void generateForAllOpCombinations(const SmallVector<int64_t, 4> &tileSizes,
                                  int64_t maxNumberLoops,
                                  int64_t currentLoop,
                                  SmallVector<int64_t, 4> &currentCombination,
                                  std::vector<SmallVector<int64_t, 4>> &combinations);
SmallVector<SmallVector<int64_t, 4>, 4>
generateTileForAllOpCombinations(int64_t maxNumberLoops,
                                 const std::vector<int64_t> &possibleTileSizes);*/
static LogicalResult FuseIntoContainingOperation(mlir::Operation *Target, std::string producerTag, std::string consumerTag)
{

  std::string transformDialectString = "module attributes {transform.with_named_sequence} { \n transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})  { \n  %0 = transform.structured.match attributes{\"" + producerTag + "\"} in %variant_op : (!transform.any_op) -> !transform.any_op \n %1 = transform.structured.match attributes{\"" + consumerTag + "\"} in %variant_op : (!transform.any_op) -> !transform.any_op \n transform.structured.fuse_into_containing_op %0 into %1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op) \n transform.yield}}";
  mlir::transform::TransformOptions options1;
  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile = parseSourceString<mlir::ModuleOp>(transformDialectString, Target->getContext());
  llvm::StringRef entryPoint = "__transform_main";
  mlir::Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(Target, *moduleFromFile, entryPoint);

  return transform::applyTransformNamedSequence(
      Target, transformEntryPoint, *moduleFromFile,
      options1.enableExpensiveChecks(false));
}
static LogicalResult fuseLinalgOpsGreedily(Operation *f)
{
  OpBuilder b(f);
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<LinalgOp, 8> linalgOps;
  std::cout << "HERE\n";

  f->walk([&](LinalgOp op)
          {
            // TODO: support multi-results.
            if (op->getNumResults() <= 1)
            {
              linalgOps.push_back(op);
            } });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  for (LinalgOp linalgOp : llvm::reverse(linalgOps))
  {

    for (OpOperand &opOperand : linalgOp->getOpOperands())
    {

      if (isa<MemRefType>(opOperand.get().getType()))
      {
        std::cout << "TYPE\n";
        continue;
      }

      if (isa<RankedTensorType>(opOperand.get().getType()))
      {
        // Tile and Fuse tensor input.

        if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())
        {
          std::cout << "NOT INPUT\n";
          continue;
        }

        auto info = fuseProducerOfTensor(b, opOperand);
        if (failed(info))
          continue;

        auto *originalOp = info->originalProducer.getOperation();
        auto *originalOpInLinalgOpsVector =
            std::find(linalgOps.begin(), linalgOps.end(), originalOp);
        *originalOpInLinalgOpsVector = info->fusedProducer;
        // Don't mark for erasure in the tensor case, let DCE handle this.
        changed = true;
      }
    }
  }
  // The `fuseProducerOfBuffer` function performs structural checks and in
  // particular that no covering read or write exist between the consumer and
  // the producer. As a consequence, the only fusions that may occur preserve
  // subsequent dependences and are guaranteed by construction to produce the
  // whole view. We may thus erase the producer once it is fused.
  for (auto *e : eraseSet)
    e->erase();

  return changed ? success() : failure();
}
DiagnosedSilenceableFailure FuseOps(Operation *f, SmallVector<mlir::Operation *, 2> producers, std::string tag, int CurrentStage)
{
  // SmallVector<Operation *> fusedOps;
  // OpBuilder b(containingOp);
  mlir::Operation *producerOp = producers[CurrentStage];
  std::string producerTag = tag + "_producer_" + std::to_string(CurrentStage);

  TagOperation(producerOp, producerTag);
  FuseIntoContainingOperation(f, producerTag, tag);

  CurrentStage++;
  mlir::PassManager pm((f)->getName());

  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

  if (!mlir::failed(pm.run((f))))
    int ClonedOpIndex = 0;
  SmallVector<mlir::Operation *, 2> NewProducers;
  f->walk([&](mlir::Operation *op)
          {
                 if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
                  {
                    NewProducers.push_back(op);
                  } });
  std::reverse(NewProducers.begin(), NewProducers.end());

  if (NewProducers.size() > 0 && CurrentStage < NewProducers.size())
  {
    FuseOps(f, NewProducers, tag, CurrentStage);
  }
  // auto producerOps = state.getPayloadOps(getProducerOp());
  // auto containingOps = state.getPayloadOps(getContainingOp());

  /*if (!llvm::hasSingleElement(containingOps)) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << llvm::range_size(containingOps) << ")";
  }*/

  /*// If nothing to fuse, propagate success.
  if (std::empty(producerOps)) {
    results.set(cast<OpResult>(getFusedOp()), SmallVector<mlir::Operation *>{});
    results.set(cast<OpResult>(getNewContainingOp()), {containingOp});
    return DiagnosedSilenceableFailure::success();
  }*/

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  /*SetVector<Operation *> remainingProducers(producerOps.begin(),
                                            producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation *> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isAncestor(op);
          });
      // TODO: When resolving the TODO below (no duplicate ops), take an op
      // that has no use among the remaining producers. This is a topological
      // sorting.
      if (numUsesInContainingOp > 0) {
        if (numUsesInContainingOp == 1)
          remainingProducers.erase(remainingProducers.begin() + it.index());
        return producerOp;
      }
    }
    return failure();
  };*/
  /*SmallVector<LinalgOp, 8> linalgOps;
  target->walk([&](LinalgOp producerOp) {
    // TODO: support multi-results.
    if (producerOp->getNumResults() <= 1){
linalgOps.push_back(producerOp);

    }

  });

  bool changed = false;
  for (LinalgOp linalgOp : linalgOps)
  {

    for (OpOperand &opOperand : linalgOp->getOpOperands()) {

      if (isa<MemRefType>(opOperand.get().getType())){
          continue;
      }

      if (isa<RankedTensorType>(opOperand.get().getType())) {
        // Tile and Fuse tensor input.

        if (opOperand.getOperandNumber() < linalgOp.getNumDpsInputs())
        {
           continue;
        }
   std::cout << "OPERATIOn\n";
    linalgOp.dump();
       /* Diagnostic diag(linalgOp->getLoc(),DiagnosticSeverity::Remark);
        diag << "could not fuse " << *linalgOp << " into " << *containingOp;
        auto [tiledOps, newContainingOp] =
        tileAndFuseFirstExtractUse(rewriter, diag, linalgOp, containingOp);

        if (!tiledOps.empty()) {
          fusedOps.append(tiledOps);
          if (newContainingOp) {
            rewriter.eraseOp(containingOp);
            containingOp = newContainingOp;
          }
        }else{
          std::cout << "TRYING TO FUSE\n";
            SmallVector<Operation *> tiledContainingOpOperand =
            tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, linalgOp, containingOp);
            std::cout << "FUSED\n";
            if (!tiledContainingOpOperand.empty()) {
              fusedOps.append(tiledContainingOpOperand);
            }else{
              /*Operation *cloned =
                cloneAndFuseFirstUse(rewriter, diag, producerOp, containingOp);
                 cloned->dump();
                if (cloned) {

                  fusedOps.push_back(cloned);
                  //continue;
                }*/
  //}
  //}

  /* auto info = fuseProducerOfTensor(b, opOperand);
   if (failed(info))
     continue;

   auto *originalOp = info->originalProducer.getOperation();
   auto *originalOpInLinalgOpsVector =
       std::find(linalgOps.begin(), linalgOps.end(), originalOp);
   *originalOpInLinalgOpsVector = info->fusedProducer;
   // Don't mark for erasure in the tensor case, let DCE handle this.*/
  /*changed = true;
}
}
}*/
  /* target->walk([&](Operation *producerOp)
                {
     // TEMP: Check if the operation is a "linalg.fill" operation
     //if ((producerOp->getName().getStringRef()).str() == "linalg.fill")
     //{
         Diagnostic diag(producerOp->getLoc(),DiagnosticSeverity::Remark);
         diag << "could not fuse " << *producerOp << " into " << *containingOp;
         auto [tiledOps, newContainingOp] =
         tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);

         if (!tiledOps.empty()) {
           fusedOps.append(tiledOps);
           if (newContainingOp) {
             rewriter.eraseOp(containingOp);
             containingOp = newContainingOp;
           }
         }else{
             SmallVector<Operation *> tiledContainingOpOperand =
             tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
             rewriter, diag, producerOp, containingOp);
             if (!tiledContainingOpOperand.empty()) {
               fusedOps.append(tiledContainingOpOperand);
             }else{

            }
         }

     //}
     });*/
  return DiagnosedSilenceableFailure::success();
}

Parallelization::Parallelization(mlir::TilingInterface *op,
                                 int OperationStage,
                                 llvm::SmallVector<int64_t, 4> tileSizes,
                                 mlir::MLIRContext *context)
{
  this->op = op;
  this->OperationStage = OperationStage;
  this->context = context;
  this->tileSizes = tileSizes;
}

int Parallelization::getOperationStage()
{
  return this->OperationStage;
}
void Parallelization::setOperationStage(int stage)
{
  this->OperationStage = stage;
}

llvm::SmallVector<int64_t, 4> Parallelization::getTileSizes()
{
  return this->tileSizes;
}

std::string Parallelization::getType()
{
  return "Parallelization";
}
std::string Parallelization::printTransformation()
{
  std::string result = "TP( ";
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
                                                                        mlir::MLIRContext *context,
                                                                        int CurrentStage,
                                                                        SmallVector<mlir::linalg::LinalgOp, 4> LinalgOpStages)
{
  // Set the maximum number of loops for parallelization (commented out)
  // int64_t maxNumberLoops = 4;

  // Define a vector of possible tile sizes (commented out)
  // std::vector<int64_t> possibleTileSizes = {32, 64, 128};

  // Initialize vectors to store tile combinations
  SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;

  // Initialize a list to store child nodes (commented out now)
  // SmallVector<SmallVector<Node *, 2>> ChildNodesList;
  SmallVector<Node *, 2> ChildNodes;
  // Extract the transformed code IR from the input node
  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  // Get the top-level operation
  Operation *target = ((Operation *)(*CodeIr)
                           .getIr());
  Operation *op = LinalgOpStages[CurrentStage];
  // Walk through operations in the target
  /*target->walk([&](Operation *op)
               {
    // TEMP: Check if the operation is not a "linalg.fill" operation
    if ((op->getName().getStringRef()).str() != "linalg.fill")
    {*/
  // std::cout<<(op->getName().getIdentifier().str())<<std::endl;
  //  Check if the operation supports tiling
  if (mlir::TilingInterface tileableOp = dyn_cast<mlir::TilingInterface>(op))
  {
    // op->getLoc()->dump();
    // SmallVector<Node *, 2> ChildNodes;
    OpBuilder builder(context);
    SmallVector<Range> iterationDomain = tileableOp.getIterationDomain(builder);

    llvm::SmallVector<int64_t> upperBounds;
    for (const auto &range : llvm::enumerate(iterationDomain))
    {
      llvm::SmallVector<Value> dynamicVec;
      llvm::SmallVector<int64_t> staticVec;
      dispatchIndexOpFoldResult(range.value().size,
                                dynamicVec,
                                staticVec);
      upperBounds.append(staticVec.begin(), staticVec.end());
    }
    SmallVector<SmallVector<int64_t, 4>, 4> possibleTileSizes;

    for (int64_t value : upperBounds)
    {
      llvm::SmallVector<int64_t, 4> dividers;
      for (int64_t i = 2; i < std::min(value, 100); ++i)
      {
        if (value % i == 0)
        {
          dividers.push_back(i);
        }
      }
      possibleTileSizes.push_back(dividers);
    }
    for (size_t NumberLoops = 2; NumberLoops <= iterationDomain.size() - 1; ++NumberLoops)
    {
      SmallVector<SmallVector<int64_t, 4>, 4> newCombinations =
          generateTileForAllOpCombinations(NumberLoops, possibleTileSizes, upperBounds);
      tileCombinations.insert(tileCombinations.end(), newCombinations.begin(), newCombinations.end());
    }

    SmallVector<SmallVector<int64_t, 4>, 4> SelectedTileCombinations;
    // SelectedTileCombinations.push_back({2, 200});
    /*std::sample(
        tileCombinations.begin(),
        tileCombinations.end(),
        std::back_inserter(SelectedTileCombinations),
        1,
        std::mt19937{std::random_device{}()});*/
    for (const auto &candidate : tileCombinations)
    {

      MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
      Node *ChildNode = new Node(ClonedCode, node->getCurrentStage());

      std::vector<Transformation *> TransList = node->getTransformationList();
      ChildNode->setTransformationList(TransList);

      Parallelization *parallelization =
          new Parallelization(&tileableOp,
                              CurrentStage,
                              candidate,
                              context);

      ChildNode->setTransformation(parallelization);

      ChildNode->addTransformation(parallelization);

      ChildNodes.push_back(ChildNode);
    }
    // ChildNodesList.push_back(ChildNodes);
  } //} });
  int OpIndex = 0;
  // for (auto ChildNodes : ChildNodesList)
  //{
  for (auto node : ChildNodes)
  {
    Operation *ClonedTarget = ((Operation *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                   .getIr());
    Parallelization *parallelization = (Parallelization *)node->getTransformation();

    SmallVector<mlir::linalg::LinalgOp, 4> linalgOps = getLinalgOps(ClonedTarget);

    // Tile and Fuse for tensors inputs (TODO: all tensor operands).
    bool changed = false;
    mlir::Operation *linalgOp = linalgOps[CurrentStage];

    // if(ClonedOpIndex==2){
    if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(linalgOp))
    {

      SmallVector<mlir::Operation *, 2> producers;
      ClonedTarget->walk([&](mlir::Operation *op)
                         {
                 // TODO: support multi-results.
                //if ((op->getName().getStringRef()).str() != "linalg.fill"){
                  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
                  {
                        producers.push_back(op);
                  //} 
                  } });
      std::reverse(producers.begin(), producers.end());

      IRRewriter rewriter(context);
      OpBuilder builder(context);

      std::optional<ArrayAttr> mapping;
      SmallVector<OpFoldResult, 4> opFoldResults;
      for (int64_t value : parallelization->getTileSizes())
      {
        opFoldResults.push_back(builder.getIndexAttr(value));
      }
      rewriter.setInsertionPoint(ClonedTileableOp);
      ArrayRef<OpFoldResult> tileSizes = llvm::makeArrayRef(opFoldResults);
      FailureOr<linalg::ForallTilingResult> tilingResult =
          linalg::tileToForallOpUsingTileSizes(rewriter, ClonedTileableOp, tileSizes, mapping);
      if (!failed(tilingResult))
        rewriter.replaceOp(ClonedTileableOp, tilingResult->tileOp->getResults());

      // IRRewriter rewriter1(context);
      std::string consumerTag = "consumer" + CurrentStage;
      TagSCFForAll(tilingResult->tileOp->getParentOp(), consumerTag);
      FuseOps(ClonedTarget, producers, consumerTag, 1);

      node->setCurrentStage(node->getCurrentStage() + producers.size());
      // FuseIntoContainingOperation(tilingResult->tileOp, ClonedTarget, rewriter1);
    }
    mlir::PassManager pm((ClonedTarget)->getName());

    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

    if (!mlir::failed(pm.run((ClonedTarget))))
      int ClonedOpIndex = 0;
    /*ClonedTarget->walk([&](Operation *op)
                       {

      if (mlir::TilingInterface ClonedTileableOp
                        = dyn_cast<mlir::TilingInterface>(op)) {
   if ((op->getName().getStringRef()).str() == "linalg.matmul"){
          ClonedOpIndex++;
        }

//if(ClonedOpIndex==2){
          if ((op->getName().getStringRef()).str() != "linalg.fill" ){

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

            IRRewriter rewriter1(context);

             /*PassManager pm((ClonedTarget)->getName());
            mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
          optPM.addPass(std::make_unique<TestLinalgGreedyFusion>());
          if (failed(pm.run(ClonedTarget))){}
          /*RewritePatternSet patterns =
              linalg::getLinalgTilingCanonicalizationPatterns(context);
          patterns.add<ExtractSliceOfPadTensorSwapPattern>(context);
          scf::populateSCFForLoopCanonicalizationPatterns(patterns);
          FrozenRewritePatternSet frozenPatterns(std::move(patterns));
          PassManager pm((ClonedTarget)->getName());
          pm.addPass(mlir::createLoopInvariantCodeMotionPass());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
          do {
            (void)applyPatternsAndFoldGreedily(ClonedTarget, frozenPatterns);
           if (failed(pm.run(ClonedTarget)))
              {}
          } while (succeeded(fuseLinalgOpsGreedily(ClonedTarget)));*/
    /*FuseIntoContainingOperation(tilingResult->tileOp, ClonedTarget, rewriter1);

}

//}

}
});*/
  }
  // OpIndex++;
  // }

  // Commented out when not using the nested loops for the applciation of the transformation
  /*SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }

  return ResChildNodes;*/
  // std::cout << "FUSION DONE\n";
  return ChildNodes;
}

/*void generateForAllOpCombinations(const SmallVector<int64_t, 4> &tileSizes,
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
    generateForAllOpCombinations(tileSizes,
                                 maxNumberLoops,
                                 currentLoop + 1,
                                 currentCombination,
                                 combinations);
  }
}

SmallVector<SmallVector<int64_t, 4>, 4>
generateTileForAllOpCombinations(int64_t maxNumberLoops,
                                 const std::vector<int64_t> &possibleTileSizes)
{
  SmallVector<int64_t, 4> tileSizes;
  std::copy(possibleTileSizes.begin(),
            possibleTileSizes.end(),
            std::back_inserter(tileSizes));

  SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<SmallVector<int64_t, 4>> combinations;

  generateForAllOpCombinations(tileSizes,
                               maxNumberLoops,
                               0,
                               currentCombination,
                               combinations);

  return SmallVector<SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                 combinations.end());
}*/