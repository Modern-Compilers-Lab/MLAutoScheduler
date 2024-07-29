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


using namespace mlir;

static LogicalResult FuseIntoContainingOperation(mlir::Operation *Target, std::string producerTag, std::string consumerTag)
{

  std::string transformDialectString = "module attributes {transform.with_named_sequence} { \n transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})  { \n  %0 = transform.structured.match attributes{\"" + producerTag + "\"} in %variant_op : (!transform.any_op) -> !transform.any_op \n %1 = transform.structured.match attributes{\"" + consumerTag + "\"} in %variant_op : (!transform.any_op) -> !transform.any_op \n transform.structured.fuse_into_containing_op %0 into %1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op) \n transform.yield}}";
  mlir::transform::TransformOptions options1;
  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile = 
      parseSourceString<mlir::ModuleOp>(transformDialectString, Target->getContext());
  llvm::StringRef entryPoint = "__transform_main";
  mlir::Operation *transformEntryPoint = 
      transform::detail::findTransformEntryPoint(Target, *moduleFromFile, entryPoint);

  return transform::applyTransformNamedSequence(
      Target, transformEntryPoint, *moduleFromFile,
      options1.enableExpensiveChecks(false));
}
static LogicalResult fuseLinalgOpsGreedily(Operation *f)
{
  OpBuilder b(f);
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<mlir::linalg::LinalgOp, 8> linalgOps;

  f->walk([&](mlir::linalg::LinalgOp op)
          {
            if (op->getNumResults() <= 1)
            {
              linalgOps.push_back(op);
            } });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  for (mlir::linalg::LinalgOp linalgOp : llvm::reverse(linalgOps))
  {

    for (OpOperand &opOperand : linalgOp->getOpOperands())
    {

      if (isa<MemRefType>(opOperand.get().getType()))
      {
        continue;
      }

      if (isa<RankedTensorType>(opOperand.get().getType()))
      {
        // Tile and Fuse tensor input.

        if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())
        {
          continue;
        }

        auto info = mlir::linalg::fuseProducerOfTensor(b, opOperand);
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
DiagnosedSilenceableFailure FuseOps(Operation *f, Operation *containingOp, SmallVector<mlir::Operation *, 2> producers, std::string tag, int &nbFused)
{
  // SmallVector<Operation *> fusedOps;
  // OpBuilder b(containingOp);
  int producerNb = 0;
  for (auto producerOp : producers)
  {
    if (auto func = dyn_cast<func::FuncOp>(producerOp->getParentOp()))
    {
      // std::cerr << "Dialect"<<producerOp->getParentOp()->getDialect()->getNamespace().str()<<std::endl;
      //  mlir::Operation *producerOp = producers[CurrentStage];
      std::string producerTag = tag + "_producer_" + std::to_string(producerNb);

      TagOperation(producerOp, producerTag);
      mlir::LogicalResult result = FuseIntoContainingOperation(f, producerTag, tag);
      if (result.succeeded())
      {
        nbFused++;
      }
      producerNb++;
    }
    else
    {
      std::cerr << "CONTAINEROP " << std::endl;
    }
  }
  std::cerr << "FUSIONS DONE" << nbFused << std::endl;
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
  /*SmallVector<mlir::Operation *, 2> NewProducers;
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
  }*/

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
                                                                        std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> LinalgOpStages)
{
  // Initialize vectors to store tile combinations
  SmallVector<SmallVector<int64_t, 4>, 4> tileCombinations;

  // Initialize a list to store child nodes (commented out now)
  // SmallVector<SmallVector<Node *, 2>> ChildNodesList;
  SmallVector<Node *, 2> ChildNodes;
  SmallVector<Node *, 2> ChildNodesFused;
  // Extract the transformed code IR from the input node
  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  // Get the top-level operation
  Operation *target = ((Operation *)(*CodeIr)
                           .getIr());

  Operation *op = LinalgOpStages["operation" + std::to_string(CurrentStage)].first;

  //  Check if the operation supports tiling
  if (mlir::TilingInterface tileableOp = dyn_cast<mlir::TilingInterface>(op))
  {
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
    std::cerr << "Elements of upperBounds:" << std::endl;
    for (const int64_t &element : upperBounds)
    {
      std::cerr << element << " ";
    }
    std::cerr << std::endl;
    SmallVector<SmallVector<int64_t, 4>, 4> possibleTileSizes;

    for (int64_t value : upperBounds)
    {
      llvm::SmallVector<int64_t, 4> dividers;
      for (int64_t i = 2; i < std::min((int)value, 40); ++i)
      {
        if (value % i == 0)
        {
          dividers.push_back(i);
        }
      }
      if (dividers.size() == 0){
        dividers.push_back(1);
      }
      possibleTileSizes.push_back(dividers);
    }
    std::cerr << "Elements of possibleTileSizes:" << std::endl;
    for (const auto &outer : possibleTileSizes)
    {
      // Iterate through the inner vector
      for (const int64_t &value : outer)
      {
        std::cerr << value << " ";
      }
      std::cerr << std::endl; // Move to a new line after each inner vector
    }
    for (size_t NumberLoops = 2; NumberLoops <= iterationDomain.size(); ++NumberLoops)
    {
      SmallVector<SmallVector<int64_t, 4>, 4> newCombinations =
          generateTileForAllOpCombinations(NumberLoops, possibleTileSizes, upperBounds);
      tileCombinations.insert(tileCombinations.end(), newCombinations.begin(), newCombinations.end());
    }

    SmallVector<SmallVector<int64_t, 4>, 4> SelectedTileCombinations;
    // SelectedTileCombinations.push_back({2, 200});
    // std::sample(
    //     tileCombinations.begin(),
    //     tileCombinations.end(),
    //     std::back_inserter(SelectedTileCombinations),
    //     20,
    //     std::mt19937{std::random_device{}()});
    std::cerr << "Candidates number = " << tileCombinations.size() << std::endl;
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

    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOps = getLinalgOps(ClonedTarget);

    // Tile and Fuse for tensors inputs (TODO: all tensor operands).
    bool changed = false;

    mlir::Operation *linalgOp = linalgOps["operation" + std::to_string(CurrentStage)].first;

    if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(linalgOp))
    {

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
      std::string consumerTag = "consumer" + std::to_string(CurrentStage);

      TagSCFForAll(tilingResult->tileOp, consumerTag);
      int nbFused = 0;
      SmallVector<mlir::Operation *, 2> producers;


      // ClonedTarget->walk([&](mlir::Operation *op)
      //                    {
      //              // TODO: support multi-results.
      //             //if ((op->getName().getStringRef()).str() != "linalg.fill"){
      //               if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      //               {
      //                     producers.push_back(op);
      //               //}
      //               } });
      // std::reverse(producers.begin(), producers.end());
      /*  SmallVector<mlir::Operation *, 2> sublist(std::min((int)producers.size(), 6));
      std::copy(producers.begin(), std::next(producers.begin(), std::min((int)producers.size(), 6)), sublist.begin());*/

      // Getting producers
      for (int i = 0; i < tilingResult->tileOp->getNumOperands(); i++)
      {
        std::cerr << "tilingResult->tileOp->getNumOperands()" << tilingResult->tileOp->getNumOperands() << std::endl;
        Operation *producer = tilingResult->tileOp->getOperand(i).getDefiningOp();
        if (producer)
        {
          producers.push_back(producer);
          for (int j = 0; j < producer->getNumOperands(); j++)
          {
            Operation *recProducer = producer->getOperand(j).getDefiningOp();
            if (recProducer)
            {
              producers.push_back(recProducer);

              for (int z = 0; z < recProducer->getNumOperands(); z++)
              {
                Operation *recProducer1 = recProducer->getOperand(z).getDefiningOp();
                if (recProducer1)
                {
                  producers.push_back(recProducer1);
                }
              }
            }
          }
        }
      }

      MLIRCodeIR *ClonedCodeForFusion = (MLIRCodeIR *)((MLIRCodeIR *)node->getTransformedCodeIr())->cloneIr();

      Node *ChildNodeForFusion = new Node(ClonedCodeForFusion, node->getCurrentStage());

      std::vector<Transformation *> TransList = node->getTransformationList();
      ChildNodeForFusion->setTransformationList(TransList);
      Operation *ClonedTargetForFusion = ((Operation *)(*((MLIRCodeIR *)ChildNodeForFusion->getTransformedCodeIr()))
                                              .getIr());

      ChildNodesFused.push_back(ChildNodeForFusion);

      /*std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOpsFused = getLinalgOps(ClonedTargetForFusion);
      mlir::Operation *linalgOpCurrentStageEqu = linalgOpsFused["operation" + std::to_string(CurrentStage)].first;
      linalgOpCurrentStageEqu->dump();
      for (int i = 0; i < linalgOpCurrentStageEqu->getNumOperands(); i++)
      {
        Operation *producer = linalgOpCurrentStageEqu->getOperand(i).getDefiningOp();
        if (producer)
        {
          producers.push_back(producer);
          for (int j = 0; j < producer->getNumOperands(); j++)
          {
            Operation *recProducer = producer->getOperand(j).getDefiningOp();
            if (recProducer)
            {
              producers.push_back(recProducer);
            }
          }
        }
      }*/
      /*for (const auto &[key, value] : linalgOpsFused)
      {
        size_t pos = key.find("operation");

        // Extract the number substring
        std::string number_str = key.substr(pos + strlen("operation"));

        // Convert the substring to an integer
        int number = std::stoi(number_str);

        if (number < CurrentStage && number >= CurrentStage - 2)
        {
          producers.push_back(value.first);
        }
      }*/
     std::cerr << " producers size : " << producers.size() << std::endl;

      // FuseOps(ClonedTargetForFusion, linalgOpCurrentStageEqu, producers, consumerTag, nbFused);
      FuseOps(ClonedTarget, tilingResult->tileOp, producers, consumerTag, nbFused);
      node->setCurrentStage(node->getCurrentStage() - 1);
      // ChildNodeForFusion->setCurrentStage(node->getCurrentStage());
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

    if (!mlir::failed(pm.run((ClonedTarget)))) int ClonedOpIndex = 0;

  }

  // std::copy(ChildNodesFused.begin(), ChildNodesFused.end(), ChildNodes.end());
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
