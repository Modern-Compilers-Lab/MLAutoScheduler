#include <iostream>
#include <cstdio>
#include <cstring>
#include <unordered_set>

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

#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h.inc"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

using namespace mlir;

template <typename PatternTy, typename... Args>
static FailureOr<mlir::linalg::LinalgOp> tryApply(mlir::Operation *operation, Args &&...args)
{
  // Check if the given operation has the type expected by the pattern.
  using OpTy = typename llvm::function_traits<
      decltype(&PatternTy::returningMatchAndRewrite)>::template arg_t<0>;
  auto op = dyn_cast<OpTy>(operation);
  if (!op)
    return failure();

  // Apply the pattern directly to the op.
  PatternTy pattern(operation->getContext(), std::forward<Args>(args)...);
  // We want to discourage direct use of PatternRewriter in APIs but In this
  // very specific case, an IRRewriter is not enough.
  struct TrivialPatternRewriter : public PatternRewriter
  {
  public:
    explicit TrivialPatternRewriter(mlir::MLIRContext *context)
        : PatternRewriter(context) {}
  };
  TrivialPatternRewriter rewriter(operation->getContext());
  rewriter.setInsertionPoint(operation);
  auto result = pattern.returningMatchAndRewrite(op, rewriter);

  if (failed(result))
    return failure();
  return cast<mlir::linalg::LinalgOp>(result->getOperation());
}

mlir::linalg::LinalgOp DecomposeOp(mlir::linalg::LinalgOp Target, mlir::IRRewriter *rewriter)
{

#define DOWNSCALE(trans)                                             \
  {                                                                  \
    std::cerr << "MACRO" << std::endl;                               \
    Target->dump();                                                  \
    FailureOr<mlir::linalg::LinalgOp> res = tryApply<trans>(Target); \
    if (succeeded(res))                                              \
    {                                                                \
      std::cerr << "SECCEEDED" << std::endl;                         \
      return Target;                                                 \
    }                                                                \
  }

#define DOWNSCALE_CALL(a, b) mlir::linalg::DownscaleSizeOneWindowed2DConvolution<a, b>
#define DOWNSCALE_NORMAL(a, b) DOWNSCALE(DOWNSCALE_CALL(a, b))

  DOWNSCALE_NORMAL(mlir::linalg::Conv2DNhwcHwcfOp, mlir::linalg::Conv1DNwcWcfOp)
  DOWNSCALE_NORMAL(mlir::linalg::Conv2DNchwFchwOp, mlir::linalg::Conv1DNcwFcwOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNhwcSumOp, mlir::linalg::PoolingNwcSumOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNchwSumOp, mlir::linalg::PoolingNcwSumOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNhwcMaxOp, mlir::linalg::PoolingNwcMaxOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNhwcMaxUnsignedOp, mlir::linalg::PoolingNwcMaxUnsignedOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNhwcMinOp, mlir::linalg::PoolingNwcMinOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNhwcMinUnsignedOp, mlir::linalg::PoolingNwcMinUnsignedOp)
  DOWNSCALE_NORMAL(mlir::linalg::PoolingNchwMaxOp, mlir::linalg::PoolingNcwMaxOp)
  DOWNSCALE(mlir::linalg::DownscaleDepthwiseConv2DNhwcHwcOp)
  DOWNSCALE(mlir::linalg::DownscaleConv2DOp)
#undef DOWNSCALE_NORMAL
#undef DOWNSCALE_CALL
#undef DOWNSCALE

  /*auto decomposableOp = dyn_cast<mlir::linalg::AggregatedOpInterface>(Target);
  if (decomposableOp)
  {
    std::cerr<<"WE ARE INSDE DECOMPEED"<< std::endl;
    FailureOr<SmallVector<Value>> maybeNewResults =
        decomposableOp.decomposeOperation(*rewriter);
    if (!failed(maybeNewResults))
    {
      std::cerr<<"NOT FAILED"<< std::endl;
      rewriter->replaceOp(decomposableOp, *maybeNewResults);
      for (Value val : *maybeNewResults)
      {

      }
    }
  }*/
  return Target;
}
struct VectorizationPattern : public RewritePattern
{
  explicit VectorizationPattern(mlir::MLIRContext *context,
                                bool vectorizeExtract = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeExtract) {}
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      PatternRewriter &rewriter) const override
  {
    std::cerr << "VectorizationPattern" << std::endl;
    mlir::linalg::LinalgOp linalgOp = dyn_cast<mlir::linalg::LinalgOp>(op);

    if (!linalgOp)
      return rewriter.notifyMatchFailure(op, "expected Linalg Op");
    return mlir::linalg::vectorize(rewriter, linalgOp, /*inputVectorSizes=*/{},
                                   /*scalableVecDims=*/{}, vectorizeNDExtract);
  }

private:
  /// Controls whether to vectorize `tensor.extract` when the input tensor is
  /// rank >= 2.
  bool vectorizeNDExtract = false;
};

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
  std::string extractedSubstring = 
          inputFilenameString.substr(inputFilenameString.find_last_of('/') + 1);
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
  // registry.addExtensions<mlir::TransformDialectExtension>();
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<scf::SCFDialect>();
  context.loadDialect<vector::VectorDialect>();
  context.loadDialect<mlir::transform::TransformDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile;
  mlir::ModuleOp transformModule =
      transform::detail::getPreloadedTransformModule(&context);

  transform::detail::parseTransformModuleFromFile(&context, 
                                                  inputFilename, 
                                                  moduleFromFile);

  // Parse the input file and obtain an MLIR module
  mlir::OwningOpRef<mlir::Operation *> module1 =
      (mlir::OwningOpRef<mlir::Operation *>)codeIr.parseInputFile(inputFilename, context);

  // Create a root Node for transformations
  Node *root = new Node(&codeIr, 0);
  // Initialize an evaluator for transformation evaluations
  EvaluationByExecution evaluator = 
      EvaluationByExecution(functionName + "_logs_best_exhustive_debug_single_op_vect_DNNFuison_producers_multiple_ties.txt");

  /*std::string RootEvel = evaluator.evaluateTransformation(root);
  root->setEvaluation(RootEvel);
  BeamSearch* searcher = new BeamSearch(3, &context, functionName);
  Node * res = searcher->runSearchMethod(root);*/

  // Store all the linalg operations found in the module
  std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOps = getLinalgOps(module1.get());

  // Track if transformations have been made or the best evaluation has been found
  bool changed = false;
  bool found = false;

  int stage = 0;
  Node *bestEval;
  bestEval = root;

  // Evaluate the root transformation
  std::string RootEvel = evaluator.evaluateTransformation(bestEval);
  bestEval->setEvaluation(RootEvel);
  changed = true;
  stage = bestEval->getCurrentStage();
  std::cerr << "Number of opeartions = " << linalgOps.size() << std::endl;

  /*for (auto op : linalgOps)
  {
    //(op.first)->dump();
    std::cout << "Operation classification: " 
    << getMappingTypeString(classifyLinalgOp(op.first)) << std::endl;
  }*/

  IRRewriter rewriter(&context);

  SmallVector<Node *, 2> nodesToVect;

  int fixedSize = linalgOps.size() - 1;
  stage = fixedSize;
  root->setCurrentStage(stage);
  bestEval->setCurrentStage(stage);

  // Loop through each stage while there's other operations to explore
  while (stage > 0)
  {
    // If no changes have been made in the current stage, move to the previous one
    if (!changed)
    {
      stage--;
      bestEval->setCurrentStage(stage);
    }
    // if ((linalgOps[stage]->getName().getStringRef()).str() != "linalg.fill")
    // if ((linalgOps[stage]->getName().getStringRef()).str() == "linalg.pooling_nchw_max" 
    // || (linalgOps[stage]->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
    //{
    SmallVector<Node *, 2> optList;

    // Get the linalg operation at the current stage
    mlir::Operation *newOp = ((mlir::Operation *)(*((MLIRCodeIR *)bestEval->getTransformedCodeIr()))
                                  .getIr());

    linalgOps = getLinalgOps(newOp);

    // Save the stage of the operation to be potentially vectorized
    int OpToVectStage = stage;

    std::cerr << " CURRET STAGE FOR PARALLIZATION : " << stage << std::endl;

    // Generate candidate parallelization options for the current stage
    optList = Parallelization::createParallelizationCandidates(bestEval, &context, stage, linalgOps);

    changed = false;

    // Set the child nodes of the current bestEval to the candidate nodes
    bestEval->setChildrenNodes(optList);

    // Loop through each candidate child node
    for (auto node : optList)
    {
      // #ADDED to ADD interchange for matmul Nazim test
      // mlir::Operation *newOp1 = ((mlir::Operation *)(*((MLIRCodeIR *)node1->getTransformedCodeIr()))
      //                                .getIr());
      // linalgOps = getLinalgOps(newOp1);
      // SmallVector<Node *, 2> listTilingInterchange = Tiling::createTilingCandidates(node1, &context, stage, linalgOps);

      // node1->setChildrenNodes(listTilingInterchange);
      // for (auto node : listTilingInterchange)
      // {
      // #ADDED to ADD interchange for matmul Nazim test

      nodesToVect.push_back(node);

      found = false;

      // Evaluate the candidate child node and update its evaluation
      std::string evel = evaluator.evaluateTransformation(node);
      node->setEvaluation(evel);

      // If the current candidate has a better evaluation than the current bestEval
      if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
      {
        std::cerr << "Changing the best Eval node" << std::endl;
        bestEval = node;
        stage = bestEval->getCurrentStage();
        changed = true;
      }

      // ## VECTORIZE ONE OP
      // Clone the transformed code IR of the candidate child node
      MLIRCodeIR *CodeIrVect = (MLIRCodeIR *)node->getTransformedCodeIr();
      MLIRCodeIR *ClonedCodeVect = (MLIRCodeIR *)CodeIrVect->cloneIr();

      // Create a new Node object with the cloned IR and current stage
      Node *VectNode = new Node(ClonedCodeVect, node->getCurrentStage());

      // Copy the transformation list from the candidate child node
      std::vector<Transformation *> TransList = node->getTransformationList();
      VectNode->setTransformationList(TransList);

      // Create a new vectorization transformation for the linalg operation at OpToVectStage
      Vectorization *vectorization =
          new Vectorization(&linalgOps["operation" + std::to_string(OpToVectStage)].first,
                            &context);

      // Add the vectorization transformation to the new Node
      VectNode->setTransformation(vectorization);
      VectNode->addTransformation(vectorization);

      mlir::Operation *ClonedOpVect = (mlir::Operation *)ClonedCodeVect->getIr();

      linalgOps = getLinalgOps(ClonedOpVect);

      // Get the cloned operation
      mlir::Operation *OpVect = linalgOps["operation" + std::to_string(OpToVectStage)].first;
      // Get the parent operation of the cloned operation
      mlir::Operation *OpVectParent = OpVect->getParentOp();

      //  linalgOps = getLinalgOps(ClonedOpVect);

      // for (auto oper : linalgOps)
      //{
      // OpVect = linalgOps[OpToVectStage];
      llvm::SmallVector<std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>, 4> parentOps; //= getLinalgOps(OpVectParent);

      int stageInParent = 0;
      // Store the parent's linalg operations
      OpVectParent->walk([&](mlir::linalg::LinalgOp op)
                         {
                  if (op->getNumResults() <= 1)
                  {
                        LinalgMappingClassification classification =  classifyLinalgOp(op);
                        parentOps.push_back(std::make_pair(op, classification));
                  } });

      // Loop through each parent linalg operation
      while (stageInParent < parentOps.size())
      {
        std::cerr << "Stage in parent = " << stageInParent << "  of : " << parentOps.size() << std::endl;

        bool ToDecompose = false;

        // Get the current operation
        mlir::Operation *op = parentOps[stageInParent].first;

        // If the parent operation needs tiling to be decomposed (e.g., conv2d, pooling)
        if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(op))
        {
          if ((op->getName().getStringRef()).str() == "linalg.pooling_nchw_max" 
          || (op->getName().getStringRef()).str() == "linalg.pooling_nchw_sum" 
          || (op->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
          {
            llvm::SmallVector<int64_t, 4> tilingSizes;
            OpBuilder builder(&context);
            SmallVector<Range> iterationDomain = ClonedTileableOp.getIterationDomain(builder);
            for (size_t i = 0; i < iterationDomain.size(); ++i)
            {
              if (i == 2)
              {
                tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the conv2D
              }
              else if (((op->getName().getStringRef()).str() == "linalg.pooling_nchw_max" 
              || (op->getName().getStringRef()).str() == "linalg.pooling_nchw_sum") 
              && i == 4)
              {
                tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the pooling
                break;
              }
              else if ((op->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw" 
              && i == 5)
              {
                tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the conv2D
                break;
              }
              else
              {
                tilingSizes.push_back(0);
              }
            }
            scf::SCFTilingOptions options;
            SmallVector<OpFoldResult> mixedSizes = getMixedSizes(tilingSizes, &context);
            options.setTileSizes(mixedSizes);

            std::cerr << "Modified tilingSizes " << (op->getName().getStringRef()).str() << " : [";
            for (size_t i = 0; i < tilingSizes.size(); ++i)
            {
              std::cerr << tilingSizes[i];
              if (i < tilingSizes.size() - 1)
              {
                std::cerr << ", ";
              }
            }
            std::cerr << "]\n";

            ToDecompose = true;

            FailureOr<scf::SCFTilingResult> maybeTiled =
                scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, options);

            std::cerr << "END OF TILE CONV2D" << std::endl;
            // If tiling was successful, replace the original op with the tiled version
            if (!failed(maybeTiled))
              rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
          }
        }

        if (ToDecompose)
        {
          std::cerr << "START DECOMPOSE" << std::endl;

          parentOps.clear();
          OpVectParent->walk([&](mlir::linalg::LinalgOp op)
                             {
                  if (op->getNumResults() <= 1)
                  {
                        LinalgMappingClassification classification =  classifyLinalgOp(op);
                        parentOps.push_back(std::make_pair(op, classification));
                  } });

          mlir::Operation *op1 = parentOps[stageInParent].first;

          if (mlir::linalg::LinalgOp LinalgOpVect = dyn_cast<mlir::linalg::LinalgOp>(op1))
          {
            mlir::linalg::LinalgOp DecomposedTarget = DecomposeOp(LinalgOpVect, &rewriter);
            std::cerr << "END DECOMPOSE" << std::endl;
          }
        }

        parentOps.clear();
        OpVectParent->walk([&](mlir::linalg::LinalgOp op)
                           {
                  if (op->getNumResults() <= 1)
                  {
                        LinalgMappingClassification classification =  classifyLinalgOp(op);
                        parentOps.push_back(std::make_pair(op, classification));
                  } });

        // Get the operation after decomposition
        mlir::Operation *op2 = parentOps[stageInParent].first;

        // If the op is a linalg op, attempt to vectorize it
        if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op2))
        {
          llvm::ArrayRef<int64_t> emptyArrayRef;

          llvm::ArrayRef<bool> boolArrayRef;

          mlir::LogicalResult vectorized = mlir::linalg::vectorize(rewriter, op2, emptyArrayRef,
                                                                   boolArrayRef, false);

          std::cerr << "GREEDILY APPLY AND FOLD " << vectorized.succeeded() << std::endl;

          RewritePatternSet patterns(&context);

          // Add vectorization canonicalization and lowering patterns to the set
          mlir::transform::detail::VectorizeOpGenericAdaptorBase::Properties props;

          // if (!props.getDisableTransferPermutationMapLoweringPatterns())
          mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

          // if (!props.getDisableMultiReductionToContractPatterns())
          vector::populateVectorReductionToContractPatterns(patterns);

          vector::populateSinkVectorBroadcastPatterns(patterns);

          // Add additional vectorization patterns for specific operations
          patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                       linalg::LinalgCopyVTWForwardingPattern>(&context, 2);
          vector::TransferReadOp::getCanonicalizationPatterns(patterns, &context);
          vector::TransferWriteOp::getCanonicalizationPatterns(patterns, &context);
          tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

          patterns.add<mlir::linalg::CopyVectorizationPattern>(&context);

          // if (props.getVectorizePadding())
          // linalg::populatePadOpVectorizationPatterns(patterns);

          if (!failed(applyPatternsAndFoldGreedily(ClonedOpVect, std::move(patterns))))
          {
            std::cerr << "APPLY NOT FAILED" << std::endl;
            parentOps.clear();

            OpVectParent->walk([&](mlir::linalg::LinalgOp op)
                               {
                  if (op->getNumResults() <= 1)
                  {
                        LinalgMappingClassification classification =  classifyLinalgOp(op);
                        parentOps.push_back(std::make_pair(op, classification));
                  } });
            stageInParent--;
          }
          if (!vectorized.succeeded())
          {
            parentOps.pop_back();
            stageInParent++;
          }
          // ######### GREEDILY APPLY AND FOLD #################*/
        }
        stageInParent++;
      }
      // ## VECTORIZE ONE OP

      std::cerr << "END VECT" << std::endl;

      evel = evaluator.evaluateTransformation(VectNode);
      VectNode->setEvaluation(evel);
      if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
      {
        std::cerr << "Changing the best Eval node" << std::endl;
        bestEval = VectNode;
        stage = bestEval->getCurrentStage();
        changed = true;
      }

      std::cerr << "New stage = " << stage << std::endl;
    }
    // }
  }

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
    std::cerr << "Failed to open file: " << std::endl;
  }
  outputFile << outputString;
  outputFile.close();

  // Display a message indicating the end of exploration
  std::cerr << "End of exploration!" << std::endl;
}