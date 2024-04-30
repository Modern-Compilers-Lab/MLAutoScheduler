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
namespace OptimizationEnum
{
  enum Optimization
  {
    Parallelization = 1,
    Tiling = 2,
    Vectorization = 3
  };
}
SmallVector<Node *, 2> func1(Node *root, int stage, SmallVector<mlir::linalg::LinalgOp, 4> linalgOps, mlir::MLIRContext *context, OptimizationEnum::Optimization optimization);
SmallVector<Node *, 2> func1(Node *root, int stage, SmallVector<mlir::linalg::LinalgOp, 4> linalgOps, mlir::MLIRContext *context, OptimizationEnum::Optimization optimization)
{
  SmallVector<Node *, 2> list;

  // for (int i = stage; i < linalgOps.size(); i++)

  if (stage < linalgOps.size())
  {
    SmallVector<Node *, 2> optList;
    switch (optimization)
    {
    case OptimizationEnum::Parallelization:
      optList = Parallelization::createParallelizationCandidates(root, context, stage, linalgOps);
      break;
    case OptimizationEnum::Tiling:
      optList = Tiling::createTilingCandidates(root, context, stage, linalgOps);
      break;
    default:
      std::cout << "Invalid Optimization Strategy" << std::endl;
      break;
    }
    list.insert(list.end(), optList.begin(), optList.end());
    if (optimization == OptimizationEnum::Parallelization)
    {
      for (Node *node : optList)
      {
        SmallVector<Node *, 2> tempList = func1(node, node->getCurrentStage(), linalgOps, context, optimization);
        list.insert(list.end(), tempList.begin(), tempList.end());
      }
    }
  }

  return list;
}

template <typename PatternTy, typename... Args>
static FailureOr<mlir::linalg::LinalgOp> tryApply(mlir::Operation *operation, Args &&...args)
{
  // Check if the given operation has the type expected by the pattern.
  using OpTy = typename llvm::function_traits<
      decltype(&PatternTy::returningMatchAndRewrite)>::template arg_t<0>;
  auto op = dyn_cast<OpTy>(operation);
  if (!op)
    return failure();
  std::cerr << "PART1" << std::endl;

  // Apply the pattern directly to the op.
  PatternTy pattern(operation->getContext(), std::forward<Args>(args)...);
  // We want to discourage direct use of PatternRewriter in APIs but In this
  // very specific case, an IRRewriter is not enough.
  std::cerr << "PART2" << std::endl;
  struct TrivialPatternRewriter : public PatternRewriter
  {
  public:
    explicit TrivialPatternRewriter(mlir::MLIRContext *context)
        : PatternRewriter(context) {}
  };
  TrivialPatternRewriter rewriter(operation->getContext());
  rewriter.setInsertionPoint(operation);
  std::cerr << "PART3" << std::endl;
  auto result = pattern.returningMatchAndRewrite(op, rewriter);

  std::cerr << "PART4" << std::endl;
  if (failed(result))
    return failure();
  std::cerr << "PART5" << std::endl;
  return cast<mlir::linalg::LinalgOp>(result->getOperation());
}

mlir::linalg::LinalgOp DecomposeOp(mlir::linalg::LinalgOp Target, mlir::IRRewriter *rewriter)
{
  std::cerr << "DECOMPOSE" << std::endl;
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
// Function to insert a function into another function
void insertFunction(mlir::ModuleOp &moduleOp, mlir::Operation *funcToInsert,
                    mlir::StringRef targetFunctionName)
{
  // Create an OpBuilder to insert operations
  mlir::OpBuilder builder(moduleOp);
  auto parent = builder.getBlock();
  std::cout << "GOT BLOCK" << std::endl;
  // builder.getBlock()->dump();
  //  Find the target function where insertion should occur
  mlir::Operation *targetOp = nullptr;
  moduleOp->walk([&](mlir::Operation *op)
                 {
               
           if (op->getName().getStringRef().str() == "func.func") {
      targetOp = op;
       std::cout << "GOT TARGET" <<std::endl;
    } });

  /*mlir::Operation *targetOp = moduleOp.getOperation(targetFunctionName);
   if (!targetOp) {
     llvm::errs() << "Error: Target function '" << targetFunctionName << "' not found\n";
     return;
   }*/

  // Set the insertion point before the target function
  builder.setInsertionPoint(targetOp);

  // Insert the function to be inserted
  builder.insert(funcToInsert);
}
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
  // std::cout << "  ENDED " << std::endl;
  transform::detail::parseTransformModuleFromFile(&context, inputFilename, moduleFromFile);
  std::vector<Dialect *> li = context.getLoadedDialects();
  /*for (mlir::Dialect *dialect : li)
  {
    std::cout << "  - " << dialect->getNamespace().data() << std::endl;
  }*/
  // Parse the input file and obtain an MLIR module
  mlir::OwningOpRef<mlir::Operation *> module1 =
      (mlir::OwningOpRef<mlir::Operation *>)codeIr.parseInputFile(inputFilename, context);

  // Dump the contents of the parsed module
  //(*module1)->dump();

  // Create a root Node for transformations
  Node *root = new Node(&codeIr, 0);
  EvaluationByExecution evaluator = EvaluationByExecution(functionName + "_logs_best_exhustive_debug_single_op_vect_all.txt");

  // Evaluate the root transformation
  /*std::string RootEvel = evaluator.evaluateTransformation(root);
  root->setEvaluation(RootEvel);
  BeamSearch* searcher = new BeamSearch(3, &context, functionName);
  Node * res = searcher->runSearchMethod(root);*/

  // Initialize an evaluator for transformation evaluations
  // EvaluationByExecution evaluator =  EvaluationByExecution(functionName+"_logs_best.txt");
  SmallVector<mlir::linalg::LinalgOp, 4> linalgOps = getLinalgOps(module1.get());

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  int stage = 0;
  Node *bestEval;
  bestEval = root;
  bool found = false;

  // Evaluate the root transformation
  std::string RootEvel = evaluator.evaluateTransformation(bestEval);
  bestEval->setEvaluation(RootEvel);
  changed = true;
  stage = bestEval->getCurrentStage();
  std::cerr << "Number of opeartions = " << linalgOps.size() << std::endl;
  IRRewriter rewriter(&context);
  SmallVector<Node *, 2> nodesToVect;
  while (stage < linalgOps.size() - 1)
  {

    if (!changed)
    {
      stage++;
      bestEval->setCurrentStage(stage);
    }
    // if ((linalgOps[stage]->getName().getStringRef()).str() != "linalg.fill")
    // if ((linalgOps[stage]->getName().getStringRef()).str() == "linalg.pooling_nchw_max" || (linalgOps[stage]->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
    //{
    SmallVector<Node *, 2> optList;
    mlir::Operation *newOp = ((mlir::Operation *)(*((MLIRCodeIR *)bestEval->getTransformedCodeIr()))
                                  .getIr());
    linalgOps = getLinalgOps(newOp);
    int OpToVectStage = stage;
    auto start = std::chrono::high_resolution_clock::now();
    mlir::Operation *tagged = linalgOps[stage];
    std::cerr << " CUURET STAGE FOR PARA : " << stage << std::endl;
    std::cerr << "Number of opeartions IPDATED = " << linalgOps.size() << std::endl;
    // tagged->dump();
    // mlir::Attribute attr = UnitAttr::get(&context);
    // tagged->setAttr(tagged->getName().getStringRef(), attr);

    optList = Parallelization::createParallelizationCandidates(bestEval, &context, stage, linalgOps);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by candaidte generation: " << duration.count() << " microseconds" << std::endl;
    changed = false;
    bestEval->setChildrenNodes(optList);
    for (auto node : optList)
    {
      nodesToVect.push_back(node);
      auto start_node = std::chrono::high_resolution_clock::now();

      found = false;
      std::string evel = evaluator.evaluateTransformation(node);
      node->setEvaluation(evel);

      if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
      {
        std::cerr << "We changed the node\n";
        bestEval = node;
        stage = bestEval->getCurrentStage();
        changed = true;
      }

      // ## VECTORIZE ONE OP
      MLIRCodeIR *CodeIrVect = (MLIRCodeIR *)node->getTransformedCodeIr();
      MLIRCodeIR *ClonedCodeVect = (MLIRCodeIR *)CodeIrVect->cloneIr();
      Node *VectNode = new Node(ClonedCodeVect, node->getCurrentStage());

      std::vector<Transformation *> TransList = node->getTransformationList();
      VectNode->setTransformationList(TransList);

      Vectorization *vectorization =
          new Vectorization(&linalgOps[OpToVectStage],
                            // candidate,
                            &context);

      VectNode->setTransformation(vectorization);

      VectNode->addTransformation(vectorization);
      std::cerr << "FINISHED CREATING NODE\n";
      mlir::Operation *ClonedOpVect = (mlir::Operation *)ClonedCodeVect->getIr();
      start = std::chrono::high_resolution_clock::now();
      linalgOps = getLinalgOps(ClonedOpVect);
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "Time taken by getLinalg ops: " << duration.count() << " microseconds" << std::endl;

      bool ToDecompose = false;
      std::cerr << "FINISHED CREATING LIST\n";
      mlir::Operation *OpVect = linalgOps[OpToVectStage];
      // OpVect->dump();
      if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(OpVect))
      {
        if ((OpVect->getName().getStringRef()).str() == "linalg.pooling_nchw_max" || (OpVect->getName().getStringRef()).str() == "linalg.pooling_nchw_sum" || (OpVect->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
        {
          llvm::SmallVector<int64_t, 4> tilingSizes;
          OpBuilder builder(&context);
          SmallVector<Range> iterationDomain = ClonedTileableOp.getIterationDomain(builder);
          std::cerr << "POOLING OR CONV2D" << iterationDomain.size() << std::endl;
          for (size_t i = 0; i < iterationDomain.size(); ++i)
          {
            if (i == 2)
            {
              tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the conv2D
            }
            else if (((OpVect->getName().getStringRef()).str() == "linalg.pooling_nchw_max" || (OpVect->getName().getStringRef()).str() == "linalg.pooling_nchw_sum") && i == 4)
            {
              tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the pooling
              break;
            }
            else if ((OpVect->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw" && i == 5)
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
          std::cout << "TRYING TO TILE CONV2D" << std::endl;
          SmallVector<OpFoldResult> mixedSizes = getMixedSizes(tilingSizes, &context);
          options.setTileSizes(mixedSizes);
          std::cout << "Modified tilingSizes " << (OpVect->getName().getStringRef()).str() << " : [";
          for (size_t i = 0; i < tilingSizes.size(); ++i)
          {
            std::cout << tilingSizes[i];
            if (i < tilingSizes.size() - 1)
            {
              std::cout << ", ";
            }
          }
          std::cout << "]\n";

          std::cout << "TRYING TO TILE CONV2D" << std::endl;

          ToDecompose = true;

          FailureOr<scf::SCFTilingResult> maybeTiled =
              scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, options);
          std::cout << "END OF TILE CONV2D" << std::endl;

          if (!failed(maybeTiled))
            rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
        }
      }
      // ClonedOpVect->dump();
      if (ToDecompose)
      {
        std::cout << "START DECOMPOSE" << std::endl;
        linalgOps = getLinalgOps(ClonedOpVect);
        OpVect = linalgOps[OpToVectStage];
        if (mlir::linalg::LinalgOp LinalgOpVect = dyn_cast<mlir::linalg::LinalgOp>(OpVect))
        {
          std::cout << "IS LINALG" << std::endl;

          mlir::linalg::LinalgOp DecomposedTarget = DecomposeOp(LinalgOpVect, &rewriter);
          // MLIRCodeIR *DecomposedCodeIr = (MLIRCodeIR *)ClonedCodeVect->setMLIRIR(DecomposedTarget);
          // VectNode->setTransformedCodeIr(DecomposedCodeIr);
          std::cout << "END DECOMPOSE" << std::endl;
        }

        // DecomposedTarget->dump();
      }
      // ClonedOpVect->dump();
      // IRRewriter rewriter(&context);
      std::cout << "START VECT" << std::endl;
      ClonedOpVect->dump();
      linalgOps = getLinalgOps(ClonedOpVect);

      // for (auto oper : linalgOps)
      //{
      OpVect = linalgOps[OpToVectStage];
      mlir::Operation *OpVectParent = OpVect->getParentOp();
      OpVectParent->walk([&](mlir::Operation *op)
                         {
               if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
                  llvm::ArrayRef<int64_t> emptyArrayRef;

                  llvm::ArrayRef<bool> boolArrayRef;

                  mlir::linalg::vectorize(rewriter, op, emptyArrayRef,
                                                      boolArrayRef, false);
                  std::cerr << "GREEDILY APPLY AND FOLD " << std::endl;

                  RewritePatternSet patterns(&context);
                  mlir::transform::detail::VectorizeOpGenericAdaptorBase::Properties props;


                  //if (!props.getDisableTransferPermutationMapLoweringPatterns())
                    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

                  //if (!props.getDisableMultiReductionToContractPatterns())
                    vector::populateVectorReductionToContractPatterns(patterns);

                  vector::populateSinkVectorBroadcastPatterns(patterns);

                  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                                linalg::LinalgCopyVTWForwardingPattern>(&context, 2);
                  vector::TransferReadOp::getCanonicalizationPatterns(patterns, &context);
                  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, &context);
                  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

                  patterns.add<mlir::linalg::CopyVectorizationPattern>(&context);

                  //if (props.getVectorizePadding())
                    linalg::populatePadOpVectorizationPatterns(patterns);

                  if (!failed(applyPatternsAndFoldGreedily(ClonedOpVect, std::move(patterns))))
                  {
                    std::cerr << "APPLY NOT FAILED" << std::endl;
                  }
       // ######### GREEDILY APPLY AND FOLD #################*/

               } });
      // ## VECTORIZE ONE OP

      std::cout << "END VECT" << std::endl;
      //}
      evel = evaluator.evaluateTransformation(VectNode);
      VectNode->setEvaluation(evel);
      if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
      {
        std::cerr << "We changed the node" << std::endl;
        bestEval = VectNode;
        // MLIRCodeIR *CodeIrTEST = (MLIRCodeIR *)bestEval->getTransformedCodeIr();
        // mlir::Operation *targetTEST = ((mlir::Operation *)(*CodeIrTEST)
        //                                  .getIr());
        // targetTEST->dump();
        stage = bestEval->getCurrentStage();
        changed = true;
      }

      /*ClonedOpVect->walk([&](mlir::Operation *op)
            {
                // TODO: support multi-results.
               if ((op->getName().getStringRef()).str() == "scf.forall"){
                 OpVect = op;
                 }
                  });*/
      /*std::string str;
      llvm::raw_string_ostream operationOut(str);
      OpVect->print(operationOut);

      size_t pos = str.find('=');
      if (pos != std::string::npos)
      {
        str = str.substr(pos + 1);
      }
      std::cout << "OPERATION " << str << std::endl;
      std::string res = function_wrapper(str);

      std::cout << "OPERATION WRAPPED " << res << std::endl;
      mlir::MLIRContext ctxVect;
      mlir::OwningOpRef<mlir::Operation *> moduleVect = parseSourceString(res, &context);
      std::cout << "OPERATION PARSED " << std::endl;
      (*moduleVect)->dump();

      std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %variant_op: (!transform.any_op) -> !transform.any_op \n %func_0 = transform.structured.vectorize %func {vectorize_padding}: (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 :(!transform.any_op) -> (!transform.any_op) \n transform.structured.hoist_redundant_tensor_subsets %func_01 :(!transform.any_op) -> () }";

      mlir::PassManager pm((*moduleVect)->getName());

      // Apply any generic pass manager command line options and run the pipeline.
      applyPassManagerCLOptions(pm);

      pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
      if (!mlir::failed(pm.run((*moduleVect))))
      {
      }
      // Target->dump();
      std::cout << "END VECT FINAL" << std::endl;
      (OpVect)->dump();
      if (mlir::isa<mlir::ModuleOp>(*ClonedOpVect)) {
        std::cout << "IT IS MODULEOP" << std::endl;
        mlir::ModuleOp moduleOp =mlir::dyn_cast<mlir::ModuleOp>(ClonedOpVect);
        insertFunction(moduleOp, (*moduleVect), "matmul");
        ClonedOpVect->dump();
        (OpVect)->dump();

        // Get operands and result of linalg.matmul
        mlir::Value operand0 = OpVect->getOperands()[0];
        mlir::Value operand1 = OpVect->getOperands()[1];
        mlir::Value result = OpVect->getResults()[0];

        // Create arguments for func_call
        //mlir::Value arg0 = ClonedOpVect->getArgument(operand0.cast<mlir::OpResult>().getResultNumber());
        //mlir::Value arg1 = ClonedOpVect->getArgument(operand1.cast<mlir::OpResult>().getResultNumber());

        // Create OpBuilder before the linalg.matmul
        mlir::OpBuilder builder(OpVect);
        builder.setInsertionPoint(OpVect);

        //mlir::StringAttr func_call= mlir::StringAttr::get(&context, "@func_call");

          //OperationState state(OpVect->getLoc(), func_call, OpVect->getOperands(), OpVect->getOperands().getTypes());
        //mlir::LLVM::CallOp::build(builder, state, OpVect->getResults(),  func_call,  OpVect->getOperands())
        mlir::TypeRange  tr = OpVect->getResults().getTypes();
        mlir::ValueRange  vr = OpVect->getOperands();
        mlir::TypeRange allButLast = llvm::make_range(tr.end(), tr.end());
        mlir::ValueRange allButLastOp = llvm::make_range(vr.begin(), std::prev(vr.end()));

        // Call func_call with extracted arguments
        auto funcCallResult =
            builder.create<mlir::func::CallOp>(OpVect->getLoc(),tr,SymbolRefAttr::get((*moduleVect)),
                                        vr);
        funcCallResult->dump();
        // Replace uses of linalg.matmul result with func_call result
       result.replaceAllUsesWith(funcCallResult->getResults()[0]);

        // Erase the original linalg.matmul operation
        OpVect->erase();


      std::cout << "END MODULEOP" << std::endl;

        // Use moduleOp for further processing
      } else {
        llvm::errs() << "Error: Operation* does not point to a ModuleOp\n";
      }
       ClonedOpVect->dump();
      // auto module = ClonedOpVect->getParentOfType<mlir::ModuleOp>();

      // module->dump();
      // ClonedOpVect->getParentOfType<mlir::ModuleOp>().push_back(*moduleVect);

      // OpVect->dump();*/

      int nbFunc = 0;
      mlir::Operation *parentOp;

      /*ClonedOpVect->walk([&](mlir::Operation *op)
                         {
                             if (nbFunc == 9) {parentOp = op;op->dump();}
                // TODO: support multi-results.
              if ((op->getDialect()->getNamespace()).str() == "func"){
                op->dump();
                nbFunc++;

              } });*/
      /*mlir::PassManager pm((ClonedOpVect)->getName());

      // Apply any generic pass manager command line options and run the pipeline.
      applyPassManagerCLOptions(pm);
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());

      if (!mlir::failed(pm.run((ClonedOpVect))))
      {
      }*/
      /*std::cout << "PARENT OP" << std::endl;
      std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %variant_op: (!transform.any_op) -> !transform.any_op \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func :(!transform.any_op) -> (!transform.any_op) \n transform.structured.hoist_redundant_tensor_subsets %func_01 :(!transform.any_op) -> () }";
      std::cout << "START AFTER VECT\n";

      mlir::PassManager pm((ClonedOpVect)->getName());

      // Apply any generic pass manager command line options and run the pipeline.
      applyPassManagerCLOptions(pm);

      pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
      if (!mlir::failed(pm.run((ClonedOpVect))))
      {
        ClonedOpVect->dump();
      }*/
      // ######### GREEDILY APPLY AND FOLD #################
      /*std::cerr << "GREEDILY APPLY AND FOLD " << std::endl;
       OpVect = linalgOps[OpToVectStage];
       OpVect->getParentOp()->dump();
       RewritePatternSet patterns(&context);
       mlir::transform::detail::VectorizeOpGenericAdaptorBase::Properties props;
       patterns.add<VectorizationPattern>(&context);

       if (!props.getDisableTransferPermutationMapLoweringPatterns())
         mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

       if (!props.getDisableMultiReductionToContractPatterns())
         vector::populateVectorReductionToContractPatterns(patterns);

       vector::populateSinkVectorBroadcastPatterns(patterns);

       patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                    linalg::LinalgCopyVTWForwardingPattern>(&context, 2);
       vector::TransferReadOp::getCanonicalizationPatterns(patterns, &context);
       vector::TransferWriteOp::getCanonicalizationPatterns(patterns, &context);
       tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

       patterns.add<mlir::linalg::CopyVectorizationPattern>(&context);

       if (props.getVectorizePadding())
         linalg::populatePadOpVectorizationPatterns(patterns);

       if (!failed(applyPatternsAndFoldGreedily(OpVect->getParentOp(), std::move(patterns))))
       {
         std::cerr << "APPLY NOT FAILED" << std::endl;
       }
       // ######### GREEDILY APPLY AND FOLD #################*/
      // results.push_back(oper);
      // std::cerr << "Stage at the end" << stage << std::endl;
      /*if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
      {
      }
      else
      {
        // delete node;
      }*/

      auto end_node = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end_node - start_node);
      std::cout << "Time taken by one node: " << duration.count() << " microseconds" << std::endl;
      std::cerr << "END PARA\n";
      // changed = true;
      // stage = 0;
    }

    /*}
    else
    {
      stage++;
      bestEval->setCurrentStage(stage);
    }*/
  }
  for (Node *node : nodesToVect)
  {
    changed = true;
    stage = 0;
    bestEval = node;
    mlir::Operation *BestTarget = ((mlir::Operation *)(*((MLIRCodeIR *)bestEval->getTransformedCodeIr()))
                                       .getIr());
    linalgOps = getLinalgOps(BestTarget);
    std::cerr << "Numbes of opeartions Tiling  = " << linalgOps.size() << std::endl;
    // BestTarget->dump();

    while (stage < linalgOps.size())
    {
      std::cerr << "STAGe = " << stage << std::endl;

      if ((linalgOps[stage]->getParentOp()->getName().getStringRef()).str() != "scf.forall" && (linalgOps[stage]->getParentOp()->getName().getStringRef()).str() != "scf.for")
      {
        SmallVector<Node *, 2> optList1 = Tiling::createTilingCandidates(bestEval, &context, stage, linalgOps);
        changed = false;
        for (auto node1 : optList1)
        {
          std::string evel1 = evaluator.evaluateTransformation(node1);
          node1->setEvaluation(evel1);

          if (std::stod(bestEval->getEvaluation()) > std::stod(evel1))
          {
            std::cerr << "We changed the node\n";
            bestEval = node1;
            // stage = bestEval->getCurrentStage();
            changed = true;
          }
          /*else
          {
            // delete node1;
          }*/
        }
      }

      stage++;
      bestEval->setCurrentStage(stage);
    }
    /*// ## VECTORIZE THE WHOLE CODE
      found = false;
      std::cout << "CHECKING TILING "<<found<< std::endl;


      MLIRCodeIR *CodeIr = (MLIRCodeIR *)bestEval->getTransformedCodeIr();
      mlir::Operation *Target = (mlir::Operation *)CodeIr->getIr();
      Target->dump();
      Target->walk([&](mlir::Operation *op)
                   {
         if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(op)) {
             //if ((op->getName().getStringRef()).str() != "linalg.fill" ){
                  if ((op->getParentOp()->getName().getStringRef()).str() != "scf.forall" && (op->getParentOp()->getName().getStringRef()).str() != "scf.for" ){
                 found = true;
                 std::cout << "IT'S FOUND"<< std::endl;
                 op->dump();
             }
             //}


           } });
      std::cout << "CHECKING TILING "<<found<< std::endl;
      if (!found)
      {
        std::cout << "WE VECTORIZING "<<std::endl;
        SmallVector<Node *, 2> list_vect = Vectorization::createVectorizationCandidates(bestEval, &context);
        bestEval->setChildrenNodes(list_vect);

        for (auto node3 : list_vect)
        {
          std::string evel2 = evaluator.evaluateTransformation(node3);
          node3->setEvaluation(evel2);
          if (std::stod(bestEval->getEvaluation()) > std::stod(evel2))
          {
            std::cout << "Changing nodes\n";
            // bestEval = node3;
          }
        }
      }*/
  }
  /*OptimizationEnum::Optimization optimization = OptimizationEnum::Parallelization;

  SmallVector<Node *, 2> toExplore = func1(root, 0, linalgOps, &context, optimization);
  root->setChildrenNodes(toExplore);
  for (auto node : toExplore)
  {
    found = false;
    std::string evel = evaluator.evaluateTransformation(node);
    node->setEvaluation(evel);

    if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
    {
      bestEval = node;
    }
    /*std::unordered_set<int> encounteredStages;

    for (Transformation *transformation : node->getTransformationList())
    {
      std::string type = transformation->getType();

      if (type == "Tiling")
      {
        int stage = ((Tiling *)transformation)->getOperationStage();
        // Check if the operation stage is within the valid range
        if (stage >= 0 && stage <= linalgOps.size())
        {
          // Mark the stage as encountered
          encounteredStages.insert(stage);
        }
      }
      /*if (type == "Parallelization")
      {
        int stage = ((Parallelization *)transformation)->getOperationStage();
        // Check if the operation stage is within the valid range
        if (stage >= 0 && stage <= linalgOps.size())
        {
          // Mark the stage as encountered
          encounteredStages.insert(stage);
        }
      }
    }*/
  /* MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
   mlir::Operation *Target = (mlir::Operation *)CodeIr->getIr();
   Target->walk([&](mlir::Operation *op)
                {
           if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(op)) {
               //if ((op->getName().getStringRef()).str() != "linalg.fill" ){
                    if ((op->getParentOp()->getName().getStringRef()).str() != "scf.forall" && (op->getParentOp()->getName().getStringRef()).str() != "scf.for" ){
                   found = true;
               }
               //}


             } });
   // Check if all required stages are covered
   if (!found)
   {
     std::cout << "VECTORIZING WITH ONLY ON TILING\n";
     MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
     MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
     Node *ChildNode = new Node(ClonedCode, node->getCurrentStage());
     std::vector<Transformation *> TransList = node->getTransformationList();
     ChildNode->setTransformationList(TransList);

     SmallVector<Node *, 2> list_vect1 = Vectorization::createVectorizationCandidates(ChildNode, &context);
     ChildNode->setChildrenNodes(list_vect1);

     for (auto node3 : list_vect1)
     {
       std::string evel3 = evaluator.evaluateTransformation(node3);
       node3->setEvaluation(evel3);
       if (std::stod(bestEval->getEvaluation()) > std::stod(evel3))
       {
         std::cout << "Changing nodes\n";
         //bestEval = node3;
       }
     }
   }
  }

  OptimizationEnum::Optimization optimization2 = OptimizationEnum::Tiling;
  SmallVector<Node *, 2> toExploreSecond = func1(bestEval, 0, linalgOps, &context, optimization2);
  bestEval->setChildrenNodes(toExploreSecond);
  for (auto node1 : toExploreSecond)
  {
   std::string evel1 = evaluator.evaluateTransformation(node1);
   node1->setEvaluation(evel1);

   if (std::stod(bestEval->getEvaluation()) > std::stod(evel1))
   {
     bestEval = node1;
   }
   found = false;
   MLIRCodeIR *CodeIr = (MLIRCodeIR *)node1->getTransformedCodeIr();
   mlir::Operation *Target = (mlir::Operation *)CodeIr->getIr();
   Target->walk([&](mlir::Operation *op)
                {
           if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(op)) {
               //if ((op->getName().getStringRef()).str() != "linalg.fill" ){
                    if ((op->getParentOp()->getName().getStringRef()).str() != "scf.forall" && (op->getParentOp()->getName().getStringRef()).str() != "scf.for" ){
                   found = true;
               }
               //}


             } });
   /*std::unordered_set<int> encounteredStages1;

   for (Transformation *transformation : node1->getTransformationList())
   {
     std::string type = transformation->getType();

     if (type == "Tiling")
     {
       int stage = ((Tiling *)transformation)->getOperationStage();
       // Check if the operation stage is within the valid range
       if (stage >= 0 && stage <= linalgOps.size())
       {
         // Mark the stage as encountered
         encounteredStages1.insert(stage);
       }
     }
     /*if (type == "Parallelization")
     {

       int stage = ((Parallelization *)transformation)->getOperationStage();
       // Check if the operation stage is within the valid range
       if (stage >= 0 && stage <= linalgOps.size())
       {
         // Mark the stage as encountered
         encounteredStages1.insert(stage);
       }
     }
  }*/
  /*if (!found)
  {
    std::cout << "WE VECTORIZING \n";
    SmallVector<Node *, 2> list_vect = Vectorization::createVectorizationCandidates(node1, &context);
    node1->setChildrenNodes(list_vect);

    for (auto node3 : list_vect)
    {
      std::string evel2 = evaluator.evaluateTransformation(node3);
      node3->setEvaluation(evel2);
      if (std::stod(bestEval->getEvaluation()) > std::stod(evel2))
      {
        std::cout << "Changing nodes\n";
        //bestEval = node3;
      }
    }
  }
  }*/

  /*for (mlir::linalg::LinalgOp linalgOp : linalgOps)
  {
    if ((linalgOp->getName().getStringRef()).str() != "linalg.fill")
    {
      std::cout << " OPERATION ;###############################\n";
      linalgOp->dump();
      SmallVector<Node *, 2> ParaList = Parallelization::createParallelizationCandidates(bestEval, &context, stage, linalgOps);

      // Interlist.push_back(root);
      std::cout << "size = " << ParaList.size() << std::endl;
      // Set children nodes for the root Node
      bestEval->setChildrenNodes(ParaList);
      // Loop through the children nodes of the root
      for (auto ChildNode : ParaList)
      {
        std::string evel = evaluator.evaluateTransformation(ChildNode);
        ChildNode->setEvaluation(evel);

        if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
        {

          bestEval = ChildNode;
        }
        MLIRCodeIR *CodeIr = (MLIRCodeIR *)ChildNode->getTransformedCodeIr();

        mlir::Operation *target = ((mlir::Operation *)(*CodeIr)
                                       .getIr());
        std::cout << " TILING ,###############################\n";
        SmallVector<mlir::linalg::LinalgOp, 4> linalgOpsTiling = getLinalgOps(target);
        int stage1 = linalgOpsTiling.size() - 1;
        /*for (mlir::linalg::LinalgOp linalgOp : linalgOps)
       {*/
  // SmallVector<Node *, 2> list1 = Tiling::createTilingCandidates(ChildNode, &context, stage, linalgOpsTiling);

  /*/*MLIRCodeIR *ToCloneCodeIr = (MLIRCodeIR *)ChildNode->getTransformedCodeIr();
  MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)ToCloneCodeIr->cloneIr();
  Node* ClonedNode = new Node (ClonedCode);

  std::vector<Transformation*> TransList= ChildNode->getTransformationList();
  ClonedNode->setTransformationList(TransList);
   list1.insert(list1.begin(), ClonedNode);*/

  // list1.push_back(ClonedNode);
  /*ChildNode->setChildrenNodes(list1);*/

  // Loop through the children nodes of the current child node
  /*for (auto node1 : list1)
  {

    std::string evel = evaluator.evaluateTransformation(node1);
    node1->setEvaluation(evel);

    if (std::stod(bestEval->getEvaluation()) > std::stod(evel))
    {
      std::cout << "Changing nodes\n";
      bestEval = node1;
    }
    /*MLIRCodeIR *ToCloneCodeIrInter = (MLIRCodeIR *)ChildNode->getTransformedCodeIr();
    MLIRCodeIR* ClonedCodeInter =  (MLIRCodeIR*)ToCloneCodeIrInter->cloneIr();
    Node* ClonedNodeInter = new Node (ClonedCodeInter);

    std::vector<Transformation*> TransList= ChildNode->getTransformationList();
    ClonedNodeInter->setTransformationList(TransList);*/

  // SmallVector<Node *, 2> list = Interchange::createInterchangeCandidates(node1, &context);

  // node1->setChildrenNodes(list);

  // Loop through the children nodes of the current node1
  /*for (auto node2 : list1)
  {
    std::string evel1 = evaluator.evaluateTransformation(node2);
    node2->setEvaluation(evel1);
  */

  //}
  /*}
  stage1--;*/
  /*}
  }
  }
  stage++;
  //}
  }
  SmallVector<Node *, 2> list_vect = Vectorization::createVectorizationCandidates(bestEval, &context);
    bestEval->setChildrenNodes(list_vect);

    for (auto node3 : list_vect)
    {
      std::string evel2 = evaluator.evaluateTransformation(node3);
      node3->setEvaluation(evel2);
       if (std::stod(bestEval->getEvaluation()) > std::stod(evel2))
    {
      std::cout << "Changing nodes\n";
      bestEval = node3;
    }
    }*/
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