
#include "Utils.h"

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
                              const llvm::SmallVector<mlir::Range> &iterationDomain)
{

  llvm::SmallVector<int64_t> upperBounds;
  for (const auto &range : llvm::enumerate(iterationDomain))
  {
    llvm::SmallVector<mlir::Value> dynamicVec;
    llvm::SmallVector<int64_t> staticVec;
    if (auto val = getConstantIntValue(range.value().size))
    {
      dispatchIndexOpFoldResult(range.value().size,
                                dynamicVec,
                                staticVec);
      upperBounds.append(staticVec.begin(), staticVec.end());
    }
    else
    {
      upperBounds.push_back(-1);
    }
  }
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> possibleTileSizes;
  for (int64_t value : upperBounds)
  {
    llvm::SmallVector<int64_t, 4> dividers;
    if (value == -1)
    {
      dividers.push_back(1);
    }
    for (int64_t i = 2; i <= value; ++i)
    {
      if (i < 50 && value % i == 0)
      {
        dividers.push_back(i);
      }
    }
    if (dividers.empty())
    {
      dividers.push_back(1);
    }
    possibleTileSizes.push_back(dividers);
  }

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

// Function to generate tiling sizes that are multiples of the upperBounds.
void generateForOpCombinationsForDecompostion(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
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
      generateForOpCombinationsForDecompostion(tileSizes,
                                               maxNumberLoops,
                                               currentLoop + 1,
                                               currentCombination,
                                               combinations,
                                               upperBounds);
    }
  }
}

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForOpCombinationsForDecompostion(int64_t maxNumberLoops,
                                             const llvm::SmallVector<mlir::Range> &iterationDomain)
{

  llvm::SmallVector<int64_t> upperBounds;
  for (const auto &range : llvm::enumerate(iterationDomain))
  {
    llvm::SmallVector<mlir::Value> dynamicVec;
    llvm::SmallVector<int64_t> staticVec;
    if (auto val = getConstantIntValue(range.value().size))
    {
      dispatchIndexOpFoldResult(range.value().size,
                                dynamicVec,
                                staticVec);
      upperBounds.append(staticVec.begin(), staticVec.end());
    }
    else
    {
      upperBounds.push_back(-1);
    }
  }

  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> possibleTileSizes;
  for (int64_t value : upperBounds)
  {
    llvm::SmallVector<int64_t, 4> dividers;
    if (value == -1)
    {
      dividers.push_back(1);
    }
    // if (value ==)
    for (int64_t i = 2; i <= value; ++i)
    {
      if (i < 50 && value % i == 0)
      {
        dividers.push_back(i);
      }
    }
    possibleTileSizes.push_back(dividers);
  }

  llvm::SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<llvm::SmallVector<int64_t, 4>> combinations;

  generateForOpCombinationsForDecompostion(possibleTileSizes,
                                           maxNumberLoops,
                                           0,
                                           currentCombination,
                                           combinations,
                                           upperBounds);

  return llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                             combinations.end());
}

// Function to generate tiling sizes that are multiples of the upperBounds.
void generateForAllOpCombinations(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                                  int64_t maxNumberLoops,
                                  int64_t currentLoop,
                                  llvm::SmallVector<int64_t, 4> &currentCombination,
                                  std::vector<llvm::SmallVector<int64_t, 4>> &combinations,
                                  const llvm::SmallVector<int64_t> &upperBounds)
{
  if (currentLoop >= maxNumberLoops)
  {
    if (!std::all_of(currentCombination.begin(), currentCombination.end(), [](int64_t size)
                     { return size == 1; }))
    {
      combinations.push_back(currentCombination);
    }

    return;
  }
  llvm::SmallVector<int64_t, 4> currentTileSizes = tileSizes[currentLoop];

  for (int64_t tileSize : currentTileSizes)
  {
    // Check if the current tileSize is a multiple of the corresponding upperBound.
    if (upperBounds[currentLoop] % tileSize == 0)
    {

      currentCombination[currentLoop] = tileSize;
      generateForAllOpCombinations(tileSizes,
                                   maxNumberLoops,
                                   currentLoop + 1,
                                   currentCombination,
                                   combinations,
                                   upperBounds);
    }
  }
}

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForAllOpCombinations(int64_t maxNumberLoops,
                                 const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &possibleTileSizes,
                                 const llvm::SmallVector<int64_t> &upperBounds)
{

  llvm::SmallVector<int64_t, 4> currentCombination(maxNumberLoops);
  std::vector<llvm::SmallVector<int64_t, 4>> combinations;

  generateForAllOpCombinations(possibleTileSizes,
                               maxNumberLoops,
                               0,
                               currentCombination,
                               combinations,
                               upperBounds);

  return llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>(combinations.begin(),
                                                             combinations.end());
}

std::vector<std::vector<unsigned>> generateCandidates(int64_t numLoops,
                                                      int64_t NbElement)
{
  std::vector<std::vector<unsigned>> candidates;

  if (numLoops <= 0)
  {
    return candidates; // Return an empty vector if numLoops is invalid
  }

  std::vector<unsigned> values(numLoops);
  for (unsigned i = 0; i < numLoops; ++i)
  {
    values[i] = i;
  }

  std::vector<unsigned> currentCandidate(numLoops);
  generateCandidateHelper(values, currentCandidate, candidates, 0);
  std::vector<std::vector<unsigned>> out;
  std::sample(
      candidates.begin(),
      candidates.end(),
      std::back_inserter(out),
      1,
      std::mt19937{std::random_device{}()});
  return out;
  // return candidates;
}

void generateCandidateHelper(std::vector<unsigned> &values,
                             std::vector<unsigned> &currentCandidate,
                             std::vector<std::vector<unsigned>> &candidates,
                             unsigned index)
{
  if (index == values.size())
  {
    // Add the completed candidate to the list
    candidates.push_back(currentCandidate);
    return;
  }

  for (unsigned i = 0; i < values.size(); ++i)
  {
    if (values[i] != UINT_MAX)
    {
      currentCandidate[index] = values[i];
      unsigned temp = values[i];
      values[i] = UINT_MAX; // Mark the value as used
      generateCandidateHelper(values, currentCandidate, candidates, index + 1);
      values[i] = temp; // Restore the value
    }
  }
}

std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>>  getLinalgOps(mlir::Operation *prog)
{

  std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOps;
  int i = 0;
  prog->walk([&](mlir::linalg::LinalgOp op)
             {
                 // TODO: support multi-results.
                //if ((op->getName().getStringRef()).str() == "linalg.matmul" || (op->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw"){
                  if (op->getNumResults() <= 1)
                  {
                        LinalgMappingClassification classification =  classifyLinalgOp(op);
                        std::string key = "operation" + std::to_string(i);
                        linalgOps[key] = std::make_pair(op, classification);
                        i++;
                  //} 
                  } });
  //std::reverse(linalgOps.begin(), linalgOps.end());
  /*llvm::SmallVector<std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>, 4> sublist;

  // Define your condition on LinalgMappingClassification
  auto condition = [](const LinalgMappingClassification& mapping) {
    // Replace this with your actual condition
    return mapping == LinalgMappingClassification::OneToOne; // Example condition
  };

  // Iterate through the original list
  for (const auto& element : linalgOps) {
    if (condition(element.second)) { // Access the second element (LinalgMappingClassification)
      sublist.push_back(element);
    }
  }*/
  return linalgOps;
}

std::pair<std::vector<std::string>, std::vector<std::string>> remove_duplicate_args(std::vector<std::string> args, std::vector<std::string> shapes)
{
  std::vector<std::pair<std::string, std::string>> args_shapes;
  std::set<std::pair<std::string, std::string>> seen;
  std::vector<std::pair<std::string, std::string>> result;

  for (size_t i = 0; i < args.size(); ++i)
  {
    std::pair<std::string, std::string> item(args[i], shapes[i]);
    if (seen.find(item) == seen.end())
    {
      seen.insert(item);
      result.push_back(item);
    }
  }

  std::vector<std::string> result_args;
  std::vector<std::string> result_shapes;
  for (const auto &item : result)
  {
    result_args.push_back(item.first);
    result_shapes.push_back(item.second);
  }

  return {result_args, result_shapes};
}

std::string function_wrapper(const std::string &operation, const std::string &maps)
{
  std::regex ins_outs_pattern("(?:ins|outs)\\s*\\(([^())]+)\\)");
  std::smatch match;
  std::vector<std::string> args, shapes;

  auto operation_iter = operation.cbegin();
  while (std::regex_search(operation_iter, operation.cend(), match, ins_outs_pattern))
  {
    std::string field = match[1];
    size_t pos = field.find(':');
    if (pos != std::string::npos)
    {
      std::string args_field = field.substr(0, pos);
      std::string shapes_field = field.substr(pos + 1);

      size_t comma_pos;
      while ((comma_pos = args_field.find(',')) != std::string::npos)
      {
        args.push_back(args_field.substr(0, comma_pos));
        args_field.erase(0, comma_pos + 1);
      }
      args.push_back(args_field);

      while ((comma_pos = shapes_field.find(',')) != std::string::npos)
      {
        shapes.push_back(shapes_field.substr(0, comma_pos));
        shapes_field.erase(0, comma_pos + 1);
      }
      shapes.push_back(shapes_field);
    }
    operation_iter = match.suffix().first;
  }

  auto [unique_args, unique_shapes] = remove_duplicate_args(args, shapes);

  std::string args_str;
  for (size_t i = 0; i < unique_args.size(); ++i)
  {
    args_str += unique_args[i] + ": " + unique_shapes[i];
    if (i != unique_args.size() - 1)
      args_str += ", ";
  }

  std::string out_shape = unique_shapes.back();

  std::string wrapped_operation;
  if (maps.empty())
  {
    wrapped_operation = "func.func private @func_call(" + args_str + ") -> " + out_shape + " {\n" +
                        "  %ret = " + operation + "\n" +
                        "  return %ret : " + out_shape + "\n" +
                        "}";
  }
  else
  {
    wrapped_operation = maps + "\n" +
                        "func.func private @func_call(" + args_str + ") -> " + out_shape + " {\n" +
                        "  %ret = " + operation + "\n" +
                        "  return %ret : " + out_shape + "\n" +
                        "}";
  }

  return wrapped_operation;
}

llvm::SmallVector<mlir::OpFoldResult> getMixedSizes(llvm::ArrayRef<int64_t> tileSizes, mlir::MLIRContext *context)
{

  llvm::SmallVector<mlir::OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  mlir::Builder builder(context);
  for (int64_t size : tileSizes)
  {
    if (size == mlir::ShapedType::kDynamic)
    {
      // results.push_back(dynamic[dynamicPos++]);
    }
    else
    {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

mlir::LogicalResult TagSCFForAll(mlir::Operation *Target, std::string tag)
{
  std::string transformDialectString = "module attributes {transform.with_named_sequence} { \n transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})  { \n  %1 = transform.structured.match ops{[\"scf.forall\"]}  in %variant_op : (!transform.any_op) -> !transform.any_op transform.annotate %1 \"" + tag + "\" : !transform.any_op transform.yield}}";
  mlir::transform::TransformOptions options1;
  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile = mlir::parseSourceString<mlir::ModuleOp>(transformDialectString, Target->getContext());
  llvm::StringRef entryPoint = "__transform_main";
  mlir::Operation *transformEntryPoint = mlir::transform::detail::findTransformEntryPoint(Target, *moduleFromFile, entryPoint);

  return mlir::transform::applyTransformNamedSequence(
      Target, transformEntryPoint, *moduleFromFile,
      options1.enableExpensiveChecks(false));
}
mlir::LogicalResult TagOperation(mlir::Operation *Target, std::string tag)
{
  std::string transformDialectString = "module attributes {transform.with_named_sequence} { \n transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})  { \n  %1 = transform.structured.match interface{LinalgOp}  in %variant_op : (!transform.any_op) -> !transform.any_op transform.annotate %1 \"" + tag + "\" : !transform.any_op transform.yield}}";
  mlir::transform::TransformOptions options1;
  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile = mlir::parseSourceString<mlir::ModuleOp>(transformDialectString, Target->getContext());
  llvm::StringRef entryPoint = "__transform_main";
  mlir::Operation *transformEntryPoint = mlir::transform::detail::findTransformEntryPoint(Target, *moduleFromFile, entryPoint);

  return mlir::transform::applyTransformNamedSequence(
      Target, transformEntryPoint, *moduleFromFile,
      options1.enableExpensiveChecks(false));
}

// Function to classify the mapping type of a linalg operation
LinalgMappingClassification classifyLinalgOp(mlir::Operation *op)
{
  bool hasManyToOne = false;
  bool hasOneToMany = false;
  // Cast the operation to a linalg::GenericOp
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(op);
  if (!linalgOp)
  {
    return LinalgMappingClassification::Unknown; // Handle non-linalg operation
  }
  else
  {
    // Check for number of operands and results
    int numInputs = op->getNumOperands() - 1;
    int numOutputs = op->getNumResults();

    // One-to-One mapping
    if (numInputs == numOutputs)
    {

      bool allEqual = true;
      for (auto result : op->getResults())
      {

        if (auto resultShape = llvm::dyn_cast<mlir::RankedTensorType>(result.getType()))
        {

          for (auto operand : op->getOperands())
          {

            if (auto operandShape = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType()))
            {
              if (resultShape.getShape() != operandShape.getShape())
              {
                allEqual = false;
                break;
              }
            }
          }
          if (!allEqual)
          {
            break;
          }
        }
      }
      if (allEqual)
      {
        return LinalgMappingClassification::OneToOne;
      }
    }
    // Check for Many-to-One and One-to-Many based on input/output dimensions
    hasManyToOne = false;
    hasOneToMany = false;
    for (auto result : op->getResults())
    {
      if (auto rankedTensor = llvm::dyn_cast<mlir::RankedTensorType>(result.getType()))
      {
        int outputDimSize = rankedTensor.getShape().size();
        for (auto operand : op->getOperands())
        {
          if (auto rankedTensorOperand = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType()))
          {
            if (rankedTensorOperand.getShape().size() != outputDimSize)
            {
              hasOneToMany = true;
            }
            else if (hasManyToOne)
            {
              // Multiple inputs with different mappings -> Many-to-Many
              return LinalgMappingClassification::ManyToMany;
            }
            else
            {
              hasManyToOne = (numInputs > 1); // Potential Many-to-One
            }
          }
        }
      }
    }
  }

  // Decide based on complexity
  return hasOneToMany ? LinalgMappingClassification::OneToMany : LinalgMappingClassification::ManyToSingle;
}
std::string getMappingTypeString(LinalgMappingClassification classification)
{
  switch (classification)
  {
  case LinalgMappingClassification::Unknown:
    return "Unknown";
  case LinalgMappingClassification::OneToOne:
    return "One-to-One";
  case LinalgMappingClassification::OneToMany:
    return "One-to-Many";
  case LinalgMappingClassification::ManyToSingle:
    return "Many-to-One";
  case LinalgMappingClassification::ManyToMany:
    return "Many-to-Many";
  }
}