
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
      10,
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
