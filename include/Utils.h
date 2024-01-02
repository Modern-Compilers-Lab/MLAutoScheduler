#ifndef MLSCEDULER_UTILS_H_
#define MLSCEDULER_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <random>

void generateCombinations(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                          int64_t maxNumberLoops,
                          int64_t currentLoop,
                          llvm::SmallVector<int64_t, 4> &currentCombination,
                          std::vector<llvm::SmallVector<int64_t, 4>> &combinations);

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForAllOpCombinations(int64_t maxNumberLoops,
                                 const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &possibleTileSizes,
                                 const llvm::SmallVector<int64_t> &upperBounds);

void generateCombinations(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                          int64_t maxNumberLoops,
                          int64_t currentLoop,
                          llvm::SmallVector<int64_t, 4> &currentCombination,
                          std::vector<llvm::SmallVector<int64_t, 4>> &combinations);

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForOpCombinations(int64_t maxNumberLoops,
                              const llvm::SmallVector<mlir::Range> &iterationDomain);

void generateCombinationsForDecompostion(const llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> &tileSizes,
                                         int64_t maxNumberLoops,
                                         int64_t currentLoop,
                                         llvm::SmallVector<int64_t, 4> &currentCombination,
                                         std::vector<llvm::SmallVector<int64_t, 4>> &combinations);

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>
generateTileForOpCombinationsForDecompostion(int64_t maxNumberLoops,
                                             const llvm::SmallVector<mlir::Range> &iterationDomain);


std::vector<std::vector<unsigned>> generateCandidates(int64_t numLoops,
        int64_t NbElement);

void generateCandidateHelper(std::vector<unsigned> &values,
                             std::vector<unsigned> &currentCandidate,
                             std::vector<std::vector<unsigned>> &candidates,
                             unsigned index);

#endif // MLSCHEDULER_UTILS_H_
