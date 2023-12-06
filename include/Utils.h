
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

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