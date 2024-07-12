#ifndef MLSCEDULER_UTILS_H_
#define MLSCEDULER_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <set>
#include <random>

// Enum representing the mapping classification
enum class LinalgMappingClassification
{
  Unknown,
  OneToOne,
  OneToMany,
  ManyToSingle,
  ManyToMany,
};
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

std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> getLinalgOps(mlir::Operation *prog);
std::pair<std::vector<std::string>, std::vector<std::string>> remove_duplicate_args(std::vector<std::string> args, std::vector<std::string> shapes);
std::string function_wrapper(const std::string &operation, const std::string &maps = "");

llvm::SmallVector<mlir::OpFoldResult> getMixedSizes(llvm::ArrayRef<int64_t> tileSizes, mlir::MLIRContext *context);

mlir::LogicalResult TagSCFForAll(mlir::Operation *Target, std::string tag);
mlir::LogicalResult TagOperation(mlir::Operation *Target, std::string tag);

// Function to classify the mapping type of a linalg operation
LinalgMappingClassification classifyLinalgOp(mlir::Operation *op);
std::string getMappingTypeString(LinalgMappingClassification classification);
#endif // MLSCHEDULER_UTILS_H_
