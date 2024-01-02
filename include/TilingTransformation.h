
//===----------------------- TilingTransformation.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the TilingTransformation class, which  
/// contains the declartion of the tiling transformation
///
//===----------------------------------------------------------------------===//
#ifndef MLSCEDULER_TILING_TRANSFORMATION_H_
#define MLSCEDULER_TILING_TRANSFORMATION_H_

#include "Transformation.h"
#include "MLIRCodeIR.h"
#include "Node.h"
#include "Utils.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <iostream>
#include <random>

class Tiling: public Transformation{
    private:
        mlir::TilingInterface* op;
        mlir::scf::SCFTilingOptions options;
        mlir::MLIRContext *context;
        llvm::SmallVector<int64_t, 4> tileSizes;
    public:
        Tiling();

        /// Constructor for Tiling that allows specifying the tile size.
        Tiling(mlir::TilingInterface* op, mlir::scf::SCFTilingOptions &options, llvm::SmallVector<int64_t, 4> tileSizes, mlir::MLIRContext *context);

        /// Applies the tiling transformation to the given CodeIR object.
        /// Overrides the applyTransformation() method from the base class Transformation.
        void applyTransformation(CodeIR CodeIr) override;
        std::string printTransformation() override;

        std::string getType() override;
        llvm::SmallVector<int64_t, 4> getTilingSizes();
        /// Creates a list of tiling transformation candidates for the given CodeIR object.
        /// Overrides the createCandidates() method from the base class Transformation.
        static SmallVector<Node* , 2>  createTilingCandidates(Node *node, mlir::MLIRContext *context);

        mlir::scf::SCFTilingOptions getOptions();
};

#endif // MLSCEDULER_TILING_TRANSFORMATION_H_