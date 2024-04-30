
//===----------------------- VectorizationTransformation.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the VectorizationTransformation class, which  
/// contains the declartion of the Vectorization transformation
///
//===----------------------------------------------------------------------===//

#ifndef MLSCEDULER_VECTORIZATION_TRANSFORMATION_H_
#define MLSCEDULER_VECTORIZATION_TRANSFORMATION_H_

#include "Transformation.h"
#include "MLIRCodeIR.h"
#include "Node.h"
#include "TilingTransformation.h"
#include "ParallelizationTransformation.h"
#include "TransformDialectInterpreter.h"
#include "TransformInterpreterPassBase.h"
#include "Utils.h"

#include <iostream>
#include <random>

class Vectorization: public Transformation{
    private:
        mlir::linalg::LinalgOp * op;
        mlir::MLIRContext *context;

    public:
        Vectorization();

        /// Constructor for Tiling that allows specifying the tile size.
        Vectorization(mlir::linalg::LinalgOp * op, /*llvm::SmallVector<int64_t, 4> tileSizes,*/ mlir::MLIRContext *context);

        /// Applies the tiling transformation to the given CodeIR object.
        /// Overrides the applyTransformation() method from the base class Transformation.
        void applyTransformation(CodeIR CodeIr) override;
        std::string printTransformation() override;
        std::string getType() override;
        /// Creates a list of tiling transformation candidates for the given CodeIR object.
        /// Overrides the createCandidates() method from the base class Transformation.
        static SmallVector<Node* , 2>  createVectorizationCandidates(Node *node, mlir::MLIRContext *context);

};

#endif // MLSCHEDULER_VECTORIZATION_TRANSFORMATION_H_