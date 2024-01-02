
//===----------------------- InterchangeTransformation.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the InterchangeTransformation class, which  
/// contains the declartion of the tiling transformation
///
//===----------------------------------------------------------------------===//

#ifndef MLSCEDULER_INTERCHANGE_TRANSFORMATION_H_
#define MLSCEDULER_INTERCHANGE_TRANSFORMATION_H_

#include "Transformation.h"
#include "MLIRCodeIR.h"
#include "Node.h"
#include "Utils.h"

#include "mlir/Dialect/Linalg/Passes.h"

#include <iostream>
#include <random>

class Interchange: public Transformation{
    private:
    mlir::linalg::LinalgOp* op;
    mlir::MLIRContext *context;
    std::vector<unsigned>InterchangeVector;

    public:
        Interchange();

        /// Constructor for Tiling that allows specifying the tile size.
        Interchange(linalg::LinalgOp *op, std::vector<unsigned> InterchangeVector, mlir::MLIRContext *context);

        /// Applies the tiling transformation to the given CodeIR object.
        /// Overrides the applyTransformation() method from the base class Transformation.
        void applyTransformation(CodeIR CodeIr) override;
        std::string printTransformation() override;
        std::string getType() override;

        std::vector<unsigned> getInterchangeVector();

        /// Creates a list of tiling transformation candidates for the given CodeIR object.
        /// Overrides the createCandidates() method from the base class Transformation.
        static SmallVector<Node* , 2>  createInterchangeCandidates(Node* node, mlir::MLIRContext *context);
};

#endif // MLSCEDULER_INTERCHANGE_TRANSFORMATION_H_