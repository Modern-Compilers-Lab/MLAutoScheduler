//===----------------------- BeamSearch.h ---------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the BeamSearch class, which  
/// contains a definition of the beam search method 
///
//===----------------------------------------------------------------------===//

#include "SearchMethod.h"
#include "Node.h"
#include "EvaluationByExecution.h"
#include "TilingTransformation.h"
#include "InterchangeTransformation.h"
#include "ParallelizationTransformation.h"
#include "VectorizationTransformation.h"

#include <queue>
#pragma once

using namespace mlir;
class BeamSearch : public SearchMethod{
    private:
        int beamSize;
        mlir::MLIRContext *context;
        std::string functionName;

    public:
        /// Constructor for the BeamSearch class, initializing beam size, MLIR context, and the function name.
        BeamSearch(int beamSize, mlir::MLIRContext *context, std::string functionName);
        /// Runs the beam search algorithm starting from a given root node
        Node * runSearchMethod(Node * root) override;

};