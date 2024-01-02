//===----------------------- EvaluationByExecution.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the EvaluationByExecution class, which  
/// contains an evaluator of the transformed code, the evaluator returns a the 
/// the reel execution time after running the code
///
//===----------------------------------------------------------------------===//
#ifndef MLSCEDULER_EVALUATION_BY_EXECUTION_H_
#define MLSCEDULER_EVALUATION_BY_EXECUTION_H_

#include "Evaluation.h"
#include "Node.h"
#include "TransformDialectInterpreter.h"
#include "TransformInterpreterPassBase.h"
#include "CustomPasses/Passes.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "mlir/Parser/Parser.h"

#include <utility>
#include <chrono>
#include <iostream>

#include <stdio.h>
#include<sys/wait.h>
#include<unistd.h>

#include <fstream>
#include <ctime>

#define READ 0
#define WRITE 1

using namespace mlir;
class EvaluationByExecution {
    public:
        std::string LogsFileName;

        EvaluationByExecution();
        EvaluationByExecution(std::string LogsFileName);
        /// Evaluates the transformation by executing it with the given parameters.
        /// Parameters:
        /// - registry: A reference to the DialectRegistry used for execution.
        /// - node: A pointer to the Node object representing the transformation.
        /// Returns: The evaluation result as a double value.
        std::string evaluateTransformation(/*int argc, char** argv, DialectRegistry &registry,*/ Node* node);
};

#endif // MLSCEDULER_EVALUATION_BY_EXECUTION_H_