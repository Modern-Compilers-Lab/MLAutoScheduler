//===----------- EvaluationByExecution.cpp - EvaluationByExecution  --------===//
//
///===----------------------------------------------------------------------===//
///
/// \file 
/// This file contains the implmentation of the EvaluationByExecution class, which  
/// contains an evaluator of the transformed code, the evaluator returns a the 
/// the reel execution time after running the code
///
//===----------------------------------------------------------------------===//

#include "EvaluationByExecution.h"

double EvaluationByExecution::evaluateTransformation(int argc, char** argv, DialectRegistry &registry, Node* node){

    llvm::raw_ostream &output = llvm::outs();
    FallbackAsmResourceMap fallbackResourceMap;
    
    
    mlir::OwningOpRef<Operation*>* module = (mlir::OwningOpRef<Operation*>*)(node->getTransformedCodeIr()->getIr());
    AsmState asmState((*module).get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                        &fallbackResourceMap);
    //(* module).get()->print(output, asmState);

    llvm::InitLLVM y(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    std::cout<<"Runner\n";
    mlir::JitRunnerMainForScheduler(argc, argv, registry, /*ChildNode->getTransformedCodeIr()*/( module));
    

    return 0.0;
}