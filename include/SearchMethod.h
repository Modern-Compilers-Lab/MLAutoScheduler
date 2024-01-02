//===----------------------- SearchMethod.h -------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SearchMethod class, which  
/// contains an abstract definition of the methods to search for the best 
/// schedule
///
//===----------------------------------------------------------------------===//
#ifndef MLSCEDULER_SEARCH_METHOD_H_
#define MLSCEDULER_SEARCH_METHOD_H_

#include "Node.h"

using namespace mlir;
class SearchMethod {
    private:
        
    public:
        virtual Node * runSearchMethod(Node * root) = 0;
};

#endif // MLSCEDULER_SEARCH_METHOD_H_