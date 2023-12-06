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

#include "Node.h"
#pragma once

using namespace mlir;
class SearchMethod {
    private:
        
    public:
        virtual Node * runSearchMethod(Node * root) = 0;
};