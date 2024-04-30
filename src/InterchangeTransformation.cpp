//===------------ InterchangeTransformation.cpp InterchangeTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the InterchangeTransformation class, which
/// contains the declartion of the Interchange transformation
///
//===----------------------------------------------------------------------===//
#include "InterchangeTransformation.h"

using namespace mlir;

Interchange::Interchange(linalg::LinalgOp *op,
                         std::vector<unsigned> InterchangeVector,
                         mlir::MLIRContext *context)
{
  // linalg::LinalgOp ClonedOp = op->clone();
  this->op = op;
  this->InterchangeVector = InterchangeVector;
  this->context = context;
}

std::string Interchange::getType()
{
  return "Interchange";
}

std::string Interchange::printTransformation()
{

  std::string result = "I( ";

  // Iterate over the elements of the vector and append them to the string
  for (size_t i = 0; i < InterchangeVector.size(); ++i)
  {
    result += std::to_string(InterchangeVector[i]);

    if (i != InterchangeVector.size() - 1)
    {
      result += ", ";
    }
  }
  result += " )";

  return result;
}
void Interchange::applyTransformation(CodeIR CodeIr)
{
  IRRewriter rewriter(this->context);
}

std::vector<unsigned> Interchange::getInterchangeVector()
{
  return InterchangeVector;
}

SmallVector<Node *, 2> Interchange::createInterchangeCandidates(
    Node *node,
    mlir::MLIRContext *context)
{
  // Initialize a list to store ChildNodes
  // SmallVector<SmallVector<Node *, 2>> ChildNodesList;
  SmallVector<Node* , 2> ChildNodes;

  // Get the target operation from the provided node's transformed code
  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
  Operation *target = ((Operation *)(*CodeIr)
                           .getIr());
  int counter = 0;

  // Traverse the operations in the target operation's hierarchy
  target->walk([&](Operation *op)
               {
    // Check if the operation is "linalg.generic"
    if (auto InterchangeableOp = dyn_cast<linalg::LinalgOp>(op)) {

      // TEMP : check if the operation is not 'linalg.fill' and counter is 3, targeting only the other operations 
      if ((op->getName().getStringRef()).str() != "linalg.fill" ){

        int64_t numLoops = InterchangeableOp.getNumLoops();
        // SmallVector<Node* , 2> ChildNodes;

        // Create a list of candidate values for interchange, with different parameters
        std::vector<std::vector<unsigned>> values = 
                generateCandidates(numLoops, 5);
                
        for (const auto& candidate : values){

          // Clone the code, create a new node, and set its transformation list
          MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
          Node* ChildNode = new Node (ClonedCode, node->getCurrentStage());        

          std::vector<Transformation*> TransList= node->getTransformationList();
          ChildNode->setTransformationList(TransList);

          // Create an interchange transformation and add it to the child node
          Interchange *interchange = 
            new Interchange(&InterchangeableOp,
                            candidate, 
                            context);

          ChildNode->setTransformation(interchange);
          ChildNode->addTransformation(interchange);

          // Add the child node to the list of child nodes
          ChildNodes.push_back(ChildNode);
        }
        // Add the list of child nodes to ChildNodesList
        // ChildNodesList.push_back(ChildNodes);
      } 
      counter++; 
    } });
  int OpIndex = 0;
  // for (auto ChildNodes : ChildNodesList)
  // {
    for (auto node : ChildNodes)
    {
      // Get the target operation from the child node's transformed code
      Operation *ClonedTarget = ((Operation *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                     .getIr());
      Interchange *inter = (Interchange *)node->getTransformation();

      std::vector<unsigned> candidate = inter->getInterchangeVector();
      ArrayRef<unsigned> interchangeVector(candidate);
      int ClonedOpIndex = 0;

      // Walk through operations in the cloned target operation
      ClonedTarget->walk([&](Operation *op)
                         {
        if (linalg::LinalgOp ClonedInterchangeableOp = 
                  dyn_cast<linalg::LinalgOp>(op)) {
             // TEMP: Check if the operation is not 'linalg.fill' and ClonedOpIndex is 3 
            if ((op->getName().getStringRef()).str() != "linalg.fill"  ){
                //auto start = std::chrono::high_resolution_clock::now();
                IRRewriter rewriter(context);
                rewriter.setInsertionPoint(ClonedInterchangeableOp);
                FailureOr<linalg::GenericOp> generalizeResult =
                    generalizeNamedOp(rewriter, ClonedInterchangeableOp);
                
                auto genericOp = *generalizeResult;

                // Perform interchange on the cloned operation
                FailureOr<linalg::GenericOp> interOp = 
                    linalg::interchangeGenericOp(rewriter,
                                                genericOp, 
                                                interchangeVector);
                /*auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "Time taken by Interchange: " << duration.count() << " microseconds" << std::endl;*/
               
            }
          ClonedOpIndex++;
          } });
    }
    OpIndex++;
  // }
  // Merge the child nodes into a single list and return it
  /*SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }

  return ResChildNodes;*/
  return ChildNodes;
}

