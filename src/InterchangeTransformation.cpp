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

Interchange::Interchange(linalg::GenericOp *op,
                         std::vector<unsigned> InterchangeVector,
                         mlir::MLIRContext *context)
{
  // linalg::LinalgOp ClonedOp = op->clone();
  this->op = op;
  this->InterchangeVector = InterchangeVector;
  this->context = context;
}

std::vector<std::vector<unsigned>> generateCandidates(int64_t numLoops,
                                                      int64_t NbElement);

void generateCandidateHelper(std::vector<unsigned> &values,
                             std::vector<unsigned> &currentCandidate,
                             std::vector<std::vector<unsigned>> &candidates,
                             unsigned index);

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

  mlir::OwningOpRef<Operation *> *module =
      (mlir::OwningOpRef<Operation *> *)CodeIr.getIr();
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

  SmallVector<SmallVector<Node *, 2>> ChildNodesList;
  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  Operation *target = ((mlir::OwningOpRef<Operation *> *)(*CodeIr)
                           .getIr())
                          ->get();
  int counter = 0;
  std::vector<int64_t> ListNumLoops;
  target->walk([&](Operation *op)
               {
      if (auto InterchangeableOp = dyn_cast<linalg::GenericOp>(op)) {
          int64_t numLoops = InterchangeableOp.getNumLoops();
          ListNumLoops.push_back(numLoops);
          SmallVector<Node* , 2> ChildNodes;
          std::vector<std::vector<unsigned>> values = 
                  generateCandidates(numLoops, 5);
          for (const auto& candidate : values){

            MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
            Node* ChildNode = new Node (ClonedCode);        

            std::vector<Transformation*> TransList= node->getTransformationList();
            ChildNode->setTransformationList(TransList);

            Interchange *interchange = 
              new Interchange(&InterchangeableOp,
                              candidate, 
                              context);

            ChildNode->setTransformation(interchange); 
            
            ChildNode->addTransformation(interchange);
            
            ChildNodes.push_back(ChildNode);
          }
          ChildNodesList.push_back(ChildNodes);
          ListNumLoops.push_back(numLoops);;
        } });
  int OpIndex = 0;
  for (auto ChildNodes : ChildNodesList)
  {
    for (auto node : ChildNodes)
    {
      Operation *ClonedTarget = ((mlir::OwningOpRef<Operation *> *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                     .getIr())
                                    ->get();
      Interchange *inter = (Interchange *)node->getTransformation();

      std::vector<unsigned> candidate = inter->getInterchangeVector();
      ArrayRef<unsigned> interchangeVector(candidate);
      int ClonedOpIndex = 0;
      ClonedTarget->walk([&](Operation *op){
        if (linalg::GenericOp ClonedInterchangeableOp = 
                  dyn_cast<linalg::GenericOp>(op)) {
            if (ClonedOpIndex == OpIndex){
                IRRewriter rewriter(context);
                FailureOr<linalg::GenericOp> interOp = 
                  interchangeGenericOp(rewriter,
                                        ClonedInterchangeableOp, 
                                        interchangeVector);
            }
          ClonedOpIndex++;
          } });
    }
    OpIndex++;
  }

  SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }

  return ResChildNodes;
}

std::vector<std::vector<unsigned>> generateCandidates(int64_t numLoops,
                                                      int64_t NbElement)
{
  std::vector<std::vector<unsigned>> candidates;

  if (numLoops <= 0)
  {
    return candidates; // Return an empty vector if numLoops is invalid
  }

  std::vector<unsigned> values(numLoops);
  for (unsigned i = 0; i < numLoops; ++i)
  {
    values[i] = i;
  }

  std::vector<unsigned> currentCandidate(numLoops);
  generateCandidateHelper(values, currentCandidate, candidates, 0);
  std::vector<std::vector<unsigned>> out;
  std::sample(
      candidates.begin(),
      candidates.end(),
      std::back_inserter(out),
      candidates.size(),
      std::mt19937{std::random_device{}()});
  return out;
  // return candidates;
}

void generateCandidateHelper(std::vector<unsigned> &values,
                             std::vector<unsigned> &currentCandidate,
                             std::vector<std::vector<unsigned>> &candidates,
                             unsigned index)
{
  if (index == values.size())
  {
    // Add the completed candidate to the list
    candidates.push_back(currentCandidate);
    return;
  }

  for (unsigned i = 0; i < values.size(); ++i)
  {
    if (values[i] != UINT_MAX)
    {
      currentCandidate[index] = values[i];
      unsigned temp = values[i];
      values[i] = UINT_MAX; // Mark the value as used
      generateCandidateHelper(values, currentCandidate, candidates, index + 1);
      values[i] = temp; // Restore the value
    }
  }
}