//===------------------------- BeamSearch.cpp - BeamSearch  ----------------===//
//
///===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the BeamSearch class, which contains
///  the implmentation of the beam search method
///
//===----------------------------------------------------------------------===//

#include "BeamSearch.h"

BeamSearch::BeamSearch(int beamSize, mlir::MLIRContext *context, std::string functionName)
{
    this->beamSize = beamSize;
    this->context = context;
    this->functionName = functionName;
}

Node *BeamSearch::runSearchMethod(Node *root)
{

    // Initialize the exploration queue and level counter
    std::queue<Node *> exploration_queue;
    exploration_queue.push(root);
    int level = 0;

    // Clone the root's MLIR code for evaluation
    MLIRCodeIR *CodeIr = (MLIRCodeIR *)root->getTransformedCodeIr();
    MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
    Node *clone = new Node(ClonedCode);
    Node *BestNode = clone;

    // Create an evaluator for transformation evaluations
    EvaluationByExecution evaluator = EvaluationByExecution(this->functionName + "_logs_best_beam_search_test_49x512x4608.txt");

    while (!exploration_queue.empty() && level != 4)
    {
        std::cout << "################# Level = " << level << " ###############\n";
        // SmallVector<Node *,2> parent_nodes;

        // Create a list to store schedule nodes at the current level
        SmallVector<Node *, 2> level_schedules;

        // Iterate through nodes in the exploration queue at the current level
        while (!exploration_queue.empty())
        {

            Node *node = exploration_queue.front();

            exploration_queue.pop();
            SmallVector<Node *, 2> candidates;

            // Generate transformation candidates based on the current level.
            switch (level)
            {
            case 0:
                candidates = Parallelization::createParallelizationCandidates(node, this->context);
                break;
            case 1:
            {
                candidates = Tiling::createTilingCandidates(node, this->context);

                // Insert the parent node as a candidate
                MLIRCodeIR *ToCloneCodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
                MLIRCodeIR *ClonedCode = (MLIRCodeIR *)ToCloneCodeIr->cloneIr();
                Node *ClonedNode = new Node(ClonedCode);

                std::vector<Transformation *> TransList = node->getTransformationList();
                ClonedNode->setTransformationList(TransList);

                candidates.insert(candidates.begin(), ClonedNode);

                // parent_nodes.insert(parent_nodes.begin(),ClonedNode );
                break;
            }

            case 2:
                candidates = Interchange::createInterchangeCandidates(node, this->context);
                break;
            case 3:
                candidates = Vectorization::createVectorizationCandidates(node, this->context);
                break;
            }
            // Evaluate each transformation candidate and store their evaluation results
            for (auto ChildNode : candidates)
            {
                std::string evel = evaluator.evaluateTransformation(ChildNode);
                ChildNode->setEvaluation(evel);
            }
            // Sort the candidates based on their evaluation scores
            std::sort(candidates.begin(), candidates.end(), [](Node *a, Node *b)
                      { return std::stod(a->getEvaluation()) < std::stod(b->getEvaluation()); });

            // Set the children nodes of the current node (for printing the tree)
            node->setChildrenNodes(candidates);
            // Save the best node at level 0 (the root node of the resulting tree)
            if (level == 0)
                BestNode = node;

            level_schedules.insert(level_schedules.end(), candidates.begin(), candidates.end());
        }

        // Sort the level's schedule nodes from smallest to largest evaluation
        std::sort(level_schedules.begin(), level_schedules.end(), [](Node *a, Node *b)
                  { return std::stod(a->getEvaluation()) < std::stod(b->getEvaluation()); });

        /* // Forcing beam search to take one of the parent nodes in the next level
        std::sort(parent_nodes.begin(), parent_nodes.end(), [](Node *a, Node *b) {
            return std::stod(a->getEvaluation()) < std::stod(b->getEvaluation());
        });
        parent_nodes.resize(std::min(1, (int)parent_nodes.size()));
        level_schedules.insert(level_schedules.begin(), parent_nodes.begin(), parent_nodes.end());*/

        // keep the top 'beam_size' children and delete the rest
        /*for (int i = this->beamSize; i < level_schedules.size(); ++i)
            delete level_schedules[i];*/
        level_schedules.resize(std::min(this->beamSize, (int)level_schedules.size()));

        // Add the level's schedule nodes to the exploration queue for the next level
        for (Node *child : level_schedules)
        {
            exploration_queue.push(child);
        }
        level++;
    }

    return BestNode;
}
