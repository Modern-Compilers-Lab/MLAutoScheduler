//===------------ VectorizationTransformation.cpp VectorizationTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the VectorizationTransformation class, which
/// contains the declartion of the Vectorization transformation
///
//===----------------------------------------------------------------------===//
#include "VectorizationTransformation.h"

using namespace mlir;

mlir::Operation *DecomposeConv2dOp(mlir::Operation *Target)
{

  std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n   %conv = transform.structured.match ops{[\"linalg.conv_2d_nhwc_hwcf\"]} in %variant_op : (!transform.any_op) -> !transform.any_op %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op }";

  mlir::PassManager pm((Target)->getName());

  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
  if (!mlir::failed(pm.run((Target))))
  {
    return Target;
  }
}

Vectorization::Vectorization(mlir::linalg::LinalgOp *op,
                             mlir::MLIRContext *context)
{
  this->op = op;
  this->context = context;
}

std::string Vectorization::getType()
{
  return "Vectorization";
}

std::string Vectorization::printTransformation()
{

  std::string result = "V( ";
  result += " )";

  return result;
}
void Vectorization::applyTransformation(CodeIR CodeIr)
{
}

SmallVector<Node *, 2> Vectorization::createVectorizationCandidates(Node *node,
                                                                    mlir::MLIRContext *context)
{
  // SmallVector<SmallVector<Node *, 2>> ChildNodesList;

  SmallVector<linalg::LinalgOp, 2> LinalgOps;
  SmallVector<MLIRCodeIR *, 2> CodeIRs;

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

  /*Operation *target = ((mlir::OwningOpRef<Operation *> *)(*CodeIr)
                           .getIr())
                          ->get();*/

  /*target->walk([&](Operation *op)
               {
    //(op)->dump();
    if (auto genricOp = dyn_cast<linalg::LinalgOp>(op)) {
        SmallVector<Node* , 2> ChildNodes;

        //for (const auto& candidate : tileCombinations){

          MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
          Node* ChildNode = new Node (ClonedCode);


          std::vector<Transformation*> TransList= node->getTransformationList();
          ChildNode->setTransformationList(TransList);
          std::cout << "VECT1\n";
          Vectorization *vectorization  =
            new Vectorization(&genricOp,
                            //candidate,
                            context);

          ChildNode->setTransformation(vectorization);

          ChildNode->addTransformation(vectorization);

          ChildNodes.push_back(ChildNode);
        //}
        ChildNodesList.push_back(ChildNodes);
      } });*/
  SmallVector<Node *, 2> ChildNodes;

  MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
  Node *ChildNode = new Node(ClonedCode);

  std::vector<Transformation *> TransList = node->getTransformationList();
  ChildNode->setTransformationList(TransList);

  linalg::LinalgOp genricOp;
  Vectorization *vectorization =
      new Vectorization(&genricOp,
                        // candidate,
                        context);

  ChildNode->setTransformation(vectorization);

  ChildNode->addTransformation(vectorization);

  ChildNodes.push_back(ChildNode);

  // ChildNodesList.push_back(ChildNodes);
  //  int OpIndex = 0;

  bool ToDecompose = false;
  // for (auto ChildNodes : ChildNodesList)
  // {
  for (auto node : ChildNodes)
  {
    mlir::Operation *ClonedTarget = ((mlir::Operation *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                         .getIr());

    SmallVector<mlir::linalg::LinalgOp, 4> linalgOps = getLinalgOps(ClonedTarget);

    std::vector<Transformation *> transformations = node->getTransformationList();

    bool TilingDone = false;
    llvm::SmallVector<int64_t, 4> tilingSizes;
    for (Transformation *transform : transformations)
    {
      // Check if the dynamic cast to Tiling is successful
      if (transform->getType() == "Tiling")
      {
        TilingDone = true;
        // Found a Tiling transformation
        Tiling *tiling = (Tiling *)transform;
        int stage = tiling->getOperationStage();
        tilingSizes = tiling->getTilingSizes();
        mlir::Operation *linalgOp = linalgOps[stage];
        for (size_t i = 0; i < tilingSizes.size(); ++i)
        {
          if (i == 1)
          {
            tilingSizes[i] = 1;
          }
          else if (i == 4)
          {
            tilingSizes[i] = 1;
          }
        }
        scf::SCFTilingOptions options;

        options.setTileSizes(tilingSizes);
        std::cout << "Modified tilingSizes on stage " << stage << " : [";
        for (size_t i = 0; i < tilingSizes.size(); ++i)
        {
          std::cout << tilingSizes[i];
          if (i < tilingSizes.size() - 1)
          {
            std::cout << ", ";
          }
        }
        std::cout << "]\n";
        if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(linalgOp))
        {
          if ((linalgOp->getName().getStringRef()).str() != "linalg.fill" && TilingDone)
          {
            if ((linalgOp->getName().getStringRef()).str() == "linalg.conv_2d_nhwc_hwcf")
            {
              ToDecompose = true;
              IRRewriter rewriter(context);

              FailureOr<scf::SCFTilingResult> maybeTiled =
                  scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, options);

              if (!failed(maybeTiled))
                rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
            }
          }
        }
      }
     }

    /*ClonedTarget->walk([&](mlir::Operation *op)
                       {
            if (mlir::TilingInterface ClonedTileableOp
                              =dyn_cast<mlir::TilingInterface>(op)) {
                if ((op->getName().getStringRef()).str() != "linalg.fill" && TilingDone ){

                    if ((op->getName().getStringRef()).str() == "linalg.conv_2d_nhwc_hwcf"){
                      ToDecompose = true;
                      IRRewriter rewriter(context);

                      FailureOr<scf::SCFTilingResult> maybeTiled =
                              scf::tileUsingSCFForOp(rewriter, ClonedTileableOp, options);

                      if (!failed(maybeTiled))
                              rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
                    }
                }
              }});*/

    // int ClonedOpIndex = 0;
    //  Conv2d Decomposition
    if (ToDecompose)
    {
      mlir::Operation *DecomposedTarget = DecomposeConv2dOp(ClonedTarget);
      MLIRCodeIR *DecomposedCodeIr = (MLIRCodeIR *)CodeIr->setMLIRIR(DecomposedTarget);
      node->setTransformedCodeIr(DecomposedCodeIr);
    }

    // End Conv2d Decomposition

    mlir::Operation *Target = ((mlir::Operation *)(*((MLIRCodeIR *)node->getTransformedCodeIr()))
                                   .getIr());
    Vectorization *vectorization = (Vectorization *)node->getTransformation();

    // auto start = std::chrono::high_resolution_clock::now();
    std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %variant_op: (!transform.any_op) -> !transform.any_op \n %func_0 = transform.structured.vectorize %func {vectorize_padding}: (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 :(!transform.any_op) -> (!transform.any_op) \n transform.structured.hoist_redundant_tensor_subsets %func_01 :(!transform.any_op) -> () }";

    mlir::PassManager pm((Target)->getName());

    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
    if (!mlir::failed(pm.run((Target))))
    {
      MLIRCodeIR *ClonedCodeIr = (MLIRCodeIR *)CodeIr->setMLIRIR(Target);
      node->setTransformedCodeIr(ClonedCodeIr);
    }

    /*auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by vectorization: " << duration.count() << " microseconds" << std::endl;*/
  }
  // OpIndex++;
  //}
  /*SmallVector<Node *, 2> ResChildNodes;
  for (const auto &innerVector : ChildNodesList)
  {
    ResChildNodes.insert(ResChildNodes.end(), innerVector.begin(), innerVector.end());
  }
  return ResChildNodes;*/
  return ChildNodes;

  /*
   //std::string str1;
   //llvm::raw_string_ostream output(str1);
   MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();

   // TEMP : Transforming the code using the transform dialect interpreter; it uses a system call, and applies lowerings to the results of the vectorization
   (ClonedTarget)->print(output);

   std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %variant_op: (!transform.any_op) -> !transform.any_op \n %func_0 = transform.structured.vectorize %func {vectorize_padding}: (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 :(!transform.any_op) -> (!transform.any_op) \n transform.structured.hoist_redundant_tensor_subsets %func_01 :(!transform.any_op) -> () }";
   std::string transformedString = getVectorizedCode(str1, transformDialectString);
   if (transformedString == "process did not exit normally")
   {
   }

   // TEMP : The interpreter introduces an extra "module {}", removing it
   std::string cleanedString = removeVectExtraModuleTagCreated(transformedString);

   MLIRCodeIR *ClonedCodeIr = (MLIRCodeIR *)CodeIr->setMLIRIR(cleanedString, *((ClonedTarget)->getContext()));

   node->setTransformedCodeIr(ClonedCodeIr);*/

  /*llvm::SmallVector<Operation*, 2> linalgOps;
  ClonedTarget->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto genericOp = dyn_cast<linalg::LinalgOp>(op)) {
          linalgOps.push_back(op);
      }
  });
  std::reverse(linalgOps.begin(), linalgOps.end());
  for (Operation *genericOp : linalgOps) {
     IRRewriter rewriter(context);
    OpBuilder builder(context);

    llvm::ArrayRef<int64_t> emptyArrayRef;

    llvm::ArrayRef<bool> boolArrayRef;

    mlir::linalg::vectorize(rewriter, genericOp, emptyArrayRef,
      boolArrayRef , false);
}*/
  /*ClonedTarget->walk<WalkOrder::PreOrder>([&](Operation *op)
                     {
                      op->dump();
          //std::cout << "op = "<<(op->getName().getStringRef()).str()<<std::endl;


          if (auto genricOp = dyn_cast<linalg::LinalgOp>(op)) {  // IT APPLYS VECTORIZATION ON ALL THE CODE, WITHOUT THE NEXT CONDITION
            if ((op->getName().getStringRef()).str() != "linalg.fill"){
              //if (ClonedOpIndex == OpIndex){
                IRRewriter rewriter(context);
                OpBuilder builder(context);

                llvm::ArrayRef<int64_t> emptyArrayRef;

                llvm::ArrayRef<bool> boolArrayRef;

                mlir::linalg::vectorize(rewriter, genricOp, emptyArrayRef,
                 boolArrayRef , false);

                //
              }

         //  }
            ClonedOpIndex++;} });*/
}

/*
std::string removeVectExtraModuleTagCreated(std::string input) // TODO: Figure out why Transform Dialect Interpreter introduces an extra module
{
  std::string s = "module {";
  std::string s1 = "}";
  std::string::size_type i = input.find(s);
  if (i != std::string::npos)
    input.erase(i, s.length());
  std::string::size_type i1 = input.rfind(s1);
  if (i1 != std::string::npos)
    input.erase(i1, s1.length());
  return input;
}
pid_t popenvect(const char *command, int *infp, int *outfp)
{
  int p_stdin[2], p_stdout[2];
  pid_t pid;

  if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0)
    return -1;

  pid = fork();

  if (pid < 0)
  {
    close(p_stdin[READ]);
    close(p_stdin[WRITE]);
    close(p_stdout[READ]);
    close(p_stdout[WRITE]);

    return pid;
  }
  else if (pid == 0)
  {
    close(p_stdin[WRITE]);
    dup2(p_stdin[READ], READ);
    close(p_stdout[READ]);
    dup2(p_stdout[WRITE], WRITE);
    dup2(p_stdout[WRITE], STDERR_FILENO);
    if (std::getenv("LLVM_PATH") != nullptr)
    {
      std::string llvm_path = std::getenv("LLVM_PATH");
      std::string opt = llvm_path + "/build/bin/mlir-opt";
      execl(opt.c_str(),
            "mlir-opt", "--test-transform-dialect-interpreter", "--test-transform-dialect-erase-schedule",
            NULL);
    }

    perror("execl");
    exit(1);
  }

  // Parent process
  close(p_stdin[READ]);
  close(p_stdout[WRITE]);
  if (infp == NULL)
    close(p_stdin[WRITE]);
  else
    *infp = p_stdin[WRITE];

  if (outfp == NULL)
    close(p_stdout[READ]);
  else
    *outfp = p_stdout[READ];

  return pid;
}
std::string getVectorizedCode(std::string inputCode, std::string transfromDialectString)
{
  int in_fd, out_fd;
  pid_t pid;

  // std::string str2 = str1 + "transform.sequence failures(propagate) {^bb0(%arg1: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %arg1: (!transform.any_op) -> !transform.any_op \n  %func_0 = transform.structured.vectorize %func {vectorize_padding} : (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 : (!transform.any_op) -> (!transform.any_op) \n  transform.structured.hoist_redundant_tensor_subsets %func_01 : (!transform.any_op) -> ()}";
  std::string str = inputCode + transfromDialectString;

  //  Call popen2 to execute the command and get the input and output file descriptors
  pid = popenvect("", &in_fd, &out_fd);

  if (pid < 0)
  {
    perror("Failed to execute command");
    exit(EXIT_FAILURE);
  }
  // Measure the start time
  write(in_fd, str.c_str(), str.size());

  close(in_fd);
  // Read the output of the executed command
  const int max_output_size = INT_MAX;
  std::vector<char> output_data(max_output_size); // Using a dynamic buffer

  ssize_t total_bytes_read = 0;

  while (true)
  {
    ssize_t bytes_read = read(out_fd, output_data.data() + total_bytes_read, output_data.size() - total_bytes_read);

    if (bytes_read > 0)
    {
      total_bytes_read += bytes_read;

      // Check if the buffer is full (you can adjust this condition based on your needs)
      if (total_bytes_read == max_output_size)
      {
        break; // Exit the loop to avoid buffer overflow
      }
    }
    else if (bytes_read == 0)
    {
      // No more data available to read
      break;
    }
    else
    {
      // Error occurred while reading
      perror("Error while reading output");
      break;
    }
  }

  close(out_fd); // Close the output file descriptor

  // Wait for the child process to finish
  int status;
  waitpid(pid, &status, 0);

  // Check if the child process exited normally
  if (WIFEXITED(status))
  {
    int exit_status = WEXITSTATUS(status);
    printf("OPT Child process exited with status: %d\n", exit_status);
    return output_data.data();
  }
  else
  {
    printf("OPT process did not exit normally.\n");
    return "process did not exit normally";
  }
}*/