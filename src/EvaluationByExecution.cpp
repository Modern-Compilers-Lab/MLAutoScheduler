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

//#include "/scratch/ia2280/mlir_autoScheduler/MLScheduler/src/Transform/TransformDialectInterpreter.cpp"
#pragma once
using namespace mlir;
std::string getTransformedCode(std::string inputCode, std::string transfromDialectString);
std::string getEvaluation(std::string inputCode);
std::string removeExtraModuleTagCreated(std::string input);
pid_t popen2(const char *command, int *infp, int *outfp);
pid_t popen22(const char *command, int *infp, int *outfp);

EvaluationByExecution::EvaluationByExecution()
{
}
EvaluationByExecution::EvaluationByExecution(std::string LogsFileName)
{
  this->LogsFileName = LogsFileName;
}
std::string EvaluationByExecution::evaluateTransformation(Node *node)
{
    std::string str1;
    llvm::raw_string_ostream output(str1);

    MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
    MLIRCodeIR* ClonedCode =  (MLIRCodeIR*)CodeIr->cloneIr();
    
    Operation *ClonedTarget = ((Operation *)(*node->getTransformedCodeIr()).getIr());
    Operation *op = ((Operation *)(*(ClonedCode))
                                     .getIr());
    // Printing the transformed code
    if (std::getenv("AS_VERBOSE") != nullptr)
    {
        int asVerbose = std::stoi(std::getenv("AS_VERBOSE"));
        if (asVerbose == 1)
        {
            std::ofstream debugFile;
            /*time_t now = time(0);
            char* dt = ctime(&now); // Assuming you have this correctly defined elsewhere

            std::string dtString(dt); // Convert char* to std::string

            std::string logs = "logs_" + dtString + ".txt";*/
            debugFile.open(LogsFileName, std::ios_base::app);
            if (debugFile.is_open())
            {
                std::string str;
                llvm::raw_string_ostream debugOut(str);
                if (node->getTransformation() != NULL)
                {
                    debugFile << "###################################" << std::endl;
                    debugFile << "Transformtion : " << std::endl;
                    for (const auto &transformation : node->getTransformationList())
                    {
                        debugFile << transformation->printTransformation();
                    }
                    debugFile << std::endl;
                }
                op->print(debugOut);

                debugFile << str << std::endl;
                debugFile.close();
            }
        }
    }
    /*mlir::PassManager pmBefore((*op).get()->getName());

    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pmBefore);

    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocs = true;
    options.bufferizeFunctionBoundaries = true;
    options.createDeallocs = true;
    options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);

    pmBefore.addPass(mlir::createLoopInvariantCodeMotionPass());
    pmBefore.addPass(mlir::createCSEPass());
    pmBefore.addPass(mlir::createCanonicalizerPass());
    pmBefore.addPass(mlir::createCSEPass());
    pmBefore.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    mlir::OpPassManager &optPMBefore = pmBefore.nest<mlir::func::FuncOp>();
    optPMBefore.addPass(mlir::bufferization::createBufferDeallocationPass());
    optPMBefore.addPass(mlir::bufferization::createFinalizingBufferizePass());
    pmBefore.addPass(mlir::createBufferizationToMemRefPass());
    optPMBefore.addPass(mlir::bufferization::createBufferDeallocationPass());
    if (!mlir::failed(pmBefore.run(*(*op))))*/

    // TEMP : Transforming the code using the transform dialect interpreter; it uses a system call, and applies lowerings to the results of the vectorization
    //op->print(output);
   

    /*std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n %f = transform.structured.match ops{[\"func.func\"]} in %variant_op : (!transform.any_op) -> !transform.any_op \n transform.apply_patterns to %f {  \n transform.apply_patterns.vector.lower_contraction lowering_strategy = \"outerproduct\" \n transform.apply_patterns.vector.transfer_permutation_patterns \n transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = \"innerparallel\" \n transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = \"vector-transfer\" \n transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true \n transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1 \n transform.apply_patterns.vector.lower_shape_cast \n transform.apply_patterns.vector.lower_transpose lowering_strategy = \"shuffle_1d\" \n transform.apply_patterns.canonicalization} \n : !transform.any_op}";
    std::string transformedString = getTransformedCode(str1, transformDialectString);
    if (transformedString == "process did not exit normally")
    {
        return "-1";
    }
    // TEMP : The interpreter introduces an extra "module {}", removing it
    std::string cleanedString = removeExtraModuleTagCreated(transformedString);
    //std::cout<<cleanedString<<std::endl;
    // Continue the lowerings of the code to llvm runnable code*/
    //std::string transformDialectString = "transform.sequence failures(propagate) { \n ^bb1(%variant_op: !transform.any_op): \n %named_conv = transform.structured.match ops{[\"linalg.matmul\"]} in %variant_op : (!transform.any_op) -> !transform.any_op \n %forall_l1, %conv_l1 = transform.structured.tile_to_forall_op %named_conv tile_sizes [10, 10]: (!transform.any_op) -> (!transform.any_op, !transform.any_op) }";

  
    //mlir::OwningOpRef<Operation *> module = parseSourceString(transformDialectString, (op)->getContext());
    //(*module)->dump();
    std::string outString;
    llvm::raw_string_ostream output_run(outString);
    //auto start = std::chrono::high_resolution_clock::now();
    mlir::PassManager pm((op)->getName());
    
    
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    
    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocs = true;
    options.bufferizeFunctionBoundaries = true;
    options.createDeallocs = true;
    options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
    // pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
    //pm.addPass(createTestTransformDialectEraseSchedulePass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    
    optPM.addPass(mlir::bufferization::createBufferDeallocationPass());
    //optPM.addPass(mlir::bufferization::createFinalizingBufferizePass());
    //pm.addPass(mlir::createBufferizationToMemRefPass());
    //optPM.addPass(mlir::bufferization::createBufferDeallocationPass());
    
    optPM.addPass(mlir::createConvertLinalgToLoopsPass());
    optPM.addPass(mlir::createForEachThreadLowering());
    pm.addPass(mlir::createConvertVectorToSCFPass());
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    pm.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createLowerAffinePass());
    optPM.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createLowerAffinePass());
    optPM.addPass(mlir::createArithToLLVMConversionPass());
 
    pm.addPass(createConvertOpenMPToLLVMPass());
     pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (!mlir::failed(pm.run((op))))
        (op)->print(output_run);
    /*auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);*/
    


    // Getting the evaluation uisng mlir-cpu-runner, the function uses a system call
    //auto start_eval = std::chrono::high_resolution_clock::now();
    std::string OutputData = getEvaluation(outString);
    /*auto end_eval = std::chrono::high_resolution_clock::now();
    auto duration_eval = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval);*/
    
    // Printing the evaluation 
    if (std::getenv("AS_VERBOSE") != nullptr)
    {
        int asVerbose = std::stoi(std::getenv("AS_VERBOSE"));
        if (asVerbose == 1)
        {
            std::ofstream debugFile;
            debugFile.open(LogsFileName, std::ios_base::app);
            if (debugFile.is_open())
            {
                if (node->getTransformation() != NULL)
                {
                    debugFile << OutputData << std::endl;
                    /*debugFile << "Time taken by Lowerings: " << duration.count() << " microseconds" << std::endl;
                    debugFile << "Time taken by Evaluation: " << duration_eval.count() << " microseconds" << std::endl;*/
                }
                debugFile.close();
            }
        }
    }
    return OutputData;
}



pid_t popen2(const char *command, int *infp, int *outfp)
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

        if (std::getenv("LLVM_PATH") != nullptr && std::getenv("SHARED_LIBS") != nullptr)
        {
            std::string llvm_path = std::getenv("LLVM_PATH");
            std::string runner = llvm_path + "/build/bin/mlir-cpu-runner";
            char *shared_libs = std::getenv("SHARED_LIBS");
            execl(runner.c_str(),
                  "mlir-cpu-runner", "-e", "main", "-entry-point-result=void",
                  "-shared-libs", shared_libs,
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
/*pid_t popen22(const char *command, int *infp, int *outfp)
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
}*/

std::string removeExtraModuleTagCreated(std::string input) // TODO: Figure out why Transform Dialect Interpreter introduces an extra module
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

/// Executes the evaluation command by creating a child process,
/// feeds it the input code, captures the command's output 
/// Returns the captured output as a string, optionally stripping
/// newline characters from the output.

std::string getEvaluation(std::string inputCode)
{

    std::string command = "";
    int in_fd, out_fd;
    pid_t pid;

    // Call popen2 to execute the command and get the input and output file descriptors
    pid = popen2(command.c_str(), &in_fd, &out_fd);

    if (pid < 0)
    {
        perror("Failed to execute command");
        exit(EXIT_FAILURE);
    }
    // Measure the start time

    write(in_fd, inputCode.c_str(), inputCode.size());

    close(in_fd);
    // Read the output of the executed command
    const int max_output_size = 4280;
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

    // Remove newline characters from the output data
    output_data.erase(std::remove(output_data.begin(), output_data.end(), '\n'), output_data.end());
    printf("Command output:\n%s\n", output_data.data());

    close(out_fd); // Close the output file descriptor

    // Wait for the child process to finish
    int status;
    waitpid(pid, &status, 0);

    // Check if the child process exited normally
    if (WIFEXITED(status))
    {
        int exit_status = WEXITSTATUS(status);
        printf("Cpu Runner Child process exited with status: %d\n", exit_status);
         
        std::string evalString = "";
        std::string data(output_data.begin(), output_data.end());

        size_t lastGFLOPSPos = data.rfind("GFLOPS"); // Find the position of the last "GFLOPS"
        if (lastGFLOPSPos != std::string::npos) {
            std::string substring = data.substr(0, lastGFLOPSPos-1); // Extract the substring before the last "GFLOPS"
            size_t spacePos = substring.rfind("GFLOPS"); // Find the position of the last space in the substring

            if (spacePos != std::string::npos) {
                evalString = substring.substr(spacePos + 6); // Extract the number string after the last space   
            }
        } else {
            std::cout << "No GFLOPS found in the input string." << std::endl;
        }
        std::cout<<evalString<<std::endl;

        return evalString;
    }
    else
    {
        printf("Cpu Runner Child process did not exit normally.\n");
        return "Cpu Runner Child process did not exit normally";
    }
}

/// Transforms the input code using the transform dialect string. 
/// The function uses the mlir-opt tool to run the transform dialect
/// interprter, returns the transformed code

/*std::string getTransformedCode(std::string inputCode, std::string transfromDialectString)
{
    int in_fd, out_fd;
    pid_t pid;

    // std::string str2 = str1 + "transform.sequence failures(propagate) {^bb0(%arg1: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %arg1: (!transform.any_op) -> !transform.any_op \n  %func_0 = transform.structured.vectorize %func {vectorize_padding} : (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 : (!transform.any_op) -> (!transform.any_op) \n  transform.structured.hoist_redundant_tensor_subsets %func_01 : (!transform.any_op) -> ()}";
    std::string str = inputCode + transfromDialectString;

    //  Call popen2 to execute the command and get the input and output file descriptors
    pid = popen22("", &in_fd, &out_fd);

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