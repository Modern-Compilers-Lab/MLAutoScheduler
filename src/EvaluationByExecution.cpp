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
#include <fstream>
using namespace mlir;
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

        //-shared-libs=/Users/lei/soft/llvm-project/build/lib/libmlir_runner_utils.dylib -shared-libs=/Users/lei/soft/llvm-project/build/lib/libmlir_c_runner_utils.dylib

        execl("/Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/llvm-project/build/bin/mlir-cpu-runner",
              "mlir-cpu-runner", "-e", "main", "-entry-point-result=void",
              "-shared-libs=/Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/llvm-project/build/lib/libmlir_runner_utils.dylib", 
              "-shared-libs=/Users/ericasare/Desktop/Desktop/Mac2023/School/Fall2023NewYork/Capstone/MLScheduler/llvm-project/build/lib/libmlir_c_runner_utils.dylib",
              NULL);

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

std::string EvaluationByExecution::evaluateTransformation(Node *node)
{

    std::string str1;
    llvm::raw_string_ostream output(str1);

    mlir::OwningOpRef<Operation *> *op = ((mlir::OwningOpRef<Operation *> *)(*node->getTransformedCodeIr()).getIr());
    mlir::PassManager pm((*op).get()->getName());
    // (* op)->dump();
    // return "0";
    if (std::getenv("AS_VERBOSE") != nullptr)
    {
        int asVerbose = std::stoi(std::getenv("AS_VERBOSE"));
        if (asVerbose == 1)
        {
            std::ofstream debugFile;
            debugFile.open("logs.txt", std::ios_base::app);
            if (debugFile.is_open())
            {
                std::string str;
                llvm::raw_string_ostream debugOut(str);
                if (node->getTransformation() != NULL)
                {
                    debugFile << "###########################################################################" << std::endl;
                    debugFile << "Transformtion : " << std::endl;
                    for (const auto &transformation : node->getTransformationList())
                    {
                        debugFile << transformation->printTransformation();
                    }
                    debugFile << std::endl;
                }
                (**op)->print(debugOut);

                debugFile << str << std::endl;
                debugFile.close();
            }
        }
    }
    // std::cout << str <<std::endl;
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocs = true;
    options.bufferizeFunctionBoundaries = true;
    options.createDeallocs = true;
    options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);

    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    // pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
    optPM.addPass(mlir::bufferization::createBufferDeallocationPass());
    optPM.addPass(mlir::bufferization::createFinalizingBufferizePass());
    pm.addPass(mlir::createBufferizationToMemRefPass());
    pm.addPass(mlir::createCanonicalizerPass());

    optPM.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createForEachThreadLowering());
    pm.addPass(mlir::createForLoopRangeFoldingPass());
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    optPM.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertVectorToSCFPass());
    optPM.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    optPM.addPass(mlir::createConvertSCFToCFPass());
    optPM.addPass(mlir::createLowerAffinePass());
    optPM.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (!mlir::failed(pm.run(*(*op))))
    
    //(**op)->dump();
    (**op)->print(output);
    std::string command = "llvm-project/mlir/tools/mlir-cpu-runner/mlir-cpu-runner.cpp";

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
    // auto start = std::chrono::high_resolution_clock::now();
    write(in_fd, str1.c_str(), str1.size());

    close(in_fd);
    // Read the output of the executed command
    const int max_output_size = 1280;
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
    // std::cout<< total_bytes_read;
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
        printf("Child process exited with status: %d\n", exit_status);
    }
    else
    {
        printf("Child process did not exit normally.\n");
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // std::cout<< std::to_string(elapsed.count())<< std::endl;

    // std::string s(output_data);
    // s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    // size_t position = s.find('U');

    // // Erase everything after 'U'
    // if (position != std::string::npos) {
    //     s.erase(position);
    // }
    return output_data.data(); 
}

