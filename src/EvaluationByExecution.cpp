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
pid_t
popen2(const char *command, int *infp, int *outfp)
{
    int p_stdin[2], p_stdout[2];
    pid_t pid;

    if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0)
        return -1;

    pid = fork();
   
    if (pid < 0){
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
        
    
        execl("/home/nassimiheb/MLIR/llvm-project/build/bin/mlir-cpu-runner", 
              "mlir-cpu-runner", "-e", "main", "-entry-point-result=void",
              "-shared-libs", "/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_runner_utils.so,/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_c_runner_utils.so",
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

double EvaluationByExecution::evaluateTransformation(/*int argc, char** argv, DialectRegistry &registry,*/ Node* node){


    std::string str1;
    llvm::raw_string_ostream output(str1);

    mlir::OwningOpRef<Operation*>* op = ((mlir::OwningOpRef<Operation*>*)(*node->getTransformedCodeIr()).getIr());
    mlir::PassManager pm((*op).get()->getName());

    if (std::getenv("AS_VERBOSE") != nullptr) {
        int asVerbose = std::stoi(std::getenv("AS_VERBOSE"));
        if (asVerbose == 1){
            std::ofstream debugFile;
            debugFile.open("logs.txt", std::ios_base::app);
            if (debugFile.is_open()) {
                std::string str;
                llvm::raw_string_ostream debugOut(str);
                if (node->getTransformation() != NULL){
                    debugFile << "###########################################################################" << std::endl;
                    debugFile << "Transformtion : "  << std::endl; 
                     for (const auto& transformation : node->getTransformationList()) {
                        debugFile <<  transformation->printTransformation();
                    }
                    debugFile << std::endl;
                }
                (**op)->print(debugOut);
                debugFile << str << std::endl;
                debugFile.close();
            }
        }
    }
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    bufferization::OneShotBufferizationOptions options;
    options.allowReturnAllocs = true;
    options.bufferizeFunctionBoundaries = true;

    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();    
    
    optPM.addPass(mlir::createConvertLinalgToLoopsPass());
    optPM.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertLinalgToLLVMPass ());
    optPM.addPass(memref::createExpandStridedMetadataPass ());
    optPM.addPass(mlir::createLowerAffinePass ());
    
    optPM.addPass( mlir::createArithToLLVMConversionPass());
    optPM.addPass(mlir::createConvertSCFToCFPass());
  
     
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass ());

    if (!mlir::failed(pm.run(*(*op))));

    (**op)->print(output);
    std::string command = "/home/nassimiheb/MLIR/llvm-project/build/bin/mlir-cpu-runner";


    int in_fd, out_fd;
    pid_t pid;

    // Call popen2 to execute the command and get the input and output file descriptors
    pid = popen2(command.c_str(), &in_fd, &out_fd);

    if (pid < 0) {
        perror("Failed to execute command");
        exit(EXIT_FAILURE);
    }
    // Measure the start time
    auto start = std::chrono::high_resolution_clock::now();
    write(in_fd, str1.c_str(), str1.size());

    close(in_fd);
    char output_data[9];

    // Read the output of the executed command
    ssize_t bytes_read = read(out_fd, output_data, sizeof(output_data) - 1);
    if (bytes_read > 0) {
        output_data[bytes_read] = '\0'; // Null-terminate the output data for printing
        printf("Command output:\n%s\n", output_data);
    }

    close(out_fd); // Close the output file descriptor

    // Wait for the child process to finish
    int status;
    waitpid(pid, &status, 0);

    // Check if the child process exited normally
    if (WIFEXITED(status)) {
        int exit_status = WEXITSTATUS(status);
        printf("Child process exited with status: %d\n", exit_status);

    } else {
        printf("Child process did not exit normally.\n");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //std::cout<< std::to_string(elapsed.count())<< std::endl;
        
        
        // int infp, outfp;
    // char buf[16];
//      auto begin = std::chrono::high_resolution_clock::now();
//     if (popen2(command.c_str(), &infp, &outfp) <= 0)
//     {
//         printf("Unable to exec sort\n");
//         exit(1);
//     }

//     write(infp, str1.c_str(), str1.size());

//     close(infp);

//     // Wait for the child process to finish

//     *buf = '\0';
//     read(outfp, buf, 16);
//     buf[15] = '\0';
//     printf("buf = '%s'\n", buf);
//             auto end = std::chrono::high_resolution_clock::now();
//    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

//     std::cout<< std::to_string(elapsed.count())<< std::endl;

return elapsed.count();
    
}

// #define CLEANUP_PIPE(pipe) close((pipe)[0]); close((pipe)[1])
// struct files_t {
//     FILE *in;
//     FILE *out;
// };

// typedef struct files_t files_t;

// struct files_chain_t {
//     files_t files;
//     pid_t pid;
//     struct files_chain_t *next;
// };
// typedef struct files_chain_t files_chain_t;

// static files_chain_t *files_chain;

// static int _do_popen2(files_chain_t *link, const char *command)
// {
//     int child_in[2];
//     int child_out[2];
//     if (0 != pipe(child_in)) {
//         return -1;
//     }
//     if (0 != pipe(child_out)) {
//         CLEANUP_PIPE(child_in);
//         return -1;
//     }

//     pid_t cpid = link->pid = fork();
//     if (0 > cpid) {
//        CLEANUP_PIPE(child_in);
//        CLEANUP_PIPE(child_out);
//        return -1;
//     }
//     if (0 == cpid) {
//         if (0 > dup2(child_in[0], 0) || 0 > dup2(child_out[1], 1)) {
//             _Exit(127);
//         }
//         CLEANUP_PIPE(child_in);
//         CLEANUP_PIPE(child_out);

//         for (files_chain_t *p = files_chain; p; p = p->next) {
//             int fd_in = fileno(p->files.in);
//             if (fd_in != 0) {
//                 close(fd_in);
//             }
//             int fd_out = fileno(p->files.out);
//             if (fd_out != 1) {
//                 close(fd_out);
//             }
//         }
//     // execl("/bin/sh", "sh", "-c", command, (char *) NULL);
//           execl("/home/nassimiheb/MLIR/llvm-project/build/bin/mlir-cpu-runner", 
//               "mlir-cpu-runner", "-e", "main", "-entry-point-result=void",
//               "-shared-libs", "/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_runner_utils.so,/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_c_runner_utils.so",
//               NULL);
//         _Exit(127);
//     }

//     close(child_in[0]);
//     close(child_out[1]);
//     link->files.in = fdopen(child_in[1], "w");
//     link->files.out = fdopen(child_out[0], "r");
//     return 0;
// }

// /**
//  * NAME
//  *     popen2 -- bidirectional popen()
//  *
//  * DESCRIPTION
//  *     popen2(const char *command) opens two pipes, forks a child process,
//  *     then binds the pipes to its stdin and stdout and execve shell to
//  *     execute given command.
//  *
//  * RETURN VALUES:
//  *     On success it returns a pointer to the struct with two fields
//  *     { FILE *in; FILE *out; }. The struct should be released via pclose2()
//  *     call. On failure returns NULL, check errno for more informaion about
//  *     the error.
//  */
// files_t *popen2(const char *command)
// {
//     files_chain_t *link = (files_chain_t *) malloc(sizeof (files_chain_t));
//     if (NULL == link) {
//         return NULL;
//     }

//     if (0 > _do_popen2(link, command)) {
//         free(link);
//         return NULL;
//     }

//     link->next = files_chain;
//     files_chain = link;
//     return (files_t *) link;
// }

// int pclose2(files_t *fp) {
//     files_chain_t **p = &files_chain;
//     int found = 0;
//     while (*p) {
//         if (*p == (files_chain_t *) fp) {
//             *p = (*p)->next;
//             found = 1;
//             break;
//         }
//         p = &(*p)->next;
//     }

//     if (!found) {
//         return -1;
//     }
//     if (0 > fclose(fp->in) || 0 > fclose(fp->out)) {
//         free((files_chain_t *) fp);
//         return -1;
//     }

//     int status = -1;
//     pid_t wait_pid;
//     do {
//         wait_pid = waitpid(((files_chain_t *) fp)->pid, &status, 0);
//     } while (-1 == wait_pid && EINTR == errno);

//     free((files_chain_t *) fp);

//     if (wait_pid == -1) {
//         return -1;
//     }
//     return status;
// }

// std::string createTempFile(const std::string& content) {
//     std::string tempFileName = "/tmp/tempfileXXXXXX";

//     // Generate a unique temporary file name
//     std::srand(static_cast<unsigned>(std::time(nullptr)));
//     for (int i = 0; i < 6; ++i) {
//         char randomChar = 'a' + (std::rand() % 26);
//         tempFileName += randomChar;
//     }

//     std::ofstream file(tempFileName);
//     if (!file) {
//         std::cerr << "Failed to create temporary file!" << std::endl;
//         return "";
//     }

//     file << content;
//     file.close();

//     return tempFileName;
// }
// std::string getUserTime(const std::string& command) {
//     auto start = std::chrono::high_resolution_clock::now(); // Start the timer
    
//     system(command.c_str()); // Execute the command
    
//     auto end = std::chrono::high_resolution_clock::now(); // Stop the timer
    
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // Calculate the duration in milliseconds
    
//     return std::to_string(duration.count()); // Convert the duration to string and return

//     //  // Construct the time command
    
//     // FILE* pipe = popen(command.c_str(), "r"); // Open a pipe to execute the command
//     // if (!pipe) {
//     //     return "Error executing command.";
//     // }
    
//     // const int bufferSize = 128;
//     // char buffer[bufferSize];
//     // std::string userTime;
    
//     // if (fgets(buffer, bufferSize, pipe) != nullptr) {
//     //     userTime = buffer; // Read the output into the userTime string
//     // }
    
//     // pclose(pipe); // Close the pipe
    
//     // return userTime;
// }




// pid_t popen2(const char *command, int *in_fd, int *out_fd) {
//     int     pin[2], pout[2];
//     pid_t   pid;
//     char    cmd[strlen(command)+10];

//     if (out_fd != NULL) {
//         if (pipe(pin) != 0) return(-1);
//     }
//     if (in_fd != NULL) {
//         if (pipe(pout) != 0) {
//             if (out_fd != NULL) {
//                 close(pin[0]);
//                 close(pin[1]);
//             }
//             return(-1);
//         }
//     }

//     pid = fork();

//     if (pid < 0) {
//         if (out_fd != NULL) {
//             close(pin[0]);
//             close(pin[1]);
//         }
//         if (in_fd != NULL) {
//             close(pout[0]);
//             close(pout[1]);
//         }
//         return pid;
//     }
//     if (pid==0) {
//         if (out_fd != NULL) {
//             close(pin[1]);
//             dup2(pin[0], 0);
//         }
//         if (in_fd != NULL) {
//             close(pout[0]);
//             dup2(pout[1], 1);
//         }
//         // Exec makes possible to kill this process 
//         sprintf(cmd, "exec %s", command);
//         execl("/home/nassimiheb/MLIR/llvm-project/build/bin/mlir-cpu-runner", 
//         "mlir-cpu-runner", "-e", "main", "-entry-point-result=void",
//         "-shared-libs", "/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_runner_utils.so,/home/nassimiheb/MLIR/llvm-project/build/lib/libmlir_c_runner_utils.so",
//         NULL);
//         //execlp("sh", "sh", "-c", "echo", NULL);
//         fprintf(stderr, "%s:%d: Exec failed in popen2. ", __FILE__, __LINE__);
//         perror("Error:");
//         exit(1);
//     }
//     if (in_fd != NULL) {
//         close(pout[1]);
//         *in_fd = pout[0];
//     }
//     if (out_fd != NULL) {
//         close(pin[0]);
//         *out_fd = pin[1];
//     }
//     return pid;
// }