# MLScheduler

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Build:

1. Clone the repo
   ```sh
   git clone https://github.com/nassimiheb/MLScheduler.git
   ```
2. Build LLVM and MLIR 
    Follow instructions at [https://mlir.llvm.org/getting_started/](https://mlir.llvm.org/getting_started/)
3. Build 
   ```sh
    mkdir build
    cd build/
    cmake .. -DMLIR_DIR={Path to llvm folder}/llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT={Path to llvm folder}/llvm-project/build/bin/llvm-lit
    cmake --build .
    ```
4. Add env variables :
   ```sh
   export LLVM_PATH={Path to llvm folder}/project
   export SHARED_LIBS={set of shared libs used for mlir-cpu-runner}
   export AS_VERBOSE=1 (optinal)
   ```
5. Run
   ```sh
    bin/AutoSchedulerML ../benchmarks/{name of the benchmark}.mlir
   ```
