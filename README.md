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
4. Build 
   ```sh
    mkdir build
    cd build/
    cmake .. -DMLIR_DIR={Path to llvm folder}/llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT={Path to llvm folder}/llvm-project/build/bin/llvm-lit
    cmake --build .
5. Run
   ```sh
    bin/AutoSchedulerML
   ```
