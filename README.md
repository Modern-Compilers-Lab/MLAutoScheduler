# MLAutoScheduler

MLAutoScheduler is a tool designed specifically for optimizing machine learning workloads. It leverages MLIR transformations to achieve the best schedule for a given ML program.
<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites:
###### Required
1) [CMake](https://cmake.org/): version 3.20 or greater.
2) [Ninja](https://ninja-build.org/).
3) [Gcc](https://gcc.gnu.org/) : version 13.2.

### Build:
1. Building MLIR :
   ```sh
   git clone https://github.com/llvm/llvm-project.git
   git checkout release/17.x
   mkdir llvm-project/build
   cd llvm-project/build
   cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="clang;llvm;mlir;openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   ```

   For more detalis follow instructions at [https://mlir.llvm.org/getting_started/](https://mlir.llvm.org/getting_started/)

2. Clone the repo
   ```sh
   git clone https://github.com/MLIR-Autoscheduler/MLAutoScheduler.git
   ```
3. Build 
   ```sh
    mkdir build
    cd build/
    cmake .. -DMLIR_DIR={Path to llvm folder}/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT={Path to llvm folder}/build/bin/llvm-lit
    cmake --build .
    ```
4. Add env variables :
   ```sh
   export LLVM_PATH={Path to llvm folder}
   export SHARED_LIBS={set of shared libs used for mlir-cpu-runner}
   export AS_VERBOSE=1 (optinal)
   ```
5. Run
   ```sh
    bin/AutoSchedulerML ../benchmarks/{name of the benchmark}.mlir
   ```
