# MLAutoScheduler

MLAutoScheduler is a tool designed specifically for optimizing machine learning workloads. It leverages MLIR transformations to achieve the best schedule for a given ML program. It uses the [coreAutoScheduler](https://github.com/MLIR-Autoscheduler/coreAutoScheduler) repo to implement the core functionalities of the autoscheduler.

<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites:
###### Required
1) [CMake](https://cmake.org/): version 3.20 or greater.
2) [Ninja](https://ninja-build.org/).
3) [Gcc](https://gcc.gnu.org/) : version 13.2.
4) [Gxx]: version 13.2.

### Build:
1. Building MLIR :
   ```sh
   git clone https://github.com/llvm/llvm-project.git
   git checkout release/18.x
   mkdir llvm-project/build
   cd llvm-project/build
   cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="clang;llvm;mlir;openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \

   cmake --build . --target check-mlir
   ```

   For more detalis follow instructions at [https://mlir.llvm.org/getting_started/](https://mlir.llvm.org/getting_started/)

2. Clone the repo
   ```sh
   git clone https://github.com/Modern-Compilers-Lab/MLAutoScheduler
   git checkout master
   ```
3. Clone the submodules
   ```sh
   git submodule update --init --recursive
   ```
4. Build 
   ```sh
    mkdir build
    cd build/
    cmake .. -DMLIR_DIR={Path to llvm folder}/build/lib/cmake/mlir
    cmake --build . -j
    ```
5. Add env variables :
   ```sh
   export LLVM_PATH={Path to llvm folder}
   export SHARED_LIBS=export SHARED_LIBS="${LLVM_PATH}/build/lib/libmlir_runner_utils.so,${LLVM_PATH}/build/lib/libmlir_c_runner_utils.so,${LLVM_PATH}/build/lib/libomp.so"
   export AS_VERBOSE=1 
   ```
6. Run
   ```sh
    bin/AutoSchedulerML ../benchmarks/{name of the benchmark}.mlir
   ```
