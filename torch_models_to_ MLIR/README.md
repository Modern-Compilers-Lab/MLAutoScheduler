
# PyTorch to MLIR Conversion.

This guide will walk you through the steps to convert torch modules into MLIR.

First, we will need to build and set up `torch-mlir`:

### I. Clone the Torch-MLIR Repository

First, clone the `torch-mlir` repository and initialize its submodules:

```bash
git clone https://github.com/llvm/torch-mlir
cd torch-mlir
git submodule update --init --progress
```

### II. Set Up Python Virtual Environment and Dependencies

While the submodules are updating, set up a Python virtual environment and install the necessary dependencies.

1. Create and activate a Python virtual environment:

```bash
python3 -m venv mlir_venv
source mlir_venv/bin/activate
```

2. Upgrade `pip` (older versions may not handle recent PyTorch dependencies):

```bash
python -m pip install --upgrade pip
```

3. Install PyTorch nightlies and the build requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r torchvision-requirements.txt
```

4. Install Python development headers:

```bash
sudo apt install python3-dev
```

### III. Build with CMake

Depending on whether you already have LLVM installed, follow one of the two cases below:

#### Case 1: If You Do Not Have LLVM Installed

Use `cmake` to build the project along with LLVM and MLIR:

```bash
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm
```

Then build:

```bash
cmake --build build
```

#### Case 2: If You Already Have LLVM Installed

If you already have LLVM installed, make sure to set the `$LLVM_INSTALL_DIR` variable to point to your LLVM installation directory. Here’s how to set it:

```bash
export LLVM_INSTALL_DIR=/path/to/your/llvm-project
```

Now, run the following `cmake` command:

```bash
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  .
```

Then build the project:

```bash
cmake --build build --target tools/torch-mlir/all
```

### IV. Set Up Python Environment to Export the Built Python Packages

To use the built Python packages in your environment, export the `PYTHONPATH`:

```bash
export PYTHONPATH=pwd/build/tools/torch-mlir/python_packages/torch_mlir:pwd/test/python/fx_importer
```

### V. Optional: Set Up Jupyter Notebook

If you want to work with Jupyter notebooks (as shown in the provided example):

1. Install the kernel for `torch-mlir`:

```bash
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH="$PYTHONPATH"
```

2. When opening Jupyter, select the "torch-mlir" kernel.

### VI. Make Scripts Executable

Ensure that the scripts `convert.sh` and `execute.sh` are executable:

```bash
sudo chmod +x ./convert.sh
sudo chmod +x ./execute.sh
```

### VII. Adjust the paths
In the `convert.sh` and `execute.sh` scripts, change all `/home/nouss/Desktop/PFE/HWNASACOMLIR/Solution/llvm-project` occurents into your own `/path/to/your/llvm-project`.

### VIII. Example Notebook

For a step-by-step explanation, refer to the `example.ipynb` notebook included in the project.
