#!/bin/bash
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#############SBATCH --cpus-per-task=128
#SBATCH -t 7-0:00:00
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
#SBATCH --mem=64G
#SBATCH --reservation=c2

WORKER_NUM=$((SLURM_JOB_NUM_NODES - 1))
PORT=6379
CONDA_DIR=/share/apps/NYUAD5/miniconda/3-4.11.0
CONDA_ENV=/home/ia2280/.conda/envs/mlir
RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
RAY_ALLOW_SLOW_STORAGE=1
pwd=$(pwd)

. $CONDA_DIR/bin/activate
conda activate $CONDA_ENV

cmake --build .
export LLVM_PATH=/scratch/ia2280/LLVM/llvm-project
export SHARED_LIBS=/scratch/ia2280/LLVM/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/ia2280/LLVM/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/ia2280/LLVM/llvm-project/build/lib/libomp.so
export AS_VERBOSE=1
time ./bin/AutoSchedulerML ../benchmarks/NNs/alexnet.mlir





