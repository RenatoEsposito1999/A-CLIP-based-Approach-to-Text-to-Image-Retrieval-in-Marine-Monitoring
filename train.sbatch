#!/bin/bash
#SBATCH --job-name=TI-RET
#SBATCH --output=gpu.out 
#SBATCH --error=gpu.err  
#SBATCH --time=06:00:00  
#SBATCH --nodes=1        
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8	
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=40000


# Set stack size (to avoid warning message)
ulimit -s 10240

module load cuda/10.1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Add the commands needed to perform GPU processing
python main.py --learning_rate=1e-4
