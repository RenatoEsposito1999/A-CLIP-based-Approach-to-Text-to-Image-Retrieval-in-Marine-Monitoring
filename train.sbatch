#!/bin/bash
#SBATCH --job-name=TI-RET
#SBATCH --output=gpu.out 
#SBATCH --error=gpu.err  
#SBATCH --time=06:00:00  
#SBATCH --nodes=1        
#SBATCH --partition=xgpu
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:tesla:1

# Set stack size (to avoid warning message)
ulimit -s 10240

module load cuda/10.1


# Add the commands needed to perform GPU processing
python main.py 
