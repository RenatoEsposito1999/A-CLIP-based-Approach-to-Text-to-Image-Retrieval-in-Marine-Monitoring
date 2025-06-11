#!/bin/bash
#SBATCH --job-name=CLIP_full
#SBATCH --output=gpu_clip_full.out 
#SBATCH --error=gpu_clip_full.err  
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
python main.py --resume==True --lora==True --n_epochs=68 --learning_rate=5e-5 --resume_path='./checkpoint_full_clip/' --metrics_path='./metrics_full_clip/' --best_model_mix_path='best_model_mix_full_clip.pth'  --best_model_turtle_only_path='best_model_turtle_only_full_clip.pth'
