#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --constraint="GPUMEM32GB|GPUMEM80GB"
#SBATCH --time=24:00:00
#SBATCH --mail-user=andrej.obuljen@uzh.ch
#SBATCH --mail-type=ALL 
#SBATCH --output=gpus_job_Pk2D_%j.out

module load multigpu cuda/11.4.4
module load multigpu cudnn/8.2.4
module load mamba
source activate torch_env
python -c 'import torch as t; print("is available: ", t.cuda.is_available()); print("device count: ", t.cuda.device_count()); print("current device: ", t.cuda.current_device()); print("cuda device: ", t.cuda.device(0)); print("cuda device name: ", t.cuda.get_device_name(0)); print("cuda version: ", t.version.cuda)'
python main.py >> logfile_${SLURM_JOB_ID}.log
