#!/usr/bin/env bash
# SBATCH --mem=32G
# SBATCH --time=70:00:00
# SBATCH --output=job_0_1000_%j.out

module load anaconda3
source activate nbodykit-env

for i in {0..1000}
do
    echo "doing $i"
    python generate_data.py --nmesh=256 --boxsize=1000. --seed=$i
done



