#!/bin/bash
#SBATCH -Jrun_parallel_on_cluster               # Job name
#SBATCH --account=gts-lj2                       # charge account
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                       # Memory per core
#SBATCH -t0-30:00:00                            # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=mschmitz7@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

echo "P_SET = $1"

module load abaqus/2023

module load anaconda3/2022.05.0.1
conda activate wave_env_cl

# conda info

srun python3 run_automated_simulations_cluster.py $1