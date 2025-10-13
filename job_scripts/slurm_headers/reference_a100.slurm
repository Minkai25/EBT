#!/bin/bash

#SBATCH --partition=gpu_requeue
### NODE/CPU/MEM/GPU  ### # can add mem gpu bind params below optionally
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-gpu=5
### TODO set the above mem-per-cpu and cpus-per-gpu according to your use case