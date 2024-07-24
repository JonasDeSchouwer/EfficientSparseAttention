time=`date +%m-%d-%H:%M`

# everything except job-name, potential GPU resource
slurm_directive="#!/bin/bash
#SBATCH --partition=short
#SBATCH --cluster=htc
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --output=output/${time}-%x-%j.out
#SBATCH --error=output/${time}-%x-%j.err
#SBATCH --mem=1TB
"



B=1
H=5
kq_dim=50
val_dim=50
k=10
maxLeafSize=10
num_runs=5

# sparse_cpp
sbatch <<EOT
${slurm_directive}
#SBATCH --job-name=sparse_cpp
#SBATCH --cpus-per-task=96

module load Anaconda3
source activate graphgps
which python
lscpu
python setup.py build; python setup.py install --force

python profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --k ${k} --maxLeafSize ${maxLeafSize} \
    --num_runs ${num_runs} --method sparse_cpp
EOT


# sparse_symbolic CPU
sbatch <<EOT
${slurm_directive}
#SBATCH --job-name=sparse_sym_cpu
#SBATCH --cpus-per-task=96

module load Anaconda3
source activate graphgps
which python
lscpu
python setup.py build; python setup.py install --force

python profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --num_runs ${num_runs} --method sparse_symbolic
EOT


# full attention CPU
sbatch <<EOT
${slurm_directive}
#SBATCH --job-name=full_attn_cpu
#SBATCH --cpus-per-task=96

module load Anaconda3
source activate graphgps
which python
lscpu
python setup.py build; python setup.py install --force

python profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --num_runs ${num_runs} --method full_attention
EOT


# sparse symboolic GPU
sbatch <<EOT
${slurm_directive}
#SBATCH --job-name=sparse_sym_gpu
#SBATCH --gres=gpu:1

module load Anaconda3
module load CUDA/12.0.0
source activate graphgps
which python
lscpu
nvidia-smi
python setup.py build; python setup.py install --force

python profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --num_runs ${num_runs} --method sparse_symbolic --device cuda
EOT


# full attention GPU
sbatch <<EOT
${slurm_directive}
#SBATCH --job-name=full_attn_gpu
#SBATCH --gres=gpu:1

module load Anaconda3
module load CUDA/12.0.0
source activate graphgps
which python
lscpu
nvidia-smi
python setup.py build; python setup.py install --force

python profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --num_runs ${num_runs} --method full_attention --device cuda
EOT





