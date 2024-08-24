time=`date +%m.%d-%H:%M`

B=1
H=5
val_dim=50
k=10
maxLeafSize=10
num_runs=10
maxN=100000
device="cpu"


for kq_dim in {1..10} {20..100..10}
do

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=devel
#SBATCH --cluster=arc
#SBATCH --nodes=1
#SBATCH --output=output/${time}-%x.out
#SBATCH --error=output/${time}-%x.err
#SBATCH --job-name=sparse_cpp_kq_${kq_dim}


module load Anaconda3
source activate graphgps
which python
lscpu

/data/engs-oxnsg/lady6515/.conda/envs/graphgps/bin/python \
    profiling-experiment.py \
    --B ${B} --H ${H} --kq_dim ${kq_dim} --val_dim ${val_dim} \
    --num_runs ${num_runs} --method sparse_cpp --device ${device} --maxN ${maxN} \
    --maxLeafSize ${maxLeafSize} --k ${k}
EOT


done