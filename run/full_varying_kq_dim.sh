# kq_dims = (1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50)
kq_dims=(1 5 10 20 30 50)

for kq_dim in ${kq_dims[@]}
do
    python profiling-experiment.py --method full --kq_dim $kq_dim --device cuda --folder full-varying-kq-dim --name kq$kq_dim --maxN 90000
done