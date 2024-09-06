
## Environment setup

```
conda create -n mips python=3.9
conda activate mips

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

conda install pytorch::faiss-gpu

pip install pykeops
pip install ipykernel
```