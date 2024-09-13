
## Environment setup

```
conda create -n mips python=3.9 -y
conda activate mips

conda install "numpy<2.0"
conda install pytorch=1.10 torchvision torchaudio faiss-gpu==1.8.0 -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

conda install pytorch::faiss-gpu==1.8.0

pip install pykeops matplotlib
```


Optional:
```
pip install ipykernel
pip install matplotlib
```