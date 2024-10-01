# Bringing k-MIP Attention to Graph Transformers 
### Efficient k-MIP Attention

In this work, we introduce the k-MIP Graph Transformer, which is based on the k-Maximum Inner Product (k-MIP) attention mechanism and the [GraphGPS](https://github.com/rampasek/GraphGPS) framework.

The k-MIP Graph Transformer is:

- **Efficient:** The k-MIP self-attention mechanism is two orders of magnitude faster than full attention and has a negligible memory footprint, allowing us to scale to graphs with up to 500K nodes.
- **Versatile:** The k-MIP-GT incorporates edge features and supports node-level, graph-level, and edge-level tasks.
- **Performant:** We have demonstrated results competitive with prominent graph Transformers across a variety of graph learning tasks, with graphs ranging from 20 to 500K nodes.
- **Expressive:** We have established universal approximation guarantees for the k-MIP Graph Transformer, analogous to those previously established for full-attention Transformers and graph Transformers.

This repository was used to run the experiments in the following sections:

- 6.2: Efficiency
- 6.3: Ball Tree Search
- 6.7: An Approximation for Full Attention?

The integrated experiments in 6.1, 6.4, 6.5, 6.6, and 6.8 were run with our other repository: k-MIP-Graph-Transformer


## Environment setup with conda

```
conda create -n kmip python=3.9 -y
conda activate kmip

conda install "numpy<2.0"
conda install pytorch=1.10 torchvision torchaudio faiss-gpu==1.8.0 -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

conda install pytorch::faiss-gpu==1.8.0
pip install pykeops matplotlib

# installs the Ball Tree Search C++ extension
python setup.py build; python setup.py install
```


## Navigating the codebase

We highlight some important files and folders in the codebase.

| File/Folder                | Description                                      |
|----------------------------|--------------------------------------------------|
| `methods/`                 | Contains various methods used in the experiments, including:<ul> <li> `naive_sparse.py` </li> <li> `symbolic_sparse.py` </li> <li> `full.py` </li> <li> `faiss.py` </li> <li> `annoy.py` </li> <li> `post_processing.py` </li></ul> |
| `src/`                     | Contains the source code for the PyTorch C++ extension that we implemented for the Ball Tree Search algorithm.                     |
| `plotting.ipynb`           | Jupyter Notebook used for plotting results.           |
| `profiling-experiment.py`  | The main script for running the efficiency comparison experiments from 6.2 and 6.3.                |
| `approximation-experiment.py` | The main script for running the approximation quantification experiment from 6.7.         |


## Running efficiency experiments

```
# naive brute-force search
python profiling-experiment --method naive-sparse

# brute-force search with symbolic matrices
python profiling-experiment --method sym    

# ball tree search
python profiling-experiment --method 
```

`profiling-experiment.py` has the following further arguments: `B, H, kq_dim, val_dim, k, maxLeafSize, do_backward, require_grad`



Note: For running the approximation experiment, one would need the saved tokens in the GraphGPS repository.


<!-- Optional:
```
pip install ipykernel
pip install matplotlib
``` -->