#  PR-GAT

This is the code of paper: Graph Attention Networks with LSTM-based Path Reweighting


# Requirements

* Python 3.8.5
* PyTorch 1.7.1
* Please install other pakeages by `pip install -r requirements.txt`

# Datasets
* Cora and Citeseer are included in `.\data\cora` and `.\data\citeseer` respectively.

* Large datasets will be dowloaded automatically by PyTorch-Geometric when you run `python pr-gat.py --dataset ['CoraFull', 'CoauthorCS', 'AmazonComputers', 'AmazonPhoto']`

# Test PR-GAT

First you should unzip `pretraining-model.zip`,
* Test PR-GAT on Citeseer: `sh test_citeseer.sh`
* Test PR-GAT on Cora: `sh test_cora.sh`

# Train PR-GAT

* Train PR-GAT on Citeseer: `sh train_citeseer.sh`
* Train PR-GAT on Cora: `sh train_cora.sh`

# Results of PR-GAT

The test accuracies of PR-GAT on 7 datasets (3 citation graphs with public split and 4 large datasets with random split) are as follows.

| cora | citeseer | pubmed | Cora-Full | CoauthorCS | AmazonComputers| AmazonPhoto |
| ---- | -------- | ------ | ------ | ------ | ------ | ------ |
| 85.3 | 76.2     | 82.1   |44.2 |93.4 | 81.9 | 92.0 |



# Running Environments

The experiments of Cora, Citeseer are conducted on NVIDIA GeForce RTX 3070 Ti with 8GB memory size, the experiments of PubMed, Cora Full, Amazon Computer, Amazon Photo and Cauthor CS are conducted on Tesla V100 with 32GB memory size. 



