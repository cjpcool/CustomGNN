#  PR-GAT

This is the code of paper: Graph Attention Networks with LSTM-based Path Reweighting.


# Requirements

* Python 3.8.5
* PyTorch 1.7.1
* Please install other pakeages by `pip install -r requirements.txt`

# Datasets
* Cora and Citeseer are included in `.\data\cora` and `.\data\citeseer` respectively.

* Large datasets will be dowloaded automatically by PyTorch-Geometric when you run `python pr-gat.py --dataset ['CoraFull', 'CoauthorCS', 'AmazonComputers', 'AmazonPhoto']`

# Test PR-GAT

First you should unzip *pre-trained-model.zip*,
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

The details of datasets are as followes:
| Dataset         | Nodes | Edges  | Classes | Features | Label Rate |
|-----------------|-------|--------|---------|----------|------------|
| Cora            | 2708  | 5429   | 7       | 1433     | 0.0516     |
| Citeseer        | 3327  | 4732   | 6       | 3703     | 0.0360     |
| PubMed          | 19717 | 44338  | 3       | 500      | 0.0030     |
| Cora-Full       | 19749 | 63262  | 68      | 8710     | 0.0689     |
| Coauthor CS     | 18333 | 81894  | 15      | 6805     | 0.0164     |
| Amazon Computer | 13752 | 245861 | 10      | 767      | 0.0145     |
| Amazon Photo    | 7650  | 119081 | 8       | 745      | 0.0209     |



# Running Environments

The experiments of Cora, Citeseer are conducted on NVIDIA GeForce RTX 3070 Ti with 8GB memory size, the experiments of PubMed, Cora Full, Amazon Computer, Amazon Photo and Cauthor CS are conducted on Tesla V100 with 32GB memory size. 



