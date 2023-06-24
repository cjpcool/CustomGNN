#  CustomGNN

This is the source code for paper: "Customizing Graph Neural Networks using Path Reweighting"

# Requirements

* Python 3.8.5
* PyTorch 2.0
* Please install other pakeages by `pip install -r requirements.txt`

# Datasets
Please unzip `data.zip` in `./data/` for preparing Cora, Citeseer, and PubMed datasets.

* Cora and Citeseer are included in `.\data\cora` and `.\data\citeseer` respectively.

* Large datasets will be dowloaded automatically by PyTorch-Geometric when you run `python CustomGNN.py --dataset ['CoraFull', 'CoauthorCS', 'AmazonComputers', 'AmazonPhoto']`

* The semi-synthetic Cora datasets with homophily ratio from 0.1 to 1.0 are also provided for further study.

* We also provided the codes for training on a subgraph of ogbn-arxiv with 25% nodes. The detailed data load code is shown in `./utils/def load_Ogbn`.

# Test CustomGNN
Please unzip `saved_models.zip` in `./checkpoint/` to prepare the trained models on Cora, Citeseer, and PubMed datasets.

* Test CustomGNN on Citeseer: `sh test_citeseer.sh`
* Test CustomGNN on Cora: `sh test_cora.sh`
* Test CustomGNN on Cora: `sh train_pubmed.sh`

# Train CustomGNN

* Train CustomGNN on Citeseer: `sh train_citeseer.sh`
* Train CustomGNN on Cora: `sh train_cora.sh`
* Train CustomGNN on Cora: `sh train_pubmed.sh`
* Train CustomGNN on ogbn-arxiv(0.25): `sh train_ogbn-arxiv0.25.sh`

# Results of CustomGNN

The test accuracies of CustomGNN on 7 datasets (3 citation graphs with public split and 4 large datasets with random split) are as follows.

| cora | citeseer | pubmed | Cora-Full | CoauthorCS | AmazonComputers| AmazonPhoto |
|------|----------|--------| ------ | ------ | ------ | ------ |
| 85.4 | 76.4     | 83.2   |44.2 |93.4 | 81.9 | 92.0 |



# Running Environments

The experiments the experiments of Cora-Full, Amazon Computer, Amazon Photo and Cauthor CS are conducted on Tesla V100 with 32GB memory size, and the experiments of Cora, Citeseer and PubMed are conducted on V100 with 80GB memory size.



---



Please feel free to contact me if you have any question about this code.
