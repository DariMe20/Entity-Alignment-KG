# Entity-Alignment-KG

## Overview

This repository integrates two submodules for entity alignment:

1. **[EAkit](https://github.com/THU-KEG/EAkit)**: A PyTorch-based toolkit for embedding-based entity alignment.
2. **[OpenEA](https://github.com/nju-websoft/OpenEA)**: A TensorFlow-based framework for embedding-based entity
   alignment, including multiple models and benchmark datasets.

This README provides detailed instructions on setting up the local environment, configuring the datasets, and running
both projects.

---

## Prerequisites

### General Requirements

- **Python**: 3.7 (Recommended)
- **Conda**: Miniconda or Anaconda installed ([Download Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- **Git**: Installed and configured ([Download Git](https://git-scm.com/downloads))
- **Ubuntu WSL**: Windows Subsystem for Linux (if using
  Ubuntu) ([Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install))
- **Pycharm Community Edition**: Optional but
  useful – [Pycharm Download](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC)

---

### Cloning the Repository

Clone this repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/DariMe20/Entity-Alignment-KG.git
cd Entity-Alignment-KG
```

If you forgot to clone with submodules, initialize them manually:

```bash
git submodule update --init --recursive
```

---

## Environment Setup

### Environment for EAkit

1. Create a Conda environment for EAkit:
   ```bash
   conda create --name eakit python=3.7 -y
   conda activate eakit
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
   pip install numpy scipy scikit-learn tensorboard
   ```

### Environment for OpenEA

1. Create a Conda environment for OpenEA:
   ```bash
   conda create --name openea python=3.7 -y
   conda activate openea
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow==1.12 scipy numpy pandas scikit-learn gensim==3.8.3 protobuf==3.20.3
   ```
3. Install graph-tool (if using Ubuntu WSL):
   ```bash
   sudo apt update
   sudo apt install -y python3-graph-tool
   ```
4. Install OpenEA in editable mode:
   ```bash
   cd openea
   python -m pip install -e .
   cd ..
   ```

---

## Dataset Setup

### EAkit Dataset

1. Navigate to the `EAkit` directory:
   ```bash
   cd EAkit
   ```
2. Create a `data` folder:
   ```bash
   mkdir data
   ```
3. Clone the JAPE submodule for dataset preparation (if not already cloned):
   ```bash
   git submodule add https://github.com/nju-websoft/JAPE.git JAPE
   ```
4. Extract the dataset into the `data` folder:
   ```bash
   unzip data.zip -d data/
   ```
5. Verify the dataset structure:
   ```
   EAkit/data/DBP15K/
   ├── zh_en/
   ├── ja_en/
   └── fr_en/
   ```

### OpenEA Dataset

1. Navigate to the `OpenEA` directory:
   ```bash
   cd openea
   ```
2. Create a `data` folder:
   ```bash
   mkdir data
   ```
3. Download the OpenEA v2.0 dataset
   from [Figshare](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3?file=34234391).
4. Extract the dataset into the `data` folder:
   ```bash
   unzip downloaded_dataset.zip -d data/
   ```
5. Verify the dataset structure:
   ```
   OpenEA/data/EN_FR_15K_V1/
   ├── attr_triples_1
   ├── attr_triples_2
   ├── rel_triples_1
   ├── rel_triples_2
   ├── ent_links
   ├── 721_5fold/
   │   ├── 1/
   │   │   ├── test_links
   │   │   ├── train_links
   │   │   └── valid_links
   ```
6. Return to the root directory:
   ```bash
   cd ..
   ```

---

## Running EAkit

1. Activate the EAkit environment:
   ```bash
   conda activate eakit
   ```
2. Navigate to the `examples` folder:
   ```bash
   cd EAkit/examples
   ```
3. Run a predefined script. For example, to run **GCN-Align**:
   ```bash
   ./run_GCN-Align.sh
   ```
4. Alternatively, execute the Python command directly:
   ```bash
   python ../run.py --log gcnalign                     --data_dir "../data/DBP15K/zh_en"                     --rate 0.3                     --epoch 1000                     --check 10                     --update 10                     --train_batch_size -1                     --encoder "GCN-Align"                     --hiddens "100,100,100"                     --decoder "Align"                     --sampling "N"                     --k "25"                     --margin "1"                     --alpha "1"                     --feat_drop 0.0                     --lr 0.005                     --train_dist "euclidean"                     --test_dist "euclidean"
   ```
5. Monitor metrics using TensorBoard:
   ```bash
   ./Tensorboard.sh
   ```
   Open `http://localhost:6006` in a browser.

---

## Running OpenEA

1. Activate the OpenEA environment:
   ```bash
   conda activate openea
   ```
2. Navigate to the `run` directory:
   ```bash
   cd OpenEA/run
   ```
3. Run a predefined script. For example, to run BootEA:
   ```bash
   python main_from_args.py ./args/bootea_args_15K.json D_W_15K_V1 721_5fold/1/
   ```
4. Results (Hits@1, Hits@10, MR, MRR) will be displayed in the terminal.

---

## Common Issues

### Conda Environment Issues

- Ensure Conda is added to your PATH. Run:
  ```bash
  conda init
  ```
  Then restart your terminal.

### Dataset Not Found

- Ensure datasets are downloaded and extracted in the correct structure.

### Dependency Conflicts

- Update `pip`, `setuptools`, and `wheel`:
  ```bash
  pip install --upgrade pip setuptools wheel
  ```

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

---

## Citation

If you use this repository, please cite the original papers for **EAkit** and **OpenEA**:

- **EAkit**:
  ```plaintext
  @article{eakit,
    title={A comprehensive survey of entity alignment for knowledge graphs},
    author={Zeng, Kaisheng and Li, Chengjiang and Hou, Lei and Li, Juanzi and Feng, Ling},
    journal={AI Open},
    volume={2},
    pages={1--13},
    year={2021},
    publisher={Elsevier}
  }
  ```

- **OpenEA**:
  ```plaintext
  @article{OpenEA,
    author={Zequn Sun and Qingheng Zhang and Wei Hu and Chengming Wang and Muhao Chen and Farahnaz Akrami and Chengkai Li},
    title={A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs},
    journal={Proceedings of the VLDB Endowment},
    volume={13},
    number={11},
    pages={2326--2340},
    year={2020},
    url={http://www.vldb.org/pvldb/vol13/p2326-sun.pdf}
  }
  ```
