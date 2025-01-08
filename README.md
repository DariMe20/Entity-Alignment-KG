# Entity Alignment Using RAG (EA_RAG)

## Overview

This repository introduces **EA_RAG**, a novel framework for entity alignment in knowledge graphs (KGs) using **Retrieval-Augmented Generation (RAG)** with Large Language Models (LLMs). The framework dynamically integrates retrieval mechanisms with generative models to align entities across multilingual and multi-modal KGs. 

The primary contribution of this project is the implementation of RAG-based alignment methods, evaluated against traditional embedding-based approaches provided by the **EAkit** framework.

## Features

- **RAG-Based Alignment**: A novel entity alignment approach leveraging generative capabilities of LLMs.
- **Dynamic Retrieval**: Incorporates external knowledge during the alignment process for improved accuracy.
- **Evaluation Using EAkit**: Benchmarked against state-of-the-art embedding-based methods such as GCN-Align, MTransE, and BootEA.
- **Cross-Lingual Matching**: Optimized for datasets like DBP15K, supporting multilingual knowledge graphs.
- **Scalable Architecture**: Uses Pinecone for efficient vector storage and retrieval.

---

## Prerequisites

### General Requirements

- **Python**: 3.11
- **Conda**: Miniconda or Anaconda ([Download Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- **Git**: Installed and configured ([Download Git](https://git-scm.com/downloads))
- **Pinecone**: API for vector embedding management ([Pinecone Documentation](https://docs.pinecone.io/))

---

## Getting Started

### Clone the Repository

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/your-repo/EA_RAG.git
cd EA_RAG
```

### Create a Conda Environment

1. Create and activate the environment:
   ```bash
   conda create --name ea_rag python=3.11 -y
   conda activate ea_rag
   ```

2. Verify the Python version:
   ```bash
   python --version  # Should output 3.11.x
   ```

### Install Dependencies

Install required packages using pip:

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file with the following details:

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

---

## Dataset Setup

### Download DBP15K Dataset

1. Download the DBP15K dataset from [Hugging Face](https://huggingface.co/datasets/HackCz/DBP15K_raw/blob/main/DBP_raw.zip).
2. Extract and rename the folder to `fr_en` in the repository directory.


## Using EA_RAG

The implementation and usage of the EA_RAG framework are detailed in the Jupyter Notebook `ea-rag/main.ipynb`. Follow the instructions in the notebook to:

- Embed knowledge graph entities using LLM-based text embedding models.
- Store and retrieve embeddings with Pinecone.
- Query and align entities using a combination of retrieval mechanisms and generative models.

## EA_RAG Results
Experiments were conducted on subsets of the DBP15K dataset (English-French), comparing the RAG-based alignment against ground truth. Below are the results:

| Dataset Size      | Hits@1 | Processing Time         |
|-------------------|--------|-------------------------|
| 150 Entities      | 0.6667 | ~30 seconds           |
| 1,500 Entities    | 0.6421 | ~7 minutes            |
| 15,000 Entities   | 0.6465 | ~61 minutes           |

## EAkit Results
The embedding-based methods implemented in EAkit were evaluated on the same DBP15K (English-French) dataset, focusing on Hits@1 accuracy. Below are the results for each method:

| Method     | Hits@1 | Processing Time |
|------------|--------|-----------------|
| BootEA     | 51.5%  |~12 hours      |
| GCN-Align  | 43.0%  |~25 minutes     |
| MTransE    | 40.7%  |~11 minutes     |


## Running EAkit

To benchmark EA_RAG, we evaluated it against embedding-based methods provided by **EAkit**. Note that EAkit requires a separate environment setup:

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

### Running EAkit Methods

1. Activate the EAkit environment:
   ```bash
   conda activate eakit
   ```

2. Navigate to the examples folder:
   ```bash
   cd EAkit/examples
   ```

3. Run a predefined alignment script (e.g., GCN-Align):
   ```bash
   ./run_GCN-Align.sh
   ```

4. Alternatively, execute directly with custom parameters:
   ```bash
   python ../run.py --data_dir "../data/DBP15K/zh_en" --encoder "GCN-Align"
   ```
5. Monitor metrics using TensorBoard:
   ```bash
   ./Tensorboard.sh
   ```
   Open `http://localhost:6006` in a browser.

---


## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Citation

If you use this framework, please cite the relevant works:

- **EAkit**:
  ```plaintext
  @article{zeng2021comprehensive,
    title={A comprehensive survey of entity alignment for knowledge graphs},
    author={Zeng, Kaisheng and Li, Chengjiang and Hou, Lei and Li, Juanzi and Feng, Ling},
    journal={AI Open},
    volume={2},
    pages={1--13},
    year={2021},
    publisher={Elsevier}
  }
  ```

