README for Enhanced Knowledge Graph Attention Networks (EKGAT)

Overview
This repository contains the implementation of Enhanced Knowledge Graph Attention Networks (EKGAT) designed to improve representation learning for graph-structured data. The model integrates TransformerConv layers and disentanglement learning techniques to enhance node classification accuracy and convergence speed. Experiments have been conducted on the Cora, PubMed, and Amazon datasets, demonstrating substantial improvements over traditional KGAT models.

Repository Structure
ekgat.py: Main script containing the implementation of the EKGAT model.
datasets/: Directory to store datasets used in the experiments.
results/: Directory to store the results and metrics from experiments.
figures/: Directory for storing figures generated from PCA and t-SNE analyses.
Requirements
Python 3.10.8
PyTorch 1.10.1
PyTorch Geometric
scikit-learn
numpy
matplotlib
psutil
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/ekgat.git
cd ekgat
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate
Install dependencies:

pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv scikit-learn psutil
Datasets
The experiments are conducted on the following datasets:

Cora: A citation network dataset where nodes represent documents and edges represent citation links between documents.
PubMed: A citation network dataset containing scientific publications from PubMed database.
Amazon (Computers): A co-purchase network where nodes represent products and edges indicate products frequently bought together.
Usage
Download datasets:
The datasets will be automatically downloaded and saved in the datasets/ directory when running the script for the first time.

Run the training script:

bash
Copy code
python ekgat.py --epochs 500 --lr 0.005 --weight_decay 0.001 --dataset_path ./datasets
Model Architecture
KGAT Model: Utilizes two KGATConv layers to perform node classification tasks.
KGAT with TransformerConv: Enhances the KGAT model by integrating TransformerConv layers to capture complex relationships.
EKGAT: Adds a DisentangleLayer after the TransformerConv layer to segment entity representations into independent components.
Evaluation
The models are evaluated based on the following metrics:

Training and validation accuracy
Precision, recall, and F1-score
ROC-AUC
Convergence speed
Memory usage during training and inference
Experimental Results
The results from experiments on the Cora, PubMed, and Amazon datasets show that the EKGAT model significantly improves node classification accuracy and convergence speed compared to traditional KGAT models. Detailed results and figures from PCA and t-SNE analyses can be found in the results/ and figures/ directories.

Acknowledgments
This research was funded in part by NSF grant number CCF-2109988.

References
Velickovic, P., et al. "Graph Attention Networks." arXiv:1710.10903
Ying, Z., et al. "Transformers in Graph Representation Learning." arXiv:2107.00154
Wu, Z., et al. "A Comprehensive Survey on Graph Neural Networks." arXiv:1901.00596
Liu, Q., et al. "Disentangling representations in knowledge graphs." arXiv:2006.07127
