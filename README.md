# EGNN-GMMFit
model that extracts hidden features, train a Gaussian Mixture Model (GMM), to give a graph similarity evaluation
# **EGNN-GMM: Graph Similarity Feasibility Checker**

This project combines an **E(n)-Equivariant Graph Neural Network (EGNN)** with a **Gaussian Mixture Model (GMM)** to evaluate graph input feasibility. The hidden node features extracted from the EGNN model are used to train the GMM, which predicts the negative log-likelihood (NLL) of input graphs. This helps assess if input graphs are similar to the training set and ensures reliable predictions.

---

## **Overview**
The pipeline consists of two main components:
1. **EGNN (E(n)-Equivariant Graph Neural Network):**
   - Extracts hidden node-level features from graphs.
   - Processes node features, edge features, and coordinates in an equivariant manner.
the Equivariant Graph Convolutional Layer inspired by https://github.com/vgsatorras/egnn
2. **Gaussian Mixture Model (GMM):**
   - Trains on extracted hidden features to compute NLL for graph inputs.
   - Provides a measure to assess input graph similarity to the training distribution.

---

## **Features**
- **E(n)-Equivariant Neural Network:** Handles graph data with node features, edge features, and spatial coordinates.
- **Hidden Feature Extraction:** Extracts node-level embeddings from a specified hidden layer.
- **Graph Feasibility Evaluation:** Uses GMM to predict NLL, indicating graph similarity to training data.
- **Modular Implementation:** Easily customizable EGNN and GMM components.

---

## **Project Structure**

```
EGNN-GMM/
│
├── egnn_gmm.py        # EGNN implementation and utility functions
├── train_gmm.py       # Script to train EGNN and GMM
├── utils.py           # Helper functions (e.g., edge computation)
├── requirements.txt   # Required Python libraries
└── README.md          # Project documentation
```

---

## **Installation**

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EGNN-GMM.git
cd EGNN-GMM
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Training the EGNN Model**

The EGNN model processes graphs to extract node embeddings.

**Example Usage:**
```python
from egnn_gmm import EGNN, initialize_weights_egnn, get_edges
import torch

# Generate Random Graph Data
num_nodes = 50
node_features = torch.randn(num_nodes, 5)
node_coords = torch.randn(num_nodes, 3)
edge_indices, edge_attr = get_edges(node_coords, cutoff=1.5)

# Initialize EGNN
egnn = EGNN(in_node_nf=5, hidden_nf=10, out_node_nf=2, n_layers=3)
initialize_weights_egnn(egnn)

# Forward Pass
hidden_features, updated_coords = egnn(node_features, node_coords, edge_indices)
print("Hidden Features Shape:", hidden_features.shape)
```

---

### **2. Extracting Hidden Features**

You can extract features from a specific layer using the `get_hidden_representation` method:

```python
# Extract Hidden Features from Layer 2
hidden_features = egnn.get_hidden_representation(node_features, node_coords, edge_indices, layer_index=2)
print("Extracted Hidden Features:", hidden_features.shape)
```

---

### **3. Training the GMM**

The extracted node features are used to train a Gaussian Mixture Model:

```python
from sklearn.mixture import GaussianMixture

# Train GMM on Hidden Features
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(hidden_features.detach().numpy())

# Evaluate NLL
nll = -gmm.score_samples(hidden_features.detach().numpy())
print("Mean NLL:", nll.mean())
```

---

### **4. Feasibility Check for Input Graphs**

To assess the feasibility of a new input graph:

```python
# New Graph Input
new_node_features = torch.randn(num_nodes, 5)
new_node_coords = torch.randn(num_nodes, 3)
new_edges, _ = get_edges(new_node_coords, cutoff=1.5)

# Pass Graph Through EGNN
input_hidden_features = egnn.get_hidden_representation(new_node_features, new_node_coords, new_edges)

# Compute NLL
input_nll = -gmm.score_samples(input_hidden_features.detach().numpy())
print("Graph NLL:", input_nll.mean())

# Feasibility Check
threshold = 5.0  # Example threshold
if input_nll.mean() < threshold:
    print("Graph is similar to the training set. Prediction is feasible.")
else:
    print("Graph is out-of-distribution. Prediction may be unreliable.")
```

---

## **Code Structure**

### **EGNN Class**
- **Input:** Node features, coordinates, and edge indices.
- **Layers:** Custom `E_GCL` layers for equivariant graph processing.
- **Output:** Updated node embeddings and coordinates.

### **`E_GCL` Layer**
- Implements an E(n)-Equivariant Graph Convolutional Layer.
- Processes edge and node features while respecting equivariance.

### **Hidden Feature Extraction**
- Allows access to intermediate node embeddings at any layer using `get_hidden_representation()`.

### **GMM Feasibility Check**
- Trains a Gaussian Mixture Model on node embeddings.
- Computes NLL for input graphs to determine their similarity to the training data.

---

## **Dependencies**
The project requires the following libraries:

- `torch` (>= 1.10)
- `numpy`
- `matplotlib`
- `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Results**
- **Hidden Feature Visualization:** Plot embeddings of training graphs to assess clustering.
- **NLL Thresholding:** Identify graphs that lie outside the training distribution.
- **Prediction Reliability:** Flag unreliable predictions for out-of-distribution graphs.

---

## **Contributions**
Contributions are welcome! Feel free to open issues or submit pull requests to enhance this project.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Cuauhtemoc**
**Cuauhtemoc Nuñez Valencia**  
[GitHub](https://github.com/cuauhtemocnv) | [LinkedIn](https://linkedin.com/in/cuauhtemocnv)

---

## **Future Work**
- Extend GMM to other graph embeddings or probabilistic models.
- Integrate dynamic graph processing for time-dependent data.
- Implement uncertainty quantification using Variational Inference.

---

**Feel free to star ⭐ this repository and share feedback!**
