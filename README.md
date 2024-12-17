# EGNN-GMM: Reliability of Molecular Property Predictions

## Overview

Imagine you are training an **Equivariant Graph Neural Network (EGNN)** to predict a molecular property \( X \). Once trained, you want to use this model to predict the same property \( X \) for a new set of molecules. 

But how confident are you that the predictions for these new molecules are reliable?

This is where **EGNN-GMM Fit** comes in. By combining the EGNN with a **Gaussian Mixture Model (GMM)** trained on the hidden (latent) layer of the EGNN, you can evaluate the **reliability** of your model's predictions for unseen data.

---

## Purpose

- **EGNN**: A model that predicts properties of molecules from graph-based representations.  
- **GMM in Latent Space**: The GMM is trained on the hidden representation (latent space, often dimmensional space higher than the node features) of the EGNN to model the distribution of the training data.  
- **Reliability Assessment**: By evaluating the Negative Log-Likelihood (NLL) of new data under the GMM, you can quantify how "in-distribution" or "out-of-distribution" the new molecules are.

---

## Why Train the GMM in Latent Space?
The latent space of the EGNN is the internal representation (hidden layer) learned by the network during training. Fitting a GMM in this space achieves two goals:

- **Model Training Data Distribution**:
 The GMM identifies the regions of the latent space where the training data is concentrated.

- **Evaluate New Data Reliability**:
New molecular graphs can be passed through the EGNN, and their hidden representations compared against the GMM. If these representations fall outside the GMM's learned density, it indicates the new data is dissimilar to the training set, making predictions less reliable.

- **Measure Confidence with Negative Log-Likelihood (NLL)**:
The GMM computes the likelihood of new data under its learned distribution. High Negative Log-Likelihood (NLL) values indicate high uncertainty.


This allows you to assess the confidence of predictions for new molecules.



## Features

1. **Train an EGNN**: Predict molecular properties using graph-based data.  
2. **Fit a GMM in Latent Space**: Train the GMM every 20 epochs on the hidden representations.  
3. **Reliability Quantification**: Compute the NLL of new molecular data to assess prediction reliability.  

---

## Usage

### 1. Training the Model

Train the EGNN for 100 epochs while fitting the GMM every 20 epochs:

```python
from egnn_gmm import train_egnn_gmm
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from torch_geometric.data import Data, DataLoader
from egnn_gmm import train_egnn_gmm, compute_reliability

# ----- Dummy Data Generation -----
# Simulate molecular graph data with random coordinates, node features, and bond types.

def generate_dummy_data(num_molecules=100, num_atoms=20, cutoff_radius=5.0):
    """
    Generate dummy molecular data with:
    - Node features: Atomic number (Z-number) or other properties.
    - Edge features: Bond types (single=1, double=2, triple=3).
    - Atom coordinates: Random 3D positions.
    """
    data_list = []
    for _ in range(num_molecules):
        # Node Features: Z-number (atomic number), random integers for simplicity
        node_features = torch.randint(1, 100, (num_atoms, 1))  # Shape: (num_atoms, num_node_features)

        # Coordinates: Random positions in 3D space
        coordinates = torch.randn(num_atoms, 3)  # Shape: (num_atoms, 3)

        # Edge Features: Simulate bond types (1=single, 2=double, 3=triple)
        edge_indices, edge_features = [], []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                # Randomly decide if a bond exists (simulate molecular connectivity)
                bond_exists = torch.rand(1).item() > 0.7  # Probability of a bond
                if bond_exists:
                    bond_type = torch.randint(1, 4, (1,)).item()  # Random bond type (1, 2, or 3)
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Add reverse edge for undirected graph
                    edge_features.append(bond_type)
                    edge_features.append(bond_type)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)

        # Create a PyTorch Geometric Data object
        data = Data(
            x=node_features,  # Node features (e.g., Z-number)
            pos=coordinates,  # Atom positions (3D coordinates)
            edge_index=edge_index,  # Edge indices
            edge_attr=edge_attr,  # Edge features (bond types)
        )
        data_list.append(data)
    
    return data_list

# Generate dummy molecular data
train_data = generate_dummy_data(num_molecules=100, num_atoms=20)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# ----- Model Training and GMM Fitting -----
# Define a dummy EGNN model (replace with your own EGNN implementation)
class DummyEGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DummyEGNN, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Forward pass:
        - Node features `x` and edge features `edge_attr` are processed.
        """
        x, edge_attr = data.x, data.edge_attr
        # Combine node and edge features (e.g., summing or concatenating)
        node_input = torch.cat([x.float(), edge_attr.mean().unsqueeze(0).expand_as(x)], dim=1)
        hidden = torch.relu(self.hidden_layer(node_input))
        return hidden, self.output_layer(hidden)

# Initialize the EGNN
input_dim = 2  # Node features + bond influence
hidden_dim = 16
output_dim = 1  # Property prediction
model = DummyEGNN(input_dim, hidden_dim, output_dim)

# Train the EGNN and Fit GMM every 20 epochs
n_epochs = 100
gmm_epochs = 20
n_components = 3  # Number of GMM components

trained_egnn, gmm = train_egnn_gmm(
    egnn=model,
    data_loader=train_loader,
    n_epochs=n_epochs,
    gmm_epochs=gmm_epochs,
    n_components=n_components
)

# ----- Reliability Evaluation -----
# Extract hidden features for new dummy data
new_data = generate_dummy_data(num_molecules=10, num_atoms=20)
new_loader = DataLoader(new_data, batch_size=1)

hidden_features = []
for batch in new_loader:
    hidden, _ = trained_egnn(batch)
    hidden_features.append(hidden.detach().numpy())
hidden_features = np.vstack(hidden_features)

# Compute Negative Log-Likelihood (NLL) for the new data
nll = -gmm.score_samples(hidden_features)
print("Mean NLL (Negative Log-Likelihood):", nll.mean())
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
pip install torch numpy matplotlib scikit-learn
```

---

## **Results**
- **Hidden Feature Visualization:** Plot embeddings of training graphs to assess clustering.
- **NLL Thresholding:** Identify graphs that lie outside the training distribution.
- **Prediction Reliability:** Flag unreliable predictions for out-of-distribution graphs.


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
