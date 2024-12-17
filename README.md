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

## Why GMM in Latent Space?

The hidden (latent) layer of the EGNN contains learned, lower-dimensional embeddings of the input molecular graphs. By fitting a **Gaussian Mixture Model (GMM)** in this space, we capture the underlying distribution of the training data's latent representations. 

For **new data**, the GMM calculates the Negative Log-Likelihood (NLL):
- **Low NLL** → High reliability (similar to training data).
- **High NLL** → Low reliability (out-of-distribution).

This allows you to assess the confidence of predictions for new molecules.

---

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

# Initialize and train EGNN
trained_egnn, gmm = train_egnn_gmm(
    egnn=model, 
    data_loader=train_loader, 
    n_epochs=100, 
    gmm_epochs=20, 
    n_components=3
)


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
