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

# Initialize and train EGNN
trained_egnn, gmm = train_egnn_gmm(
    egnn=model, 
    data_loader=train_loader, 
    n_epochs=100, 
    gmm_epochs=20, 
    n_components=3
)
```

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
