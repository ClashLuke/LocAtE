# LocAtE
Location-based Attention Exhaustion

## Model architecture
* **Inception-GAN**: optimizes computational cost while allowing the model to leverage larger receptive fields
* **Style-based**: using a dense ffnn with tanh activation
* **SpectralNorm + BatchNorm**: to enfore the lipschitz constant globally and therefore alleviate the gradient problems introduced by the multiplication of layer-outputs
* **Self-Attention**: to prioritize image regions 
* **Feature-Attention**: learnable location-based feature-level attention
* **Factorized Residual Layers**: enabling 64-layer deep generators