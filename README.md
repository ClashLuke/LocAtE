# LocAtE
Location-based Attention Exhaustion

## Model architecture
* **Bottleneck**: optimizes computational cost, allowing the model to leverage deeper subnetworks
* **Style-based**: using a dense ffnn with roottanh activation
* **SpectralNorm + BatchNorm**: to enforce the lipschitz constant globally and therefore alleviate the gradient problems introduced by the multiplication of layer-outputs
* **Self-Attention**: to prioritize image regions 
* **Feature-Attention**: learnable location-based feature-level attention
* **Deep**: Every block (up/down) contains 6 convolutional layers (x2 if factorized), two attention layers (+2, +3) and one scale layer (=12 or 18)
* **RootTanh**: A brand-new customizable activation function