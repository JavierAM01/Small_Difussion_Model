# Homework 2: Generative Models of Images

<p align="center">
  <img width="300" height="150" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/cat1.png">
  <img width="300" height="150" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/cat2.png">
</p>
  
## Course: 10-423/10-623 Generative AI

### Overview
This project explores different generative models for image synthesis, including Convolutional Neural Networks (CNNs), Encoder-only Transformers, Generative Adversarial Networks (GANs), and Denoising Diffusion Probabilistic Models (DDPMs). We implement and experiment with these architectures, analyzing their effectiveness for image generation and inpainting tasks.

## Project Structure
```
├── hw2/
│   ├── data/
│   ├── diffusion.py
│   ├── main.py
│   ├── requirements.txt
│   ├── run_in_cloud.ipynb
│   ├── trainer.py
│   ├── unet.py
│   ├── utils.py
├── README.md
```

## Getting Started

### Installation
To set up the environment locally, follow these steps:
1. Install Python dependencies:
   ```sh
   pip install torch einops clean-fid
   ```
2. Run the main training script:
   ```sh
   python main.py
   ```

---

## Convolutional Neural Networks (CNNs)
CNNs are widely used for image-based generative tasks, especially in architectures like U-Net. 

### U-Net for Image Inpainting
U-Net employs skip connections between the encoder and decoder layers to preserve spatial details during reconstruction. In this task, we use U-Net for inpainting, where missing pixels are filled based on surrounding image features.

#### Implementation
- Input: Partially masked image with a binary mask indicating missing pixels.
- Output: Reconstructed image with missing pixels filled.
- Loss Functions:
  - **MSE Loss** ensures that predicted pixels match the original ones.
  - **Adversarial Loss** (when used with a discriminator) improves realism.

<img width="900" height="500" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/unet.jpeg">

---

## Encoder-only Transformers
Transformers process entire sequences in parallel, making them effective for structured image representations.

### Model Implementation
We implement an encoder-only Transformer for part-of-speech tagging and analyze how these models differ from decoder-only variants.

- **Encoder-only Models** (e.g., BERT) generate contextual embeddings for all input tokens simultaneously.
- **Decoder-only Models** (e.g., GPT) use autoregressive generation, predicting tokens sequentially.

#### Applications:
- **Encoder-only**: Classification, segmentation, token-wise prediction.
- **Decoder-only**: Text/image generation, machine translation.

<img width="900" height="500" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/encoder_only.png">

---

## Generative Adversarial Networks (GANs)
GANs use an adversarial setup where a generator learns to create realistic samples while a discriminator tries to distinguish generated images from real ones.

### GAN-based Inpainting
For inpainting tasks, we use:
- **Generator (U-Net-based)**: Predicts missing pixels given an input mask.
- **Discriminator**: Distinguishes inpainted images from real ones.

#### Training Objectives:
- **Generator Loss:**
  ```
  L_G = E(x,m)[∥m⊙ (y − x')∥^2] - λ * E(x)[log D(x')]
  ```
- **Discriminator Loss:**
  ```
  L_D = E(x)[log D(x)] + E(x,m)[log(1 - D(x'))]
  ```

<img width="900" height="500" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/GANs.png">

---

## Denoising Diffusion Probabilistic Models (DDPMs)
Diffusion models generate images by gradually denoising random noise through a learned reverse process.

### Model Implementation
The **Diffusion** class implements forward and reverse diffusion using:
- **Cosine noise schedule** to control variance.
- **U-Net architecture** for denoising function.
- **Reparameterization trick** for efficient sampling.

#### Forward Process:
```python
xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
```
<img width="800" height="200" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/diffusion.png">

#### Reverse Process (Denoising):
```python
x_hat_0 = (xt - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
```
!()[diffusion_reverse_process]

### Training & Sampling
Training minimizes the L1 loss between predicted and actual noise:
```python
loss = F.l1_loss(pred_noise, noise)
```
During sampling, the model iteratively refines noisy images to generate realistic outputs.

<img width="900" height="100" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/forward_sample.png">
<img width="900" height="100" src="https://github.com/JavierAM01/Small_Difussion_Model/blob/main/images/backward_sample.png">

---

## How to Run the Code
1. Train the diffusion model:
   ```sh
   python main.py --train
   ```
2. Evaluate FID score:
   ```sh
   python main.py (...) --fid
   ```

---

## Key Learnings
- **CNNs** (U-Net) effectively reconstruct missing image regions.
- **Transformers** capture contextual dependencies in structured tasks.
- **GANs** produce sharper inpainted images but can be unstable.
- **Diffusion models** generate high-quality images with iterative refinement.

---

## Acknowledgments
This project is part of **10-623 Generative AI** at **Carnegie Mellon University**, with datasets and starter code provided by the course instructors.

