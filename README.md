# PMIRS

**A Privacy-Preserving Multi-User Retrieval System for
Multimodal AI**

## Overview
![figure1](https://github.com/user-attachments/assets/59cefc66-6cfb-40dd-a92f-539223ca2335)

PMIRS is a lightweight multimodal retrieval framework that supports secure image-to-text and text-to-image retrieval. It is built on distilled transformer encoders and introduces embedding-level obfuscation to protect feature privacy without compromising retrieval accuracy.

## Features

- Multimodal retrieval (image-text, text-image)
- Embedding-level obfuscation to ensure privacy
- Adaptive thresholding for scalable retrieval
- Optional federated training with multi-process simulation

## Dataset

We use a customized variant of the official ImageNet-1K dataset for evaluation and retrieval testing. The variant is structured for class-wise folder organization and reduced scale for faster experimentation. Scripts to construct this variant from the original ImageNet-1K dataset are provided in the data/ directory.

If you already have access to [ImageNet](https://www.image-net.org/), you can use the provided scripts to convert it into the required format.

## Model

Model can be trained using code in the src/training/ directory.

## Evaluation

Results in tables and figures can be supported by evaluation code in the src/inference/ directory.

## Quick Start

Install dependencies:

```bash
pip install torch torchvision open_clip_torch ftfy regex tqdm pandas
