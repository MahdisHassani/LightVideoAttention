# LightVideoAttention

> Self-Supervised Video Attention using Optical Flow and Deep Learning

---

## Overview

This project implements a **self-supervised deep learning model** to learn attention maps from videos without manual annotations.

Instead of using labeled data, the model leverages **optical flow** to automatically generate pseudo-labels and learn motion-based attention.

---

## Key Features

- ✅ Self-supervised learning (no manual labels)
- ✅ Video-based attention modeling
- ✅ Lightweight U-Net architecture (GPU-friendly)
- ✅ Real-time inference (webcam support)
- ✅ Video & GIF demo generation

---

## Why Self-Supervised?

This project does not rely on human annotations.

Instead:

- Optical flow is computed between consecutive frames  
- Motion information is used as pseudo-labels  
- The model learns directly from these automatically generated labels  

> The model learns from automatically generated pseudo-labels without human annotations.

---

## Method

Pipeline:

1. Extract frames from raw videos  
2. Compute optical flow between consecutive frames  
3. Generate motion-based pseudo-labels  
4. Train a deep model to predict attention maps  

---

## Model Architecture

- U-Net based encoder-decoder
- Skip connections
- Lightweight design (runs on low GPU memory)

---

## Results

| Metric | Score |
|------|------|
| Dice | ~0.66 |
| IoU  | ~0.54 |

> The model converges early due to the simplicity of motion-based pseudo-labels and limited dataset size.

---

## Dataset

This project uses a subset of the UCF101 dataset.

Download dataset:
https://huggingface.co/datasets/aisuko/ucf101-subset

---

## Data Preparation

To generate the processed dataset from raw videos:

```bash
python src/data/extract_frames.py
python src/data/compute_flow.py
python src/data/dataset.py
python main.py
```

This will:

- Extract frames from videos  
- Compute optical flow  
- Build the processed dataset  

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Inference

### Video Prediction (Save Output)

```bash
python predict_video.py
```

---

### Quick Demo (Display Only)

```bash
python demo_video.py
```

---

### Real-Time Webcam

```bash
python webcam.py
```

---
