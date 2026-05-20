# 🧠 CIFAR-10 Image Classification with Custom ResNet

A PyTorch implementation of a custom **Residual Neural Network (ResNet)** trained from scratch on the CIFAR-10 dataset, achieving strong classification accuracy across 10 object categories.

---

## 📌 Project Overview

This project builds and trains a custom ResNet architecture on CIFAR-10 without using pretrained weights. It demonstrates:

- Building residual blocks and skip connections from scratch
- Proper data augmentation pipeline for small images
- Training loop with loss/accuracy tracking
- Visual evaluation of predictions on test samples

---

## 🏗️ Model Architecture

```
Input (3×32×32)
    ↓
Conv2d(3 → 64) + BN + ReLU
    ↓
Layer 1: ResidualBlock(64→64) × 2
    ↓
Layer 2: ResidualBlock(64→128, stride=2) × 2
    ↓
Layer 3: ResidualBlock(128→256, stride=2) × 2
    ↓
AdaptiveAvgPool → Flatten
    ↓
FC(256 → 10)
```

Each **ResidualBlock** contains:
- Conv → BN → ReLU → Conv → BN
- Skip connection (with 1×1 conv projection when dimensions change)

---

## 📊 Dataset

**CIFAR-10** — 60,000 color images (32×32), 10 classes:

| Label | Class  | Label | Class |
|-------|--------|-------|-------|
| 0     | Plane  | 5     | Dog   |
| 1     | Car    | 6     | Frog  |
| 2     | Bird   | 7     | Horse |
| 3     | Cat    | 8     | Ship  |
| 4     | Deer   | 9     | Truck |

- **Train:** 50,000 images
- **Test:** 10,000 images
- Downloaded automatically via `torchvision`

---

## ⚙️ Training Configuration

| Hyperparameter     | Value                            |
|--------------------|----------------------------------|
| Optimizer          | Adam (lr=0.001)                  |
| LR Scheduler       | StepLR (step=3, gamma=0.5)       |
| Batch Size         | 128                              |
| Epochs             | 13                               |
| Loss Function      | CrossEntropyLoss                 |

**Data Augmentation (Train):**
- Random Horizontal Flip
- Random Crop (32×32, padding=4)
- Normalize: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

---

## 📈 Results

Training and test curves are plotted at the end of the notebook:

- **Loss Curve** — Train vs Test loss over epochs
- **Accuracy Curve** — Train vs Test accuracy over epochs
- **Prediction Grid** — 10 sample test images with true/predicted labels (green = correct, red = wrong)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-resnet.git
cd cifar10-resnet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook CIFAR10_ResNet.ipynb
```

> The CIFAR-10 dataset will be downloaded automatically on first run (~170MB).

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

> GPU is recommended but not required — the code auto-detects CUDA.

---

## 📁 Project Structure

```
cifar10-resnet/
│
├── CIFAR10_ResNet.ipynb   # Main notebook (model, training, evaluation)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔮 Potential Improvements

- [ ] Switch to SGD + Momentum + CosineAnnealingLR for better convergence
- [ ] Use CIFAR-10 standard normalization: mean=(0.4914, 0.4822, 0.4465)
- [ ] Train for 50–100 epochs for higher accuracy
- [ ] Add per-class accuracy breakdown
- [ ] Save/load model weights (`torch.save`)
- [ ] Add confusion matrix visualization

---

## 🧑‍💻 Author

**[Your Name]**
- GitHub: (https://github.com/Azouz2005
- LinkedIn: (https://www.linkedin.com/in/mohamed-azouz-54a467327/))

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
