# Single Image Reflection Removal — Project README

## Team
- **Carolina (Yuhan) Li**
  - Implemented the *Traditional Optimization Baseline* (`traditional_method.py`)
  - Co-developed evaluation metrics and dataset preprocessing pipeline  
- **Peiyao Tao**
  - Developed the *CNN-based U-Net with ResNet encoder* (`Reflection_Removal.ipynb`)
  - Responsible for training scripts, data augmentation, and composite loss integration (L1 + VGG + Exclusion + Gradient)
- **Zihui Jiang**
  - Implemented the *Laplacian-Pyramid Transformer model* (`Laplacian_Based_Transformer_RR.ipynb`)
  - Focused on multi-scale feature fusion and transformer fine-tuning for reflection suppression
- **Ziyan Chen**
  - Implemented and trained *ERRNet* (Edge-aware Reflection Removal Network) using PyTorch (`train_errnet.py`, `errnet_model.py`)
  - Designed edge-map modules, GAN loss schedule, and evaluation loops on CEILNet and real-world datasets

## Operating System
- **Linux (Ubuntu 20.04 / 22.04)**

---

## Project Overview
This project evaluates **four reflection-removal approaches**:
1. **Traditional Optimization Baseline** – Relative smoothness and kernel-based estimation (`traditional_method.py`)  
2. **U-Net (ResNet encoder) CNN** – Deep learning-based reflection suppression (`Reflection_Removal.ipynb`)  
3. **Laplacian-Pyramid Transformer** – Transformer-based multi-scale decomposition (`Laplacian_Based_Transformer_RR.ipynb`)  
4. **ERRNet (Edge-aware Residual Refinement Network)** – GAN-enhanced architecture for edge-preserving restoration (`train_errnet.py`, `errnet_model.py`)

All methods were tested on **PASCAL VOC2012 (synthetic)** and **SIR² WildScene (real-world)** datasets using **PSNR** and **SSIM** metrics.

---

## Repository Structure
```
.
├── traditional_method.py
├── Reflection_Removal.ipynb
├── Laplacian_Based_Transformer_RR.ipynb
├── train_errnet.py
├── errnet_model.py
└── docs/
    └── Single Image Reflection Removal_ A Comparative Performance Report.docx
```

---

## Environment Setup
We recommend using **conda** (or mamba) with Python ≥ 3.10 and PyTorch.

```bash
# 1. Create environment
conda create -n sirr python=3.10 -y
conda activate sirr

# 2. Install PyTorch (CPU example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install opencv-python pillow matplotlib tqdm torchmetrics numba scipy
```

> To train ERRNet, additional modules from the original [Vandermode/ERRNet](https://github.com/Vandermode/ERRNet) repo are required under `/models`, `/util`, and `/data`.

---

## Dataset Layout
### VOC2012 (Synthetic)
```
VOC2012/
├── blended/
├── transmission_layer/
├── reflection_layer/
└── val_list.json
```

### SIR² WildScene (Real-world)
```
Wildscene/Wildscene/
├── 1/
│   ├── m.jpg   # mixed
│   ├── g.jpg   # ground truth
│   └── r.jpg   # reflection
└── ...
```

---

## Running Instructions

### 1️⃣ Traditional Method
```bash
python traditional_method.py   --voc_path ./VOC2012   --voc_json ./VOC2012/val_list.json   --wildscene_path ./Wildscene/Wildscene   --subset_size 20
```
**Outputs:** PSNR/SSIM scores and visualization plots.

---

### 2️⃣ CNN (U-Net + ResNet Encoder)
Open and run **`Reflection_Removal.ipynb`**:
1. Set dataset paths.  
2. Run *training* or *inference* cells.  
3. View PSNR/SSIM results and visual comparisons.

---

### 3️⃣ Laplacian-Pyramid Transformer
Run **`Laplacian_Based_Transformer_RR.ipynb`**:
1. Configure paths and pretrained weights (if any).  
2. Execute all cells for inference and evaluation.

---

### 4️⃣ ERRNet (Edge-aware GAN)
**Training Command:**
```bash
python train_errnet.py --dataroot /scratch/$USER/datasets/ --nEpochs 60
```

**Highlights:**
- Trains on CEILNet-style synthetic data.  
- Gradually introduces **GAN loss** after 20 epochs.  
- Evaluates every 5 epochs on synthetic + real testsets.  
- Model definition in `errnet_model.py` includes edge-map module, VGG/Contextual losses, and discriminator updates.  

---

## Reproduction Notes
- GUI issues in OpenCV on headless systems can be avoided by saving plots instead of `plt.show()`.  
- `traditional_method.py` and `ERRNet` can run on CPU; GPU is recommended for speed.  
- Reduce batch size if you encounter CUDA OOM errors.

---

## Report & Results
See the accompanying **Word report**:
> `docs/Single Image Reflection Removal_ A Comparative Performance Report.docx`

It summarizes each model’s architecture, training curves, and quantitative comparisons.

---

## Time-Travel Days
We **used 3 time-travel days** for this project.

---

## Acknowledgments
- Course staff and mentors for feedback  
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/) & [SIR² WildScene](https://sir2dataset.github.io/) datasets  
- ERRNet baseline by [Vandermode et al.](https://github.com/Vandermode/ERRNet)  
- Libraries: PyTorch, TorchMetrics, NumPy, SciPy, OpenCV, Matplotlib, Numba
