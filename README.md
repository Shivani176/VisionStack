# VisionStack — SSD Object Detection + NMS + HOI Analysis (PyTorch)

VisionStack is a 3-part computer vision pipeline that covers:
1) **Lightweight Object Detection (SSD)**  
2) **Non-Maximum Suppression (NMS)** (custom implementation verified against PyTorch)  
3) **Human–Object Interaction (HOI)** analysis using a Vision–Language Model

This project is designed to be **reproducible, testable, and demo-friendly** in Google Colab.

---

## Highlights
- Trained a lightweight **SSD detector** (D2L Banana dataset) with loss curves and saved weights.  
- Implemented **custom greedy NMS** and validated it matches `torchvision.ops.nms` (**identical kept boxes** on unit tests).  
- Ran HOI analysis with a VLM workflow and reported metrics + visualizations.

---

## Repo Contents
### Notebooks
- `Part1_OD.ipynb` — SSD implementation + training  
- `Part2_nms.ipynb` — Custom NMS + comparison with PyTorch NMS  
- `hoi_part3.ipynb` — HOI detection / analysis with Vision–Language Models

### Report
- `Project_3_Report.pdf` — Full write-up covering Parts 1–3

> Your original run instructions + expected outputs are preserved and refined here.  

---

## Quickstart (Recommended: Google Colab)

### Part 1 — SSD Training (≈ 5 minutes on GPU)
1. Open `Part1_OD.ipynb` in Google Colab  
2. **Runtime → Change runtime type → GPU**  
3. Run all cells  
4. The model is saved as: `banana_ssd_model.pth`

### Part 2 — NMS (Custom vs PyTorch)
1. Open `Part2_nms.ipynb` in Colab  
2. Load `banana_ssd_model.pth` from Part 1  
3. Run all cells  
4. Upload test images when prompted

### Part 3 — HOI (VLM-based Analysis)
1. Open `hoi_part3.ipynb` in Colab  
2. Follow dataset download instructions inside the notebook  
3. Run all cells

**Total estimated time:** ~30–45 minutes (GPU recommended only for Part 1).

---

## Expected Outputs

### Part 1 — SSD
- Training loss curves (total / class / bbox)
- Final losses (approx):
  - Train: **~0.30**
  - Val: **~0.39**
- Sample detections on validation images
- Saved weights: `banana_ssd_model.pth`

### Part 2 — NMS
- Unit tests: **3/3 PASS**
- Example visualization:
  - Before NMS: **~50 overlapping boxes**
  - After PyTorch NMS: **3 clean boxes**
  - After Custom NMS: **3 clean boxes**
  - Result: **IDENTICAL** outputs

### Part 3 — HOI
- HOI predictions + evaluation metrics
- Interaction visualizations + qualitative analysis

---

## Technical Details

### Part 1 — SSD Model
- Input size: **256×256**
- Feature maps: **32×32** and **16×16**
- Anchors: **10,240 total** (8 per location)
- Parameters: **~701,984**
- Dataset: **D2L Banana** (1000 train / 100 val)
- Training: **20 epochs**, Adam (**lr=1e-3**)
- Loss: Cross-Entropy + Smooth L1

### Part 2 — NMS
- Algorithm: **Greedy NMS** (PyTorch-equivalent)
- IoU threshold: **0.5**
- Confidence threshold (viz): **0.3**
- Complexity: **O(n²)**
- Implementation: **Pure PyTorch**

### Part 3 — HOI
- VLM-driven HOI detection + evaluation + visualization (see notebook + report)

---

## Troubleshooting

**Part 2: “Model not found”**
- Run Part 1 first OR upload `banana_ssd_model.pth` manually.

**Part 2: “No detections found”**
- Lower confidence threshold (e.g., `conf_thresh=0.01` instead of `0.3`).

**Part 1: Out of memory**
- Reduce `batch_size` (e.g., 32 → 16 → 8).

**Slow training**
- Ensure GPU is enabled in Colab.

---

## Requirements
**Recommended:** Google Colab

If running locally:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, matplotlib, Pillow

Hardware:
- GPU recommended for Part 1 training
- CPU is sufficient for Parts 2 & 3


---

## References
- Liu et al., **“SSD: Single Shot MultiBox Detector”** (2016)
- D2L Banana Dataset
- PyTorch `torchvision.ops.nms`
