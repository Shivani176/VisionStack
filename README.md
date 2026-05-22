# VisionStack — Zero-Shot HOI Analysis with Vision-Language Models

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Sonnet-VLM-8A2BE2?style=for-the-badge&logo=anthropic&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

> **Exploring the boundary between classical Computer Vision and Vision-Language Models** —
> from SSD object detection and custom NMS to zero-shot HOI analysis using Claude Sonnet as a VLM.

---

## 🏆 Key Results

| Component | Result |
|-----------|--------|
| 🎯 SSD Detector | Train loss **0.30** / Val loss **0.39** |
| ✅ Custom NMS | **IDENTICAL** to torchvision — 3/3 unit tests PASS |
| 🤖 HOI Analysis | F1 improved **+0.49** on hardest cases via few-shot prompting |
| ⚡ Model Size | Lightweight **~701K parameters** |

---

## 💡 Why This Project

Most CV pipelines stop at object detection. This project goes further.

By replacing classical HOI detection with **Claude Sonnet** as a frontier Vision-Language Model and measuring the difference empirically, this project demonstrates that:

- Zero-shot VLM inference meaningfully outperforms classical approaches for interaction understanding
- Few-shot prompting improves F1 by **+0.49** on the hardest interaction categories
- VLM integration is a practical, measurable upgrade to classical CV pipelines

---

## 🗂️ Project Structure
VisionStack/
├── 📓 Part1_OD.ipynb        # SSD implementation + training
├── 📓 Part2_nms.ipynb       # Custom NMS + torchvision comparison
├── 📓 hoi_part3.ipynb       # HOI analysis with Claude Sonnet VLM
└── 📄 Project_3_Report.pdf  # Full methodology + results write-up

---

## 📊 Detailed Results

### Part 1 — SSD Object Detection
| Metric | Value |
|--------|-------|
| Train Loss | ~0.30 |
| Val Loss | ~0.39 |
| Parameters | ~701,984 |
| Training Time | ~5 min on GPU |
| Dataset | D2L Banana (1000 train / 100 val) |

### Part 2 — Custom NMS vs torchvision
| Test | Result |
|------|--------|
| Unit Tests | **3/3 PASS** |
| Boxes Before NMS | ~50 overlapping |
| After PyTorch NMS | 3 clean boxes |
| After Custom NMS | 3 clean boxes |
| Match | ✅ **IDENTICAL** |

### Part 3 — HOI: VLM vs Classical
| Prompting Strategy | Performance |
|-------------------|-------------|
| Zero-shot Claude Sonnet | Baseline |
| Few-shot Claude Sonnet | **+0.49 F1** on hardest cases |

---

## 🚀 Quickstart (Google Colab Recommended)

### Part 1 — SSD Training (~5 min on GPU)
```python
# 1. Open Part1_OD.ipynb in Google Colab
# 2. Runtime → Change runtime type → GPU
# 3. Run all cells
# Model saved as: banana_ssd_model.pth
```

### Part 2 — Custom NMS Verification
```python
# 1. Open Part2_nms.ipynb in Colab
# 2. Load banana_ssd_model.pth from Part 1
# 3. Run all cells — unit tests run automatically
```

### Part 3 — HOI Analysis with Claude Sonnet
```python
# 1. Open hoi_part3.ipynb in Colab
# 2. Add your Anthropic API key
# 3. Follow dataset download instructions
# 4. Run all cells — outputs: predictions, metrics, visualizations
```

⏱️ **Total estimated time:** ~30–45 minutes *(GPU recommended for Part 1 only)*

---

## ⚙️ Technical Details

<details>
<summary><b>SSD Model Architecture</b></summary>

| Parameter | Value |
|-----------|-------|
| Input Size | 256×256 |
| Feature Maps | 32×32 and 16×16 |
| Anchors | 10,240 total (8 per location) |
| Loss Function | Cross-Entropy + Smooth L1 |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 20 |

</details>

<details>
<summary><b>NMS Implementation</b></summary>

| Parameter | Value |
|-----------|-------|
| Algorithm | Greedy NMS (PyTorch-equivalent) |
| IoU Threshold | 0.5 |
| Complexity | O(n²) |
| Verified Against | torchvision.ops.nms |

</details>

<details>
<summary><b>HOI — VLM Setup</b></summary>

| Parameter | Value |
|-----------|-------|
| Model | Claude Sonnet (Anthropic) |
| Dataset | HICO-DET |
| Evaluation Metrics | F1, Precision, Recall |
| Prompting | Zero-shot and few-shot comparison |

</details>

---

## 🛠️ Requirements

**Recommended:** Google Colab (free GPU)

```bash
# Local setup
pip install torch torchvision anthropic
pip install numpy matplotlib Pillow
```

**Hardware:**
- 🖥️ GPU recommended for Part 1 training
- 💻 CPU sufficient for Parts 2 & 3

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| "Model not found" in Part 2 | Run Part 1 first or upload `banana_ssd_model.pth` manually |
| "No detections found" | Lower confidence: `conf_thresh=0.01` |
| Out of memory | Reduce batch size: 32 → 16 → 8 |
| Slow training | Enable GPU in Colab Runtime settings |

---

## 📚 References

- Liu et al., *"SSD: Single Shot MultiBox Detector"* (2016)
- D2L Banana Detection Dataset
- PyTorch `torchvision.ops.nms` documentation
- Anthropic Claude Sonnet API

---

## 📬 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shivani_Kalal-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/shivani-rk/)
[![GitHub](https://img.shields.io/badge/GitHub-Shivani176-181717?style=flat&logo=github)](https://github.com/Shivani176)
