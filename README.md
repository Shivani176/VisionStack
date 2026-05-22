# VisionStack — Zero-Shot HOI Analysis with Vision-Language Models

A computer vision project exploring the transition from classical CV approaches 
to Vision-Language Model (VLM)-based inference — implementing SSD object detection, 
custom NMS, and zero-shot HOI analysis using Claude Sonnet as a VLM.

**Key Results:**
- SSD detector: Train loss 0.30 / Val loss 0.39
- Custom NMS: Identical outputs to torchvision (3/3 unit tests PASS)
- HOI Analysis: F1 improved by +0.49 on hardest cases via few-shot prompting vs zero-shot baseline

---

## Highlights

- Trained a lightweight **SSD detector** (~701K parameters) on D2L Banana dataset; 
  achieved train loss 0.30 / val loss 0.39 with saved weights and loss curves.
- Implemented **custom greedy NMS** in pure PyTorch; verified identical outputs to 
  `torchvision.ops.nms` across all unit tests (3/3 PASS, 50 boxes → 3 clean boxes).
- Extended classical CV pipeline with **zero-shot HOI analysis** using **Claude Sonnet** 
  as a Vision-Language Model over HICO-DET images — demonstrating VLM integration 
  beyond classical detection approaches.
- Compared **few-shot prompting vs zero-shot** inference; few-shot improved F1 by 
  +0.49 on hardest interaction categories.

---

## Why This Project

Most CV pipelines stop at object detection. This project goes further — replacing 
classical HOI detection with a frontier VLM (Claude Sonnet) and measuring the 
difference empirically. The result demonstrates that zero-shot VLM inference can 
meaningfully improve on classical approaches for interaction understanding, 
particularly when combined with few-shot prompting strategies.

---

## Repo Structure
VisionStack/
├── Part1_OD.ipynb          # SSD implementation + training
├── Part2_nms.ipynb         # Custom NMS + torchvision comparison
├── hoi_part3.ipynb         # HOI analysis with Claude Sonnet VLM
└── Project_3_Report.pdf    # Full write-up with methodology and results

---

## Results

### SSD Object Detection
| Metric | Value |
|--------|-------|
| Train Loss | ~0.30 |
| Val Loss | ~0.39 |
| Parameters | ~701,984 |
| Training Time | ~5 min (GPU) |

### Custom NMS vs torchvision
| Test | Result |
|------|--------|
| Unit Tests | 3/3 PASS |
| Boxes before NMS | ~50 overlapping |
| Boxes after PyTorch NMS | 3 clean boxes |
| Boxes after Custom NMS | 3 clean boxes |
| Match | **IDENTICAL** |

### HOI Analysis — VLM vs Classical
| Approach | F1 Score |
|----------|----------|
| Zero-shot Claude Sonnet | Baseline |
| Few-shot Claude Sonnet | +0.49 on hardest cases |

---

## Quickstart (Google Colab Recommended)

### Part 1 — SSD Training (~5 min on GPU)
1. Open `Part1_OD.ipynb` in Google Colab
2. Runtime → Change runtime type → **GPU**
3. Run all cells
4. Model saved as `banana_ssd_model.pth`

### Part 2 — Custom NMS
1. Open `Part2_nms.ipynb` in Colab
2. Load `banana_ssd_model.pth` from Part 1
3. Run all cells — unit tests run automatically

### Part 3 — HOI Analysis with Claude Sonnet
1. Open `hoi_part3.ipynb` in Colab
2. Add your Anthropic API key
3. Follow dataset download instructions
4. Run all cells — outputs include predictions, metrics, and visualizations

**Total estimated time:** ~30–45 minutes (GPU recommended for Part 1 only)

---

## Technical Details

### SSD Model
- Input: 256×256
- Feature maps: 32×32 and 16×16
- Anchors: 10,240 total (8 per location)
- Loss: Cross-Entropy + Smooth L1
- Optimizer: Adam (lr=1e-3), 20 epochs

### NMS Implementation
- Algorithm: Greedy NMS (PyTorch-equivalent)
- IoU threshold: 0.5
- Complexity: O(n²)
- Verified against: `torchvision.ops.nms`

### HOI — VLM Setup
- Model: Claude Sonnet (Anthropic)
- Dataset: HICO-DET
- Evaluation: F1, Precision, Recall
- Prompting strategies: Zero-shot and few-shot comparison

---

## Troubleshooting

**"Model not found" in Part 2** → Run Part 1 first or upload `banana_ssd_model.pth` manually

**"No detections found"** → Lower confidence threshold: `conf_thresh=0.01`

**Out of memory** → Reduce batch size: 32 → 16 → 8

**Slow training** → Ensure GPU is enabled in Colab Runtime settings

---

## Requirements

**Recommended:** Google Colab (free GPU)

Local setup:
Python 3.8+
PyTorch 2.0+
torchvision
anthropic
numpy, matplotlib, Pillow

---

## References

- Liu et al., "SSD: Single Shot MultiBox Detector" (2016)
- D2L Banana Detection Dataset
- PyTorch torchvision.ops.nms documentation
- Anthropic Claude Sonnet API
