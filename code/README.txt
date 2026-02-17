================================================================================
                   COMPUTER VISION PROJECT 3
        Object Detection and Human-Object Interaction Analysis
================================================================================

Student: Shivani Kalal
Date: November 17, 2025
Course: Computer Vision

================================================================================
                        PROJECT OVERVIEW
================================================================================

This project implements object detection pipeline with three parts:

Part 1: Lightweight Object Detection (SSD)
Part 2: Non-Maximum Suppression (NMS)  
Part 3: Human-Object Interaction (HOI) Analysis

================================================================================
                        FILE STRUCTURE
================================================================================

NOTEBOOKS (Code):
-----------------
1. Part1_OD.ipynb - SSD implementation and training
2. Part2_nms.ipynb - Custom NMS with PyTorch comparison
3. hoi_part3.ipynb - HOI detection with Vision-Language Models

REPORTS (PDF Documentation):
----------------------------
1. Project_3_Report.pdf - Part 1 report (SSD training, results) Part 2 (NMS implementation, comparison),Part 3 (HOI implementation)

================================================================================
                    HOW TO RUN
================================================================================

PART 1 - Object Detection:
1. Open Part1_OD.ipynb in Google Colab
2. Run all cells in order
3. Training takes ~5 minutes on GPU
4. Model saves as "banana_ssd_model.pth"

PART 2 - NMS:
1. Open Part2_nms.ipynb in Google Colab
2. Load trained model from Part 1
3. Run all cells
4. Upload test images when prompted

PART 3 - HOI:
1. Open hoi_part3.ipynb in Google Colab
2. Follow dataset download instructions
3. Run all cells in order

================================================================================
                    EXPECTED OUTPUTS
================================================================================

PART 1:
- Training loss curves (total, class, bbox)
- Final train loss: ~0.30, val loss: ~0.39
- Sample detections on validation set
- Trained model: banana_ssd_model.pth

PART 2:
- Unit test results: 3/3 tests PASS ✓
- Before NMS: 50 overlapping boxes
- After PyTorch NMS: 3 clean boxes (red)
- After Custom NMS: 3 clean boxes (green)
- Comparison: IDENTICAL results

PART 3:
- HOI detection results
- Evaluation metrics
- Visualization of interactions



================================================================================
                    TECHNICAL SPECIFICATIONS
================================================================================

PART 1 - SSD MODEL:
- Architecture: 2 feature maps (32×32, 16×16)
- Input size: 256×256 pixels
- Anchors: 10,240 total (8 per location)
- Parameters: ~701,984
- Training: 20 epochs, Adam (lr=1e-3)
- Dataset: D2L Banana (1000 train, 100 val)
- Loss: Cross-Entropy + Smooth L1

PART 2 - NMS:
- Algorithm: Greedy (same as PyTorch)
- IoU threshold: 0.5 (standard)
- Confidence threshold: 0.3 (visualization)
- Time complexity: O(n²)
- Implementation: Pure PyTorch

PART 3 - HOI:
[Add specifications]

================================================================================
                    QUICK START GUIDE
================================================================================

STEP-BY-STEP:
1. Open Google Colab (colab.research.google.com)
2. Upload Part1_OD.ipynb → Runtime → Run all (~5 min)
3. Download banana_ssd_model.pth (or keep in Colab)
4. Upload Part2_nms.ipynb → Run all → Upload test images
5. Upload hoi_part3.ipynb → Follow instructions → Run all
6. Review all reports (PDFs) for methodology and results

Total estimated time: 30-45 minutes

================================================================================
                    TROUBLESHOOTING
================================================================================

Problem: "Model not found" error in Part 2
Solution: Run Part 1 first to train model OR upload saved model file

Problem: "No detections found" in Part 2
Solution: Lower confidence threshold: conf_thresh=0.01 (instead of 0.3)

Problem: "Out of memory" during training
Solution: Reduce batch_size from 32 to 16 or 8

Problem: Dataset download fails
Solution: Check internet connection; datasets auto-download in notebooks

Problem: Slow training
Solution: Ensure GPU is enabled (Runtime → Change runtime type → GPU)

================================================================================
                    REQUIREMENTS
================================================================================

Software:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- Pillow
- Google Colab (recommended) OR local setup with GPU

Hardware:
- GPU recommended for Part 1 (training)
- CPU sufficient for Parts 2 & 3
- ~2-4 GB GPU memory for training
- ~500 MB for inference

================================================================================
                    CONTACT & REFERENCES
================================================================================

Student: Shivani Kalal
Date: November 17, 2025


References:
- Liu et al. "SSD: Single Shot MultiBox Detector" (2016)
- D2L Banana Dataset
- PyTorch Documentation (torchvision.ops.nms)


