# 🎯 ScanRefer 3D Visual Grounding (VLM)

A multimodal 3D visual grounding framework that localizes objects in 3D indoor point clouds from natural language descriptions via cross-attention.

---

## 🌟 Highlights
- **End-to-End Multimodal Fusion**: Combines geometric point cloud representations (XYZ + RGB) with contextual text embeddings (`bert-base-uncased`) using multi-head cross-attention.
- **3D Bounding Box Localization**: Predicts oriented 3D bounding boxes `[cx, cy, cz, dx, dy, dz]` and proposal confidence scores.
- **Interactive 3D Visualizer**: Streamlit web interface with real-time Plotly 3D rendering of point clouds, predicted boxes (Red), and ground-truth boxes (Green).
- **Formal Evaluation Suite**: Offline evaluation script computing standard 3D visual grounding metrics: **Acc@0.25 IoU**, **Acc@0.50 IoU**, and **Mean IoU**.
- **Modular Codebase**: Cleanly decoupled model architecture, data loaders, 3D box geometry utilities, training pipeline, and application.

---

## 📁 Project Structure

```text
scanrefer_vlm_project/
├── app.py                      # Interactive Streamlit 3D visual grounding web app
├── evaluate.py                 # Standalone benchmark evaluation script
├── train.py                    # Training & fine-tuning pipeline
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py
│   └── vlm_grounding.py        # VLM3DVisualGrounding PyTorch architecture
├── utils/
│   ├── __init__.py
│   ├── box_utils.py            # 3D bounding box geometry & 3D IoU calculation
│   ├── data_loader.py          # PyTorch Dataset for ScanRefer point clouds & annotations
│   └── metrics.py              # Visual grounding evaluation metrics (Acc@0.25, Acc@0.5)
├── checkpoints/
│   └── vlm_3d_grounding_model.pth  # Pre-trained model weights
├── data/
│   ├── scannet/scans/          # Preprocessed ScanNet scenes (scene0000_00.npy, scene0001_00.npy)
│   └── scanrefer/              # Filtered train & val ScanRefer annotations
└── visualizations/             # Generated interactive 3D HTML visualizations
```

---

## 🚀 Quickstart

### 1. Environment Setup
```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (CPU PyTorch recommended if no dedicated NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers streamlit plotly numpy
```

### 2. Run the Interactive Streamlit App
```powershell
streamlit run app.py
```
- Select an indoor scene (Living Room, Office).
- Choose from ground truth queries or enter a custom utterance.
- Inspect the 3D point cloud, predicted box, and real-time **3D IoU score** against ground truth.

### 3. Run Benchmark Evaluation
To compute formal accuracy metrics across the validation benchmark:
```powershell
python evaluate.py
```
This generates Acc@0.25, Acc@0.50, and saves standalone interactive 3D HTML files in `visualizations/`.

### 4. Fine-Tuning / Training
```powershell
python train.py --epochs 15 --batch_size 2 --lr 1e-4
```

---

## 📐 Evaluation Metrics
- **Acc@0.25**: Percentage of queries where 3D Intersection-over-Union (IoU) $\ge 0.25$.
- **Acc@0.50**: Percentage of queries where 3D Intersection-over-Union (IoU) $\ge 0.50$.
- **Mean IoU**: Average 3D bounding box overlap across all queries.

