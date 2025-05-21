# Audio Anomaly Detector

An end-to-end, modular PyTorch pipeline for detecting anomalies in machine sound recordings, integrating wavelet-based, transformer-based, and VQ-VAE fusion models. Designed for research and production use.

---

## 📜 Research Publication (KCC 2025)

We submitted our work **"Multi-Modal Audio Anomaly Detection with Enhanced DeSpaWN, Masked Audio Transformers, and VQ-VAE Fusion"** to the *2025 Korea Computer Conference (KCC)*.

You can download the full paper here: [KCC 2025 Audio Anomaly Detection Paper (PDF)](./pdf/self_supervised_wavelet_transformer_fusion_industrial_audio_anomaly_detection.pdf)

---

## 🔍 Features

* **Modular Architecture**: Clean separation into utilities, model definitions, training loops, and entry points.
* **Enhanced DeSpaWN**: Wavelet-based decomposition with learnable filters and attention mechanisms for robust feature extraction.
* **Masked Audio Transformer**: Custom transformer blocks with temporal convolution and masking for self-supervised spectrogram reconstruction.
* **VQ-VAE Fusion**: Hierarchical cross-attention fusion of wavelet and transformer features, followed by vector quantization for anomaly scoring.
* **Test-Time Augmentation**: Multiple noisy augmentations improve robustness of final anomaly score.
* **Comprehensive Evaluation**: ROC AUC, Precision-Recall, F1 vs. threshold, t-SNE visualization of latent space, and confusion matrices.

---

## 📁 Directory Structure

```
AudioAnomalyDetector/
├── docs/                       # Supplementary materials (e.g., KCC 2025 paper PDF)
│   └── kcc2025_audio_anomaly_detection.pdf
├── src/
│   ├── utils/                  # seed, audio I/O, dataset preparation
│   ├── models/                 # EnhancedDeSpaWN, Transformer, VQ-VAE fusion
│   ├── train/                  # training loops for each component
│   ├── main.py                 # single-ID pipeline entrypoint
│   └── run_all.py              # batch processing across all IDs
├── requirements.txt            # pip dependencies
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone <YOUR_REPO_URL> AudioAnomalyDetector
   cd AudioAnomalyDetector
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) CUDA-enabled PyTorch**:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## 🚀 Usage

### 1. Single-ID Pipeline

Prepare and train on one sensor ID:

```bash
python src/main.py \
  --data /path/to/Sound_Dataset/valve \
  --id id_00 \
  --length 160000 \
  --device cuda
```

* **Output**:

  * `vqvae_results.png`: evaluation plots
  * `final_audio_anomaly_model.pth`: saved model states and metadata

### 2. Batch Processing

Run across all `<condition>/<machine>/<id>` combinations:

```bash
python src/run_all.py \
  --root /path/to/Sound_Dataset \
  --save ./best_models
```

* Saves best model per ID under `./best_models/` as `best_<condition>_<machine>_<id>.pth`

---

## 🔧 Configuration

* **Random Seed**: `src/utils/seed.py` → `set_seed(seed)`
* **Dataset Prep**: `src/utils/dataset.py` → `prepare_datasets(base_folder, target_id, length, ...)`
* **Model Hyperparameters**: set in `src/main.py` (e.g., levels, embed\_dim, num\_heads)
* **Training Loops**: see `src/train/` for epochs, learning rates, batch sizes

---

## 📈 Evaluation & Visualization

* **ROC AUC & PR AUC** metrics printed per epoch
* **t-SNE** of quantized latent codes colored by anomaly score
* **Histogram & Curves**:

  * Anomaly score distributions
  * ROC curve
  * Precision-Recall curve
  * F1 vs. threshold

Plots saved automatically during training and evaluation.

---

## 📖 License

Licensed under the MIT License. © 2025 Your Name / Your Organization
