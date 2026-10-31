# 🛰️ Next-Day Wildfire Spread Prediction via Multi-Modal Remote Sensing & SHAP Architectures

A deep learning benchmark predicting 24-hour lead-time wildfire front propagation across the Continental United States (CONUS). This project integrates 12 satellite and atmospheric variables, contrasts a **Residual CNN** against an **Attention-Preserving U-Net**, and quantifies environmental drivers using **SHAP Feature Importance** and **Grad-CAM**.

[![Dataset](https://img.shields.io/badge/Dataset-NextDayWildfireSpread-orange.svg)](#)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%202.x-red.svg)](#)
[![Resolution](https://img.shields.io/badge/Grid-1km%20Resolution-blue.svg)](#)

---

## 📌 1. Operational Problem & Motivation

Wildfires inflict catastrophic ecological damage, destroy residential communities, and cause significant public health degradation. Providing tactical **24-hour lead-time fire front forecasts** allows emergency management agencies to:
- Pre-position aerial firefighting tankers and ground suppression personnel.
- Construct strategic defensive fuelbreaks and backburn zones.
- Execute targeted public evacuations and de-energize vulnerable electrical grids.

This repository implements a data mining and deep segmentation pipeline processing 12 distinct spatial channels to predict pixel-level active fire masks at time (t + 1) based on antecedent conditions at time (t).

---

## 🗺️ 2. Input Tensor & Multi-Modal Covariates

The feature space merges 11 environmental, meteorological, and topographic variables with the prior day's active fire mask into a unified (64 × 64 × 12) tensor:

| Channel Index | Variable Name | Data Source | Physical Dimension |
| :---: | :--- | :--- | :--- |
| **0** | **Elevation** | SRTM GL1 (30 m) | Topography and terrain slope |
| **1** | **Wind Speed (`vs`)** | GRIDMET (4 km) | 10 m surface wind velocity |
| **2** | **Wind Direction (`th`)** | GRIDMET (4 km) | Angular wind heading |
| **3** | **Min Temperature (`tmmn`)** | GRIDMET (4 km) | Diurnal minimum temperature |
| **4** | **Max Temperature (`tmmx`)** | GRIDMET (4 km) | Diurnal peak thermal maximum |
| **5** | **Specific Humidity (`sph`)** | GRIDMET (4 km) | Atmospheric moisture ratio |
| **6** | **Precipitation (`pr`)** | GRIDMET (4 km) | 24-hour accumulated rainfall |
| **7** | **Drought Index (`pdsi`)** | GRIDMET Drought (4 km) | Palmer Drought Severity Index |
| **8** | **NDVI** | VIIRS VNP13A1 (500 m) | Live fuel load and canopy density |
| **9** | **Population Density** | GPWv4 SEDAC (1 km) | Anthropogenic assets and exposure |
| **10** | **ERC** | GRIDMET (4 km) | Energy Release Component (fuel moisture) |
| **11** | **PrevFireMask** | MODIS Terra MOD14A1 (1 km) | Active burn boundary at step t |
| **Target** | **FireMask** | MODIS Terra MOD14A1 (1 km) | Binary active fire footprint at step t + 1 |

---

## ⚙️ 3. Preprocessing, Noise Filtering & Data Pipeline

```text
Raw TFRecord Files (CONUS 2012–2020)
                 │
                 ▼
[Step 1: TFRecord Feature Parsing]
  └─ Decode 12 continuous and categorical grids to (64, 64, 12) tensors
                 │
                 ▼
[Step 2: Noisy Pixel Removal]
  └─ Discard samples containing unobserved cloud/sensor masks (value = -1)
  └─ Cleaned samples: 8,330 Train / 1,085 Eval / 867 Test
                 │
                 ▼
[Step 3: Percentile-Based Scaling]
  └─ Clip channels [0–10] to [0.5%, 99.5%] percentiles
  └─ Min-Max scale values strictly within the [0, 1] interval
                 │
                 ▼
[Step 4: Spatial Data Augmentation]
  └─ Random 32 × 32 crops, 90-degree rotations, and horizontal/vertical reflections
                 │
                 ▼
[Step 5: Generator Batching]
  └─ Train: 261 batches | Eval: 34 batches | Test: 28 batches (Batch Size = 32)
```

---

## 🧠 4. Deep Learning Architectures & Loss Formulation

### 4.1 Class-Weighted Cross-Entropy
Because active fire pixels represent less than 2% of the spatial domain, standard binary cross-entropy leads to model collapse toward the majority non-burn class. We utilize a **Weighted Binary Cross-Entropy**:

```math
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
```

Where the positive fire class receives an empirical weight multiplier of w = 2.5 and non-burn pixels w = 1.0.

### 4.2 Structural Comparison

| Feature | Model 1: Residual CNN | Model 2: Deep U-Net |
| :--- | :--- | :--- |
| **Encoder Structure** | 4 Residual Blocks (Conv2D 64 → BN → Skip Add → Dropout 0.2) | 3 Encoder Levels (Conv2D 64, 128, 256 + MaxPool 2×2) |
| **Bottleneck Layer** | Direct bottleneck representation | Conv2D 512 + Dropout 0.3 |
| **Decoder Pathway** | Cascaded Conv2D layers (32 → 16 → 8) | UpSampling2D + Concatenated Skip Features (256 → 128 → 64) |
| **Output Layer** | Conv2D (3 × 3, 1 filter, Sigmoid) | Conv2D (1 × 1, 1 filter, Sigmoid) |
| **Total Parameters** | **327,985 (~329K)** | **7,790,913 (~7.8M)** |
| **Optimizer & LR** | Adam (α = 0.001) | Adam (α = 0.001) |

---

## 📊 5. Quantitative Model Benchmarking

Models were evaluated on 867 unseen test scenes representing **3,550,664 individual 1-km spatial pixels**:

### 5.1 Comprehensive Test Metrics

| Model Architecture | Loss | Pixel Accuracy | Recall (Sensitivity) | Precision | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Residual CNN** | 0.0836 | 98.43% | 36.19% | 35.84% | 0.9189 |
| **Deep U-Net** | **0.0708** | **98.55%** | **43.09%** | **41.06%** | **0.9476** |
| **Performance Delta** | **-15.3% Error** | **+0.12%** | **+6.90%** | **+5.22%** | **+0.0287** |

---

### 5.2 Confusion Matrix Evaluation

<div align="center">
  <img width="998" height="511" alt="Confusion Matrix Benchmark" src="https://github.com/user-attachments/assets/6d7dbd61-b4ec-4526-a0c5-0032ca2f8197" />
  <p><em>Figure 1: Confusion matrices of Residual CNN versus Deep U-Net on 3.55 million test pixels.</em></p>
</div>

- **Active Detection Advantage:** The U-Net identified **18,722 true active fire pixels**, outpacing the Residual CNN by 2,996 verified detections.
- **False Alarm Mitigation:** False positive fire predictions were concurrently reduced from 28,157 down to 26,879 pixels.

---

## 🛰️ 6. Qualitative Spread Prediction Visuals

<div align="center">
  <img width="100%" alt="Qualitative Spread Prediction Visuals" src="https://github.com/user-attachments/assets/119897ce-f202-4dcf-a921-70c4ff40c4b3" />
  <p><em>Figure 2: Multi-sample spatial comparison displaying Previous Fire Mask (t), Ground Truth Mask (t + 1), and Model Inferences.</em></p>
</div>

The U-Net captures complex directional fire front expansion driven by wind-topography interactions, avoiding the blocky artifacts produced by simpler convolutional filters.

---

## 🔬 7. Model Explainability with GRAD-CAM & SHAP Feature Sensitivity

Channel perturbation ablation was conducted to calculate the marginal impact of each environmental channel on test set Recall:

**ΔRecall = Recall(baseline) - Recall(perturbed)**

<div align="center">
  <img width="100%" alt="Grad-CAM Visualization" src="https://github.com/user-attachments/assets/9865052a-c3b5-4ead-aed3-c4eefd43efdb" />
  <br><br>
  <img width="900" alt="SHAP Feature Importance" src="https://github.com/user-attachments/assets/693b6dad-97fa-442d-a501-9035acdf25bb" />
  <p><em>Figure 3: Feature sensitivity ranking quantified by baseline recall drop following channel zero-out perturbation.</em></p>
</div>

### 7.1 Quantitative Channel Influence

| Rank | Covariate Channel | Recall Drop (ΔRecall) | Physical Interpretation |
| :---: | :--- | :---: | :--- |
| 🥇 | **Prev Fire** | **+0.4114** | **Leading Indicator:** Flame front continuity; primary determinant of spatial propagation. |
| 🥈 | **NDVI** | **+0.2879** | **Fuel Availability:** Dense canopy and dry herbaceous biomass directly sustain combustion. |
| 🥉 | **Elevation** | **+0.2210** | **Topographic Slope:** Preheating of uphill fuels accelerates flame advancement. |
| 4 | **Min Temp** | **+0.0838** | Elevated nocturnal minimums hinder night fuel moisture recovery. |
| 5 | **ERC** | **+0.0745** | Energy Release Component reflects core fuel bed dryness. |
| 6 | **Max Temp** | **+0.0332** | Thermal drying during diurnal peak solar irradiance. |
| 7 | **Wind Dir** | **+0.0061** | Directional vector steering the perimeter flanks. |
| 8 | **Pop Density** | **-0.0002** | Negligible immediate influence on active flame front velocity. |
| 9 | **Precipitation** | **-0.0009** | Sparse daily effect during active, high-intensity dry-season burns. |
| 10 | **Humidity** | **-0.0057** | Secondary atmospheric dampening variable. |
| 11 | **Wind Speed** | **-0.0209** | Coarse resolution introduces localized noise; slight regularization upon ablation. |
| 12 | **Drought (PDSI)** | **-0.0414** | Multi-day composite cadence behaves as a low-frequency regularizer. |

### 7.2 Explainability Findings

1. **Autoregressive Priority:** Previous burn state is the dominant predictor (ΔRecall = 0.4114). Ablating it causes an immediate drop exceeding 41% in detection sensitivity.
2. **Terrain-Fuel Dynamics:** Vegetative biomass (NDVI) and slope angle (Elevation) constitute the next most critical physical drivers, validating classical semi-empirical fire models.
3. **Observation Cadence:** Low-frequency composite variables (PDSI, 4 km Wind Speed) produce negative drops under daily perturbation, underscoring the need for higher-resolution weather modeling.

---

## 🛠️ 8. Tech Stack & Environment

- **Core Framework:** TensorFlow 2.x, Keras, Python 3.9+
- **Data Pipelines:** TFRecord Dataset API, NumPy, Pandas
- **Metrics & Evaluation:** Scikit-learn (`roc_curve`, `auc`, `confusion_matrix`, `precision_score`, `recall_score`)
- **Explainability Tools:** SHAP Channel Perturbation, Grad-CAM, PrettyTable
- **Visualization:** Matplotlib, Seaborn
