# SRIP 2026 — AI for Sustainability: Earth Observation

This is my submission for the Earth Observation task under the SRIP 2026 AI for Sustainability program. The goal was to build a pipeline that identifies land use patterns over the Delhi-NCR region using satellite imagery and land cover data.

---

## What the Project Does

Given a set of Sentinel-2 satellite images over Delhi-NCR and an ESA WorldCover land cover raster, I built a full pipeline that:

- Draws a spatial grid over Delhi-NCR and filters only the relevant satellite images
- Labels each image based on the dominant land cover type in a 128×128 patch
- Trains a ResNet18 CNN to classify land use from satellite images

---

## Dataset

All data is from the Kaggle dataset: **Earth Observation Delhi Airshed**

| File | Description |
|------|-------------|
| `delhi_ncr_region.geojson` | Boundary shapefile of Delhi-NCR region |
| `delhi_airshed.geojson` | Boundary of the Delhi Airshed region |
| `worldcover_bbox_delhi_ncr_2021.tif` | ESA WorldCover 2021 land cover raster (10m resolution) |
| `rgb/` | Sentinel-2 RGB image patches (128×128 px, filename = `lat_lon.png`) |

---

## Questions Solved

### Q1 — Spatial Reasoning & Data Filtering

- Loaded the Delhi-NCR boundary using GeoPandas
- Plotted the boundary using matplotlib
- Reprojected to EPSG:32644 to create a proper 60×60 km grid in meters
- Overlaid the grid on the map
- Parsed lat/lon from image filenames and filtered using spatial join

**Results:**
- Images before filtering: **9,216**
- Images after filtering (inside Delhi-NCR): **8,015**

**Plots generated:**
- `delhi_boundary.png` — boundary only
- `delhi_grid.png` — boundary + red grid
- `delhi_grid_final.png` — boundary + grid + green image points

---

### Q2 — Label Construction & Dataset Preparation

- Opened the WorldCover raster using Rasterio
- For each filtered image, located the corresponding 128×128 patch in the raster using the image's center coordinate
- Assigned the label using the mode (most frequent class) in the patch
- Mapped ESA class codes to simplified categories
- Dropped classes with fewer than 10 samples (Permanent water bodies: 7, Herbaceous wetland: 2)
- Applied a 60/40 stratified train-test split

**Final classes (5):**

| Class | Train Count | Test Count |
|-------|-------------|------------|
| Cropland | ~3,290 | ~2,190 |
| Built-up | ~1,067 | ~711 |
| Tree cover | ~204 | ~137 |
| Shrubland | ~148 | ~99 |
| Grassland | ~99 | ~66 |

**Plots generated:**
- `class_distribution.png` — train/test class distribution bars

---

### Q3 — Model Training & Supervised Evaluation

- Used pretrained ResNet18 with the final layer replaced for 5-class output
- Applied data augmentation on training set: random flips, rotation, color jitter
- Used class-weighted CrossEntropyLoss to handle class imbalance
- Trained for 25 epochs with Adam optimizer and ReduceLROnPlateau scheduler
- Saved the best model based on highest Macro F1

**Final Results on Test Set:**

| Metric | Score |
|--------|-------|
| Accuracy | 91.32% |
| Macro F1 | 0.7730 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cropland | 0.98 | 0.93 | 0.95 |
| Built-up | 0.86 | 0.92 | 0.89 |
| Shrubland | 0.75 | 0.91 | 0.82 |
| Tree cover | 0.63 | 0.80 | 0.71 |
| Grassland | 0.44 | 0.55 | 0.49 |

**Brief Interpretation:**
Cropland and Built-up performed strongly — they are the dominant classes with the most training examples. Grassland had the lowest F1 (0.49) mainly because it has very few samples and visually overlaps with Cropland in satellite imagery, making it harder to distinguish.

**Plots generated:**
- `confusion_matrix.png`
- `training_curves.png`

---

## How to Run

1. Open `solution.ipynb` in Google Colab
2. Download the dataset from Kaggle (instructions in the first cell)
3. Run all cells from top to bottom

> GPU runtime recommended for Q3: Runtime → Change runtime type → T4 GPU

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `geopandas` | Shapefile loading and spatial operations |
| `rasterio` | Reading the WorldCover raster |
| `shapely` | Creating grid geometries |
| `pytorch` + `torchvision` | ResNet18 training |
| `scikit-learn` | Metrics, label encoding, train-test split |
| `matplotlib` + `seaborn` | All visualizations |

---

## AI Tool Disclosure

I used Claude (Anthropic) for guidance on code structure and debugging during this assignment. All code has been reviewed and is fully understood. I can explain any part of it during the one-on-one discussion.
