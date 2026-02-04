# Thesis
# End-to-End Deep Learning and Machine Learning Pipeline for Diffuse Glioma Segmentation and Grading Using Brain MRI

This repository contains the complete implementation of a bachelor thesis project focused on the automated analysis of diffuse gliomas from multi-modal brain MRI. The project proposes and evaluates an end-to-end pipeline that integrates deep learning–based tumor segmentation with classical machine learning–based glioma grading, reflecting realistic clinical workflows.

## Project Overview

Diffuse gliomas are highly infiltrative primary brain tumors that require accurate tumor localization and grading for effective clinical decision-making. This project addresses these challenges by developing a two-stage pipeline:

1. Deep learning–based tumor segmentation using a 2D U-Net architecture.
2. Machine learning–based tumor grading (LGG vs HGG) using features extracted from tumor masks.

A key contribution of this work is the explicit comparison between two grading scenarios:
- **GT-mask setting (upper bound):** classification using expert-annotated tumor masks.
- **Pred-mask setting (realistic pipeline):** classification using automatically predicted tumor masks.


## Repository Structure

Thesis-/
├── data_indexing/            # Patient indexing (modalities, geometry, labels)
├── data_audit/               # Validation and consistency checks
├── indices/                  # Generated patient index CSV files
├── metadata/                 # Clinical metadata (WHO grade)
├── monai_pipeline/           # Deep learning pipeline (segmentation)
│   ├── data/                 # Dataset building, transforms, dataloaders
│   ├── training/             # 2D U-Net training (5-fold CV)
│   ├── evaluation/           # Full-volume evaluation scripts
│   └── visualization/        # GT vs Pred overlays
├── ml_pipeline/              # Machine learning pipeline (grading)
├── notebooks/                # Evaluation and visualization notebooks
├── outputs/                  # Models, metrics, overlays, figures
└── requirements.txt


## Dataset

### Dataset Source

This project uses the UCSF-PDGM v5 dataset, a publicly available and well-established dataset for diffuse glioma research. The dataset was released as part of an official research initiative and competition and represents secondary data only.
The link to the dataset "https://www.cancerimagingarchive.net/collection/ucsf-pdgm/".

- Data type: Secondary data
- Ethics: No new data collection performed

### MRI Modalities

Each patient includes four MRI modalities:
- T1-weighted (T1)
- Contrast-enhanced T1-weighted (T1c)
- T2-weighted (T2)
- FLAIR

### Required Dataset Structure

After downloading the dataset, it must be placed and renamed exactly as follows:

Thesis-/
└── UCSF-PDGM/
└── UCSF-PDGM-v5/
├── UCSF-PDGM-0001_nifti/
├── UCSF-PDGM-0002_nifti/
└── …

Clinical metadata must be placed at:

Thesis-/metadata/UCSF-PDGM-metadata_v5.csv



## Environment Setup

### Python Environment

Python 3.10 or 3.11 is recommended.

Example (Windows):

python -m venv venv_train_py311
venv_train_py311\Scripts\activate
pip install -r requirements.txt

Key Libraries
	•	PyTorch – deep learning backend
	•	MONAI – medical imaging framework
	•	scikit-learn – machine learning classifiers
	•	NumPy / Pandas – data handling
	•	Matplotlib – visualization



## Step 1 - Patient Index Construction

This stage ensures that only valid, geometrically consistent patients with complete modalities and labels are used.

### 1.1. Modalities Indexing

python -m data_indexing.write_patient_index_modalities

### 1.2 Geometry and Mask Consistency Check

python -m data_indexing.write_patient_index_geometry_2

### 1.3 Grade Label Assignment (HGG / LGG)

python -m data_indexing.write_patient_index_labels_3

### 1.4 Final Training-Ready Index

python -m data_indexing.make_patient_index_final

Final output:

indices/patient_index_final.csv



## Step 2 - Deep Learning Tumor Segmentation

### 2.1 Build MONAI Dataset JSON

python -m monai_pipeline.data.build_dataset

Output:

outputs/dataset_final_index.json

### 2.2 DataLoader Sanity Check

python -m monai_pipeline.data.dataloader_cv

### 2.3 Train 2D U-Net (5-Fold Cross-Validation)

python -m monai_pipeline.training.train_unet_2d_cv

Outputs include:

outputs/unet2d_cv/fold_*/best_model.pt
outputs/unet2d_cv/cv_summary.json



## Step 3 - Full-Volume Segmentation Evaluation

### 3.1 Full-Volume Dice Evaluation

python -m monai_pipeline.evaluation.eval_full_volume_dice


### 3.2 Additional Metrics

python -m monai_pipeline.evaluation.eval_full_volume_metrics

### 3.3 GT vs Pred Overlay Visualization

python -m monai_pipeline.visualization.make_overlays_ranked

Outputs:

outputs/overlays_ranked/


### Step 4 - Machine Learning Glioma Grading

### 4.1 Feature Extraction from Ground-Truth Masks

python -m ml_pipeline.extract_features

Output:

outputs/ml_features/features_from_gtmask.csv

### 4.2 Train ML Classifiers (GT Mask)

python -m ml_pipeline.train_classifier_cv

### 4.3 Feature Extraction from Predicted Masks

python -m ml_pipeline.extract_features_from_predmask

### 4.4 Train ML Classifiers (Pred Mask)

python -m ml_pipeline.train_classifier_predmask


## Notebooks

Recommended execution order:

	1.	notebooks/01_gt_pred_overlays.ipynb (Qualitative segmentation assessment)

	2.	notebooks/02_ml_evaluation_gt_vs_predmask.ipynb (Quantitative machine learning comparison)


## Outputs

All results are saved under:

outputs/

Including:

	•	trained models
	•	evaluation metrics (JSON)
	•	feature tables
	•	overlay images
	•	exported figures


## Notes

	•	Patch-based training is used for computational efficiency
	•	Full-volume reconstruction is applied during inference
	•	Class imbalance (HGG ≫ LGG) is inherent to the dataset
	•	The pipeline is Windows-safe and GPU-optional

