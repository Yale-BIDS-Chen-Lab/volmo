<div align="center">

# VOLMO Evaluation Data

**Versatile and Open Large Models for Ophthalmology**  
*Evaluation Data*

<img src="../assets/volmo_logo.png" alt="VOLMO Logo" width="200"/>
<!-- <img src="https://huggingface.co/Yale-BIDS-Chen/VOLMO-Evaluation-Data/resolve/main/assets/volmo_logo.png" alt="VOLMO Logo" width="200"/> -->

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VOLMO--Evaluation--Data-yellow)](https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data)

</div>

---

## 📋 Overview

Evaluation data of **VOLMO: Versatile and Open Large Models for Ophthalmology**.

### ✨ Dataset Features

- 📸 **30GB+ Images** - Retinal fundus photos, OCT scans, and clinical imaging
- 🎯 **4 Task Categories** - Binary classification, staging, description, and clinical assessment
- 📝 **24 Benchmark Datasets** - Standardized evaluation across multiple conditions
- 🔬 **Clinical Annotations** - Expert-involved labels and descriptions

---

## 🚀 Download Data

The VOLMO evaluation datasets are hosted on Hugging Face:

### Option 1: Using Hugging Face Datasets (Recommended)

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("Yale-BIDS-Chen/VOLMO-Evaluation-Data")
```

### Option 2: Using Hugging Face Hub

```python
from huggingface_hub import snapshot_download

# Download entire dataset
data_path = snapshot_download(
    repo_id="Yale-BIDS-Chen/VOLMO-Evaluation-Data",
    repo_type="dataset",
    local_dir="./data"
)
```

### Option 3: Using Git LFS

```bash
# Install git-lfs if not already installed
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data
```

### Option 4: Manual Download

Visit [https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data](https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data) to download individual files.

---

## 📁 Data Structure

```
data/
├── datajsons/                     # Dataset annotations (JSON format)
│   ├── bool/                      # Binary classification (17 datasets)
│   │   ├── brset_amd.json
│   │   ├── brset_dr.json
│   │   ├── biobank_amd_tianyi.json
│   │   └── ...
│   ├── stage/                     # Disease staging (4 datasets)
│   │   ├── eyepacs.json
│   │   ├── mhstage.json
│   │   └── ...
│   ├── imgdesc/                   # Image descriptions (1 dataset)
│   │   └── pmc_serina.json
│   └── assessplan/                # Clinical assessments (1 dataset)
│       └── pmc_casereports.json
│
└── images/                        # Image files (30GB total)
    ├── brset/                     # BRSET dataset (4.2GB)
    ├── biobank/                   # UK Biobank (24GB)
    ├── eyepacs/                   # EyePACS DR (9.0GB)
    ├── sydney_innovation/         # Sydney Innovation (8.0GB)
    ├── fives/                     # FIVES glaucoma (117MB)
    ├── sustech/                   # SuSTech (469MB)
    ├── oimhs/                     # OIMHS OCT (185MB)
    ├── pmc_casereports/           # PMC case reports (31MB)
    └── pmc_eyes/                  # PMC-Eyes (6.1MB)
```

---

## 🎯 Task Categories

### 🔵 Binary Classification (17 Datasets)

Detection of presence/absence for various retinal conditions:

| Dataset | Samples | Condition |
|---------|---------|-----------|
| BRSET AMD | 108 | Age-Related Macular Degeneration |
| BRSET DR | 442 | Diabetic Retinopathy |
| BRSET Hypertension | 102 | Hypertensive Retinopathy |
| BRSET ICDR | 1,322 | Increased Cup-Disc Ratio |
| BRSET Nevus | 56 | Choroidal Nevus |
| BRSET Macular Edema | 142 | Macular Edema |
| UK Biobank AMD | 400 | AMD Screening |
| UK Biobank DR | 241 | DR Screening |
| UK Biobank Glaucoma | 400 | Glaucoma Screening |
| ... | ... | ... |

**Total**: ~5,000+ samples across 17 conditions

### 📊 Stage Classification (4 Datasets)

Multi-class disease severity grading:

| Dataset | Samples | Task |
|---------|---------|------|
| EyePACS | 8,871 | DR Severity (0-4) |
| MH Stage | 900 | Macular Hole Staging |
| Sydney Innovation | 7,982 | DR Classification |
| SuSTech SYSU | 1,151 | DR Grading |

**Total**: ~19,000 samples

### 📝 Image Description (1 Dataset)

Natural language descriptions of clinical findings:

- **PMC Serina**: 41 retinal images with detailed clinical descriptions

### 🏥 Assessment & Plan (1 Dataset)

Clinical decision support annotations:

- **PMC Case Reports**: 89 cases with assessment and treatment plans

---

## 📊 Data Format

### JSON Annotation Format

Each JSON file contains a list of samples with the following structure:

```json
{
  "image_paths": ["path/to/image.jpg"],
  "prompt": "Does this image show signs of diabetic retinopathy?",
  "GT": "Yes",
}
```

---

## 📈 Dataset Statistics

| Category | Datasets | Total Samples | Total Size |
|----------|----------|---------------|------------|
| Binary Classification | 17 | ~5,000 | ~28GB |
| Stage Classification | 4 | ~19,000 | ~17GB |
| Image Description | 1 | 41 | ~6MB |
| Assessment & Plan | 1 | 89 | ~31MB |
| **Total** | **24** | **~24,000+** | **~30GB** |

---

## 📝 License

This dataset is licensed according to the original licenses of data. 

---

## 🙏 Acknowledgments

This dataset compilation includes data from multiple sources:

- **BRSET**
- **OIMHS**
- **UK Biobank**
- **EyePACS**
- **Sydney Innovation**
- **FIVES**
- **PMC**
- etc...

We thank all original data providers and contributors.

---

## 📧 Contact

For questions or collaboration inquiries, please email:
- Zhenyue Qin: kf.zy.qin@gmail.com
- Qingyu Chen: qingyu.chen@yale.edu

---

## 🔗 Resources

- 🏠 **Main Repository**: [https://github.com/kfzyqin/volmo](https://github.com/Yale-BIDS-Chen-Lab/VOLMO)
- 🤗 **Model Weights**: [https://huggingface.co/Yale-BIDS-Chen/VOLMO-2B](https://huggingface.co/Yale-BIDS-Chen/VOLMO-2B)
- 📊 **This Dataset**: [https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data](https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data)

---

<div align="center">

**Built with ❤️ for exploring AI's potential in ophthalmology**

*VOLMO: Versatile and Open Large Models for Ophthalmology*

</div>
