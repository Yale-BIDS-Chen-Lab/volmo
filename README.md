<div align="center">

# VOLMO

**Versatile and Open Large Models for Ophthalmology**  
*Official Repository*

<img src="assets/volmo_logo.png" alt="VOLMO Logo" width="200"/>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](LICENSE)

</div>

---

## 📋 Overview

Official repository of **VOLMO: Versatile and Open Large Models for Ophthalmology**.

### ✨ Key Features

- 🤖 **Models** - Pre-trained VOLMO models for ophthalmologic vision-language tasks
- 📊 **Data for Evaluation** - Evaluation datasets for model assessment

---

## 🤖 Models

VOLMO-2B is a 2-billion parameter mulimodal large language model. The pre-trained model weights are available on Hugging Face.

### Download Model

```python
from huggingface_hub import snapshot_download

# Download VOLMO-2B model
model_path = snapshot_download(
    repo_id="Yale-BIDS-Chen/VOLMO-2B",
    local_dir="./models/volmo_2b_stage_3"
)
```

**Hugging Face Repository**: [Yale-BIDS-Chen/VOLMO-2B](https://huggingface.co/Yale-BIDS-Chen/VOLMO-2B)

For detailed model documentation, see [models/README.md](models/README.md).

---

## 📊 Data

VOLMO evaluation data includes curated datasets across four task categories: binary classification, disease staging, image description, and clinical assessment.

### Download Data

```python
from huggingface_hub import snapshot_download

# Download evaluation datasets
data_path = snapshot_download(
    repo_id="Yale-BIDS-Chen/VOLMO-Evaluation-Data",
    repo_type="dataset",
    local_dir="./data"
)
```

**Hugging Face Repository**: [Yale-BIDS-Chen/VOLMO-Evaluation-Data](https://huggingface.co/datasets/Yale-BIDS-Chen/VOLMO-Evaluation-Data)

For detailed data documentation, see [data/README.md](data/README.md).

---

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n volmo python=3.9
conda activate volmo

# Install requirements
pip install -r requirements.txt
```

### Run Complete Evaluation

```bash
python launch.py
```

The pipeline will automatically:
- ✅ Load VOLMO model
- ✅ Process all 4 task categories: binary classification, staging, image description, and clinical assessment
- ✅ Generate timestamped report: `volmo_evaluation_results/report_YYYYMMDDHHMMSS.md`

---

## 📄 Output Reports

Reports are generated in `volmo_evaluation_results/` with the following structure:

```markdown
# VOLMO Evaluation Report
**Generated:** 2026-01-10 16:45:23

## Binary Classification
**Status:** 9/9 datasets completed

### BRSET AMD
- **Accuracy:** 0.9259
- **Samples:** 108

### BRSET DR
- **Accuracy:** 0.8914
- **Samples:** 442
...
```

---

## 📝 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## ⚠️ Disclaimer

The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website

---

## 📧 Contact

For questions or collaboration inquiries, please email:
- Zhenyue Qin: kf.zy.qin@gmail.com
- Qingyu Chen: qingyu.chen@yale.edu

---

<div align="center">

**Built with ❤️ for exploring AI's potential in ophthalmology**

*VOLMO: Versatile and Open Large Models for Ophthalmology*

</div>
