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

## 📧 Contact

For questions or collaboration inquiries, please email:
- Zhenyue Qin: kf.zy.qin@gmail.com
- Qingyu Chen: qingyu.chen@yale.edu

---

<div align="center">

**Built with ❤️ for exploring AI's potential in ophthalmology**

*VOLMO: Versatile and Open Large Models for Ophthalmology*

</div>
