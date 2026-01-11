<div align="center">

# 🔬 VOLMO Evaluation Pipeline

**Vision-Language Model for Ophthalmology**  
*Automated Multi-Task Evaluation Framework*

<img src="assets/volmo_logo.png" alt="VOLMO Logo" width="200"/>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📋 Overview

A comprehensive evaluation pipeline for the **VOLMO** model, designed to assess performance across multiple ophthalmologic vision-language tasks. This framework provides automated inference, evaluation, and reporting for clinical AI applications.

### ✨ Key Features

- 🚀 **Automated Pipeline** - One-command execution for all tasks
- 📊 **Multi-Task Support** - Binary classification, staging, image description, and clinical assessment
- 🎯 **GPU Optimized** - Efficient inference with configurable GPU allocation
- 📈 **Comprehensive Metrics** - Accuracy, BERTScore F1, SBERT similarity
- 📝 **Detailed Reports** - Timestamped markdown reports with results breakdown
- 📦 **Self-Contained** - Includes InternVL model code (no external dependencies)

---

## 🚀 Quick Start

### Prerequisites

```bash
conda activate internvl_rl
```

### Run Complete Evaluation

```bash
python launch.py
```

The pipeline will automatically:
- ✅ Load VOLMO model on GPU 3
- ✅ Process all 4 task categories
- ✅ Evaluate 24 datasets across tasks
- ✅ Generate timestamped report: `volmo_evaluation_results/report_YYYYMMDDHHMMSS.md`

---

## 📁 Project Structure

```
volmo_repo/
├── 🚀 launch.py                    # Main execution script
├── 📦 requirements.txt             # Python dependencies
├── 📖 README.md                    # Documentation (this file)
│
├── 🧠 core/                        # Core inference & evaluation
│   ├── inference/
│   │   └── inference_runner.py    # VOLMO model inference
│   └── evaluators/
│       ├── bool_evaluator.py      # Binary classification metrics
│       ├── stage_evaluator.py     # Staging classification metrics
│       ├── imgdesc_evaluator.py   # Image description metrics
│       └── assessplan_evaluator.py # Clinical assessment metrics
│
├── ⚙️  configs/                    # Task configurations
│   ├── eval_settings_bool.yaml
│   ├── eval_settings_stage.yaml
│   ├── eval_settings_imgdesc.yaml
│   └── eval_settings_assessplan.yaml
│
├── 📊 datasets/                    # Curated datasets
│   ├── bool/                      # Binary classification (17 datasets)
│   ├── stage/                     # Disease staging (4 datasets)
│   ├── imgdesc/                   # Image descriptions (1 dataset)
│   └── assessplan/                # Clinical assessments (2 datasets)
│
├── 💾 data/                        # Self-contained data (22GB total)
│   └── images/                    # All images organized by source
│       ├── brset/                 # BRSET dataset (4.2GB)
│       ├── fives/                 # FIVES glaucoma (117MB)
│       ├── bento/                 # Bento annotated (31MB)
│       ├── pmc_eyes/              # PMC-Eyes (6.1MB)
│       ├── oimhs/                 # OIMHS OCT (185MB)
│       ├── eyepacs/               # EyePACS DR (9.0GB)
│       ├── sydney_innovation/     # Sydney Innovation (8.0GB)
│       └── sustech/               # SuSTech (469MB)
│
├── 🤖 models/                      # Model checkpoints
│   └── volmo_2b_stage_3/
│       └── volmo_2b_stage_3/       # VOLMO 2B with InternVL code
│           ├── modeling_internvl_chat.py  # InternVL chat model
│           ├── modeling_intern_vit.py     # Vision transformer
│           ├── modeling_internlm2.py      # Language model
│           ├── configuration_*.py         # Model configs
│           ├── tokenization_internlm2.py  # Tokenizer
│           ├── conversation.py            # Chat templates
│           └── model.safetensors          # Model weights (30GB)
│
├── 🎨 assets/                      # Media resources
│   └── volmo_logo.png             # Project logo
│
└── 📈 volmo_evaluation_results/    # Evaluation outputs
    ├── bool/                      # Binary task results
    ├── stage/                     # Staging results
    ├── imgdesc/                   # Description results
    ├── assessplan/                # Assessment results
    └── report_YYYYMMDDHHMMSS.md   # Unified reports
```

---

## 🎯 Supported Tasks

<table>
<tr>
<td width="50%">

### 🔵 Binary Classification
Diagnose presence/absence of conditions:
- Age-Related Macular Degeneration (AMD)
- Diabetic Retinopathy (DR) - Multiple grading systems
- Hypertensive Retinopathy
- Increased Cup-Disc Ratio (ICDR)
- Choroidal Nevus
- Macular Edema
- Myopic Fundus Changes
- Retinal Detachment
- Retinal Drusens
- Retinal Hemorrhage
- Retinal Scar
- Retinal Vascular Occlusion
- Glaucoma

**Datasets:** 17 (14 BRSET + 3 UK Biobank)  
**Metric:** Accuracy

</td>
<td width="50%">

### 📊 Stage Classification
Multi-class disease severity grading (0-4):
- EyePACS (DR staging)
- MH Stage (Macular hole staging)
- Sydney Innovation
- SuSTech SYSU

**Metric:** Accuracy

</td>
</tr>
<tr>
<td width="50%">

### 📝 Image Description
Natural language generation of findings:
- PMC Serina Dataset

**Metrics:** BERTScore F1, SBERT Similarity

</td>
<td width="50%">

### 🏥 Assessment & Plan
Clinical decision support:
- PMC Clinical Procedures

**Metrics:** BERTScore F1, SBERT Similarity

</td>
</tr>
</table>

---

## ⚙️ Configuration

Customize evaluation settings by editing YAML files in `configs/`:

```yaml
# Example: configs/eval_settings_bool.yaml
task: classification_bool

model:
  model_path: models/volmo_2b_stage_3
  input_size: 448
  max_num: 6

output_dir: volmo_evaluation_results/bool

data_paths:
  brset_amd: datasets/bool/brset_amd.json
  brset_dr: datasets/bool/brset_dr.json
  # ...
```

**Configurable Parameters:**
- 🎯 Dataset paths and selections
- 🤖 Model architecture settings
- 💾 Output directory structure
- 🔧 Task-specific hyperparameters

---

## 📊 Datasets

| Category | Dataset | Samples | Description |
|----------|---------|---------|-------------|
| **Binary** | BRSET AMD | 108 | Age-related macular degeneration |
| | BRSET DR | 442 | Diabetic retinopathy |
| | BRSET Scottish DR | ~400 | Scottish DR grading system |
| | BRSET Hypertension | 102 | Hypertensive retinopathy |
| | BRSET ICDR | 1,322 | Increased cup-disc ratio |
| | BRSET International Clinical DR | ~300 | International clinical DR |
| | BRSET Nevus | 56 | Choroidal nevus |
| | BRSET Macular Edema | 142 | Macular edema |
| | BRSET Myopic | ~200 | Myopic fundus changes |
| | BRSET Detachment | ~150 | Retinal detachment |
| | BRSET Drusens | ~180 | Retinal drusens |
| | BRSET Hemorrhage | ~200 | Retinal hemorrhage |
| | BRSET Scar | ~120 | Retinal scar |
| | BRSET Occlusion | ~180 | Retinal vascular occlusion |
| | UK Biobank AMD | 400 | AMD screening |
| | UK Biobank DR | 241 | DR screening |
| | UK Biobank Glaucoma | 400 | Glaucoma screening |
| **Staging** | EyePACS | 8,871 | DR severity grading |
| | MH Stage | 900 | Macular hole staging |
| | Sydney Innovation | 7,982 | DR classification |
| | SuSTech SYSU | 1,151 | DR grading |
| **Description** | PMC Serina | 41 | Image descriptions |
| **Assessment** | PMC Case Reports | 89 | Clinical procedures |

**Total:** 24 datasets, ~25,000+ samples

---

## 📈 Evaluation Metrics

### Classification Tasks
- **Accuracy**: Proportion of correct predictions
- Computed per-dataset and aggregated

### Text Generation Tasks
- **BERTScore F1**: Semantic similarity using BERT embeddings
- **SBERT Similarity**: Sentence-level semantic similarity
- Both metrics range from 0.0 to 1.0 (higher is better)

---

## 🛠️ Development

### Environment Setup

```bash
conda create -n internvl_rl python=3.9
conda activate internvl_rl
pip install -r requirements.txt
```

### Adding New Tasks

1. Create evaluator in `core/evaluators/`
2. Add configuration in `configs/`
3. Update `launch.py` task list
4. Add dataset to `datasets/`

### GPU Configuration

By default, the pipeline uses **GPU 3**. To change:

```python
# In launch.py, modify:
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # Change to your GPU ID
```

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

## 🔬 Research

This evaluation framework is part of ongoing research in vision-language models for ophthalmology. For more details on the VOLMO model architecture and training methodology, please refer to our publications.

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📧 Contact

For questions or collaboration inquiries, please contact the research team.

---

## 🎁 Self-Contained Repository

This repository is **fully self-contained** with all necessary components:

### 📊 Data (22GB)
- ✅ **24,112 images** organized in `data/images/`
- ✅ All JSON dataset files with local paths
- ✅ No external dependencies on shared datasets
- ✅ Sources: BRSET, FIVES, EyePACS, Sydney, SuSTech, OIMHS, PMC-Eyes, Bento

### 🤖 Model (30GB)
- ✅ **InternVL model code** included in `models/volmo_2b_stage_3/volmo_2b_stage_3/`
- ✅ Complete model implementation (vision + language)
- ✅ Model weights (model.safetensors)
- ✅ No need to download from external sources
- ✅ Uses `trust_remote_code=True` for custom model loading

### 🔧 Framework
- ✅ Custom inference runner
- ✅ Task-specific evaluators
- ✅ Configuration management
- ✅ Automated reporting

To verify data integrity:
```bash
python scripts/copy_images_and_update_paths.py --dry-run
```

**Total Size:** ~52GB (30GB model + 22GB data)  
**Ready to distribute and run anywhere!**

---

<div align="center">

**Built with ❤️ for advancing AI in ophthalmology**

*Automated evaluation pipeline for medical vision-language models*

</div>
