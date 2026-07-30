# Physiology-Constrained Learning (PCL)

This repository contains the implementation for **Physiology-Constrained Learning (PCL)**, a representation learning framework that enforces hemodynamic and biochemical laws (e.g., Mean Arterial Pressure, Henderson-Hasselbalch) during masked transformer pretraining to improve OOD generalization in clinical time-series.

## Quick Start

The entire experimental pipeline is managed by `run_paper_experiments.py`.

### 1. Environment Setup

We recommend using Python 3.13. Ensure you have a GPU for production runs.

```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  # Linux
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments (Test Mode)

Use this to verify the pipeline is working. It uses 15% of the data and a small model. It should finish in ~5-10 minutes.

```bash
# Linux/Ubuntu
PCL_TEST_MODE=1 python3 run_paper_experiments.py

# Windows PowerShell
$env:PCL_TEST_MODE=1; python run_paper_experiments.py
```

### 3. Run Experiments (Production Mode)

Use this to generate the final paper results. It uses 100% of the data and the full 6-layer architecture. **Warning: This takes several hours.**

```bash
# Linux/Ubuntu
PCL_TEST_MODE=0 python3 run_paper_experiments.py
```

## 📂 Project Structure

- `src/losses/pcl_loss.py`: Core PCL implementation (Physical laws).
- `src/training/train_utils.py`: Pretraining logic and Loss Balancer.
- `src/eval/evaluate_utils.py`: Finetuning and OOD scoring.
- `config.py`: Global hyperparameters and paths.
