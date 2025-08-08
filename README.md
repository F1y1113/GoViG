# GoViG: Goal-Conditioned Visual Navigation Instruction Generation
[![arXiv](https://img.shields.io/badge/arXiv-2503.14229-red)]() 
![License](https://img.shields.io/badge/license-MIT-blue)

* 🚀 [Project Web Page](Coming soon)
* 📂 [Dataset](Coming soon)

## Quick Start

```bash
conda create -n GoViG python=3.10
conda activate GoViG
pip install torch==2.4.0
pip install -r requirements.txt --user
```

## Implementation

### Data

We now release a partial dataset for the purpose of debugging and demonstrating the data format. You can find them in [data_samples](data_samples/)

### Training

```bash
bash train.sh
```

### Evaluation

```bash
bash eval.sh
```

you can find detailed metrics calculation in [taskeval_vis.py](taskeval_vis.py).