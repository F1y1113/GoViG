# 🧭 GoViG: Goal-Conditioned Visual Navigation Instruction Generation
[![arXiv](https://img.shields.io/badge/arXiv-2503.14229-red)]() 
![License](https://img.shields.io/badge/license-MIT-blue)

* 🚀 [Project Web Page](Coming soon)
* 📂 [Dataset](Coming soon)


GoViG introduces a new task in embodied AI: generating navigation instructions directly from egocentric visual observations of the initial and goal states. Unlike previous methods that rely on semantic maps or structured annotations, GoViG operates purely on egocentric visual input—making it highly adaptable to unseen and unstructured environments.

## 🔍 Overview

GoViG decomposes the instruction generation task into two interconnected subtasks:

- **Navigation Visualization**  
  Predicts intermediate visual states that bridge the initial and goal views.

- **Instruction Generation with Visual Cues**  
  Synthesizes linguistically coherent and spatially grounded instructions based on both observed and anticipated visuals.

These components are unified within an autoregressive MLLM, trained with tailored objectives to ensure spatial accuracy and linguistic clarity.

## 🧠 Reasoning Strategies

Inspired by human navigation behavior, GoViG supports two multimodal reasoning paradigms:

- **One-Pass Reasoning**: Generates instructions in a single forward pass.
- **Interleaved Reasoning**: Alternates between visual prediction and language generation for incremental planning.

## 📦 Dataset: R2R-Goal

To evaluate GoViG, we introduce **R2R-Goal**, a dataset combining synthetic and real-world trajectories.


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
