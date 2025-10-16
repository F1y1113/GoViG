<br>
<p align="center">

<h1 align="center"><strong>🧭 GoViG: Goal-Conditioned <br>Visual Navigation Instruction Generation</strong></h1>
  <p align="center"><span><a href=""></a></span>
              <a>Fengyi Wu<sup>#</sup>,</a>
             <a>Yifei Dong<sup>#</sup>,</a>
              <a>Zhi-Qi Cheng<sup>*</sup>,</a>
              <a>Yilong Dai,</a>
              <a>Guangyu Chen,</a>
              <a>Hang Wang,</a>
              <a>Qi Dai,</a>
              <a>Alexander G Hauptmann</a>
    <br>
    <sup>#</sup>Equal Contribution, Random Order, <sup>2</sup>Corresponding author<br>
  </p>
    
<p align="center">
  <a href="https://arxiv.org/abs/2508.09547" target="_blank">
    <img src="https://img.shields.io/badge/ArXiv-2508.09547-red">
  </a>
  <a href="https://github.com/F1y1113/GoViG" target="_blank">
    <img src="https://img.shields.io/badge/Project-GoViG-blue">
  </a>
    <a href="https://drive.google.com/file/d/17823NqgVHJ-xfx5bFTcm58g6j7D19Hqq/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Googledrive-dataset-purple">
  </a>
<a href="https://github.com/F1y1113/GoViG" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green">
</a>
</p>

<p align="center">
  <img src="assists/task_overview.png" alt="task" width="500"/>
</p>

**GoViG** introduces a new task in embodied AI: generating navigation instructions directly from egocentric visual observations of the initial and goal states. Unlike previous methods that rely on semantic maps or structured annotations, GoViG operates purely on egocentric visual input—making it highly adaptable to unseen and unstructured environments.


## 🔍 Overview
<p align="center">
  <img src="assists/method_overview.png" alt="task" width="1000"/>
</p>

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

### Data

We release a partial dataset for the purpose of debugging and demonstrating the data format, you can find them in [data_samples](data_samples/).
And you can access the full dataset [here](https://drive.google.com/file/d/17823NqgVHJ-xfx5bFTcm58g6j7D19Hqq/view?usp=sharing)

```bash
unzip R2R_Goal.zip
```

### Training

```bash
bash train.sh
```

### Evaluation

```bash
bash eval.sh
```

you can find detailed metrics calculation in [taskeval_vis.py](taskeval_vis.py).

## Acknowledgement

We would like to thank [ANOLE](https://arxiv.org/abs/2407.06135) and [MVOT](https://arxiv.org/abs/2501.07542) for their publicly available codebase, which we referenced during the implementation of Anole training.


## 🧭 GoViG Gallery

<div style="overflow-x: auto; white-space: nowrap;">

  <table style="table-layout: fixed; width:2000px;">
    <thead>
      <tr>
        <th style="width:180px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initial View&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th style="width:180px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Goal View&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th style="width:180px;">Trajectory (1P)</th>
        <th style="width:180px;">Instructions (1P)</th>
        <th style="width:180px;">Trajectory (Int)</th>
        <th style="width:180px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Instructions (Int)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><img src="assists/seen/example1/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/seen/example1/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/seen/example1/onepass/traj1.gif" style="width:180px;"></td>
        <td>Stop in the doorway.</td>
        <td><img src="assists/seen/example1/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Stop in front of the last door on your right.<br>Then take a slight left turn to go towards the bathroom.<br>After you leave the kitchen and go through the double doors, keep going and go into the living room.<br>Turn left at the first door past the oven and continue down the hallway.<br>Go into the powder room that is straight ahead.<br>Walk past the bathroom door.</td>
      </tr>
      <tr>
        <td><img src="assists/seen/example2/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/seen/example2/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/seen/example2/onepass/traj1.gif" style="width:180px;"></td>
        <td>Walk into the bedroom.</td>
        <td><img src="assists/seen/example2/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Walk out of the bedroom using the door on your right.<br>Walk out of bedroom and turn right.<br>Leave the bedroom.<br>Turn to your right and go outside.<br>Exit the room.<br>Exit bedroom through doorway on the right.</td>
      </tr>
      <tr>
        <td><img src="assists/seen/example3/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/seen/example3/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/seen/example3/onepass/traj1.gif" style="width:180px;"></td>
        <td>Across the kitchen.</td>
        <td><img src="assists/seen/example3/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Exit the kitchen.<br>Turn right at the counter.<br>Walk past kitchen island.<br>Turn past the sink, and in front of the oven to your left.<br>Make a left immediately through the kitchenette, then turn right into the hallway.<br>Walk past the sink.</td>
      </tr>
      <tr>
        <td><img src="assists/seen/example4/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/seen/example4/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/seen/example4/onepass/traj1.gif" style="width:180px;"></td>
        <td>Go through the door.</td>
        <td><img src="assists/seen/example4/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Straight through the bedroom with the lamp.<br>Turn left and wait in the doorway.<br>Stop in the bedroom doorway.<br>Then turn right and wait in bedroom at the end of the hall.<br>Stop in the doorway.<br>Turn slight left, continue straight. Turn slight left, stop at bed.</td>
      </tr>
      <tr>
        <td><img src="assists/unseen/example1/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example1/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example1/onepass/traj1.gif" style="width:180px;"></td>
        <td>Walk out of the kitchen.</td>
        <td><img src="assists/unseen/example1/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Walk through the kitchen stop at the oven.<br>Continue walking straight down the kitchen.<br>Turn left, walk down the kitchen hallway.<br>Turn left and enter kitchen.<br>Walk and stop right before washing area.<br>Turn right and continue down the hall until you get to a refrigerator.</td>
      </tr>
      <tr>
        <td><img src="assists/unseen/example2/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example2/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example2/onepass/traj1.gif" style="width:180px;"></td>
        <td>Walk past the room on the left.</td>
        <td><img src="assists/unseen/example2/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Walk past the door directly across from you.<br>Continue straight and continue through a second set of double doors.<br>Pass the wall on the right.<br>Turn left and enter kitchen.<br>Go down the hall into the office on the left.<br>Walk to the end of the hall and through the open door.</td>
      </tr>
      <tr>
        <td><img src="assists/unseen/example3/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example3/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example3/onepass/traj1.gif" style="width:180px;"></td>
        <td>Walk up stairs, turn right, continue up stairs</td>
        <td><img src="assists/unseen/example3/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Walk up stairs.<br>Go up the stairs.<br>Walk straight ahead passed the stairs.<br>Go up the stairs.<br>Go up three steps then wait at the top.<br>Go all of the way up the stairs.</td>
      </tr>
      <tr>
        <td><img src="assists/unseen/example4/onepass/initial_obs_0.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example4/onepass/goal_obs.png" style="width:180px;"></td>
        <td><img src="assists/unseen/example4/onepass/traj1.gif" style="width:180px;"></td>
        <td>Walk past the room on the left.</td>
        <td><img src="assists/unseen/example4/interleaved/traj1.gif" style="width:180px;"></td>
        <td>Stop in entryway of house.<br>Stop at sliding barn door.<br>Wait near the patio.<br>Turn to the front row of couches is showing and walk over to the patio. Wait in the doorway to the patio.<br>Walk straight besides the wooden tables.<br>Stop when you reach the sliding glass doors.</td>
      </tr>
    </tbody>
  </table>
</div>

More examples of GoViG results on the **Real-world** Subset of our **R2R-Goal** dataset.

<p align="center">
  <img src="assists/real.png" alt="real" width="1000"/>
</p>


## 🌟 Citation

If you find this repository or our paper useful, please consider **starring** this repository and **citing** our paper:
```bibtex
@article{wu2025govig,
  title={GoViG: Goal-Conditioned Visual Navigation Instruction Generation},
  author={Wu, Fengyi and Dong, Yifei and Cheng, Zhi-Qi and Dai, Yilong and Chen, Guangyu and Wang, Hang and Dai, Qi and Hauptmann, Alexander G},
  journal={arXiv preprint arXiv:2508.09547},
  year={2025}
}
```
